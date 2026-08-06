# Rice_Segmentation 优化补丁

针对 `iakRulan/Rice_Segmentation`（线上 0.8674，目标 0.90+）的代码级改进。

## 一、先看清瓶颈在哪

从 `outputs/logs/*.json` 反解出的空图占比和分项 IoU：

| 类别 | 空图占比 | 空图 IoU | 非空图 IoU | 该类总 IoU |
|------|---------|---------|-----------|-----------|
| wheat | **49.7%** | 0.985 | 0.832 | 0.908 |
| rape  | **47.7%** | 0.987 | 0.781 | 0.879 |
| rice  | 21.7% | 0.965 | 0.817 | 0.850 |

（取 `v3_sweep.json`，即未用 val 训练的干净模型。）

**边际收益计算：**

| 方向 | 提升幅度 | 对 MEAN 的贡献 |
|------|---------|--------------|
| 三类空图 IoU 全部打满到 1.0 | +0.015 / +0.013 / +0.035 | **+0.021** |
| 三类非空 IoU 各 +0.05 | — | **+0.043** |

PROGRESS.md 里写的"空图分类器 = 最大剩余杠杆（+0.04 潜力）"在当时（空图 IoU 0.94）
成立，但现在空图 IoU 已经 0.965–0.987，天花板只剩 +0.021 且越来越难。
**真正的瓶颈是非空图 IoU**，尤其是油菜的 0.781。后面的排序按这个结论来。

## 二、验证集已经被污染，代理分不可信

`trainplus = train + 85% val` 训出的 tp_wr / tp_r，之后所有阈值、blend 权重、
空图分类器都在**整个 val**上拟合并评估。所以：

- `全 val 0.9043` —— 模型见过 85% 的评估样本，无效。
- `诚实 holdout 0.9041` —— 一半拟合一半评估，但**两半都在训练集里**，同样无效。
- `valhold 0.8967` —— 模型没见过这 95 张，但阈值和 blend 权重是从污染的 val 扫出来的，
  而且 n=95 时标准误约 ±0.02。

唯一可靠的历史锚点是首次提交：本地 0.853 → 线上 0.8674（+0.014）。
在重建可信验证之前，任何"已经 0.90 了"的结论都不要采信。

**修复方式**：把 train(4814) + val(664) 合并成 5478 张做 5-fold（先做 3-fold 也行），
每折的 OOF 预测拼起来当验证集。阈值、blend 权重、空图门限全部在 OOF 上拟合。
额外好处是 5 个 fold 模型天然构成一个多样性真实的集成。

## 三、按 ROI 排序的改进项

### P0 —— 半天内可完成，几乎零算力成本

**1. `soft_dice` 的 batch-flatten 问题（`losses_v2.py`）**

```python
# scripts/train_local.py: CombinedLoss.soft_dice
p = torch.sigmoid(logits).reshape(-1)   # ← 把 batch 维和通道维一起摊平了
```

指标是逐图 IoU，这里却算了一个 batch 级 Dice。后果：一张空图上误检 200 个像素，
被同 batch 其它图的大面积正样本稀释到几乎没有梯度——而在指标里这张图直接 IoU=0。
wheat/rape 有一半样本是空图，这一半的监督信号基本被吃掉了。
`wheat_rape` 双通道模式下还把两个类别的 Dice 也合并成了一个。

改成逐图逐类计算（`soft_dice_per_image`），空图上 loss ≈ 1 - 1/(pred_sum+1)，
预测越多惩罚越大，正好对应指标。这是单项 ROI 最高的改动。

**2. 空图门限按指标扫，而不是按分类准确率（`postproc.py`）**

```python
# local_blend_eval.py: fit_empty_model
acc = ((p > th).astype(int) == y).mean()   # ← 最大化准确率 ≠ 最大化 IoU
```

设 q = P(空图)、J = 非空图平均 IoU，则置空的期望收益是 q，出图是 (1-q)·J，
最优规则为 q > J/(1+J)。J=0.78 时门限 0.438，J=0.87 时 0.465——都低于 0.5 且逐类不同。
直接对最终 mean IoU 扫（`search_triple`）。

**3. 三阈值后处理（`postproc.py: triple_threshold`）**

现在是"单阈值二值化 + 去小连通域"。但"判定这张图有没有目标"和"把 mask 画准"
在 per-image IoU 下最优阈值本来就不同：前者要保守（误判代价是整图 IoU=0），
后者要激进。拆成 `t_hi`(判定门) + `min_size`(整图置空) + `t_lo`(画 mask)，
是 SIIM-ACR Pneumothorax 那一类"含空图 + 逐图指标"任务的标准解法。

**4. checkpoint 选择（`model_multitask.py`）**

- 训练中 `evaluate()` 固定用 t=0.5，实际提交用 0.41–0.55 + 后处理，选出的
  best epoch 不是最终设置下的最优 epoch。至少改成用目标阈值评估。
- 在 664 张上按 best-epoch 取 130 次最大值，σ≈0.005 的噪声会带来约 +0.005~0.01
  的虚高且不迁移。改成 top-3 checkpoint 权重平均或末段 SWA。

### P1 —— 2–3 天，非空图 IoU 的主力

**5. 加图像级分类头（`model_multitask.py`）**

现在的空图分类器只有 12 个概率图手工特征，且在 val 上拟合。真正的判别信息在
encoder 特征里。smp 的 `aux_params` 一行就能加 GAP→Dropout→Linear 头，
forward 返回 `(seg_logits, cls_logits)`，多任务 BCE 联合训练。
分类头输出直接当 `triple_threshold(cls_prob=...)` 用。既是正则化又是免费的空图判别器。

**6. Copy-Paste 增强（`copy_paste.py`）**

油菜非空 IoU 0.781 是全场最低，原因是目标小而散（中位占 7.7%）。
Copy-Paste 在保持原始尺度和背景统计的前提下提高正样本密度，对小目标分割是标配。
`crop_zoom` 是另一条路（把目标放大），但它破坏了上下文尺度且只覆盖一半样本——
r1 实验证明它是负结果，正好说明"放大"不是正确的方向，"增密"才是。

**注意**：不要往空图上贴（会消灭掉占一半的空图样本）。默认只在非空图之间做。

**7. 增强消融**

当前增强非常重：elastic + grid + optical distortion + CLAHE/sharpen/emboss/blur
+ CoarseDropout + GaussNoise。4814 张 256×256 上这可能过强，而且仓库里没有任何
消融记录。建议做一次对照：只保留几何变换（rot90/flip/transpose/ShiftScaleRotate）
+ 轻度亮度对比度，看看是不是反而更好。这类实验成本低、经常出乎意料。

**8. 更强的 backbone**

现在最强的是 mit_b3 / efficientnet-b3。可选升级：
`tu-convnext_small`、`tu-convnextv2_tiny`、`mit_b4/b5`，或 UperNet+Swin。
6GB 的 3060 是硬约束——`model.encoder.set_grad_checkpointing(True)` +
梯度累积可以撑一撑，但这条路真正需要的是回服务器跑。

**9. 训练分辨率**

"TTA 加 384/448 掉分"只证明了**测试时**改分辨率不行（train/test 不一致），
不等于训练分辨率无用。把输入 upsample 到 384 或 512 训练、测试同样分辨率，
是干净的对照实验，对小目标经常有效。这个和第 6 项是两条独立的路。

### P2 —— 收益中等但确定性高

**10. 三类联合训练**：wheat/rape 共享图像，rice 是另一组图像，现在 rice 单独训。
用一个 3 通道模型在两个域上联合训练、对无标签通道 mask 掉 loss，
encoder 的有效数据量从 4814 翻到 9628。

**11. testA 伪标签自训练**：PROGRESS 里计划了但没做完。testA 664 张，
数据量只 +12%，收益主要在空图判定的分布对齐上。做两轮 self-training。

**12. 集成多样性**：现在说"集成已饱和"，但所有成员都是 smp + ImageNet 预训练 +
同一套增强 + 同一个 256 分辨率，多样性本来就低。加入不同 backbone 家族
（CNN vs ViT）、不同训练分辨率、不同 loss、不同 fold 之后，集成还会继续涨。

## 四、集成方式

```python
# 1) loss
from opt_patch.losses_v2 import MultiTaskLoss
criterion = MultiTaskLoss(bce=1.0, dice=1.0, lovasz=0.5, cls=0.5,
                          focal_gamma=2.0, pos_weight=1.3)

# 2) model（带分类头）
from opt_patch.model_multitask import build_model, unpack
model = build_model('unet', 'mit_b3', classes=1, aux=True).cuda()

seg_logits, cls_logits = unpack(model(images))
loss = criterion(seg_logits, masks, cls_logits)

# 3) Copy-Paste：在 CropDataset.__getitem__ 里，读完 image/masks 之后、crop_zoom 之前
from opt_patch.copy_paste import CopyPasteMixer
self.cp = CopyPasteMixer(p=0.4, max_objs=3)
src = np.random.randint(len(self))
image, masks = self.cp(image, masks, *self._raw(src))

# 4) 后处理与阈值搜索（只在 OOF / 拟合半区上 search，另一半评估）
from opt_patch.postproc import search_triple, report
best = search_triple(fit_probs, fit_tgts, cls_probs=fit_cls)
print(report(eval_probs, eval_tgts, best, eval_cls))
```

## 五、提交策略

每天只有一次机会，所以：

1. 先花一天把 3-fold OOF 建起来，之后所有决策都在 OOF 上比较，不再靠单个 val。
2. 维护一张 (本地 OOF 分, 线上分) 对照表。目前只有一个点（0.853 → 0.8674，+0.014），
   两三个点之后就能建立可靠的映射，用来判断"这次值不值得提交"。
3. 只提交 OOF 分明显高于上一次（差距 > 2×标准误，约 0.008）的方案。
