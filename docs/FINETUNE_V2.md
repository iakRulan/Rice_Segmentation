# 下一轮微调方案（v2）

## 目标和判断标准

这轮的目标不是继续堆高同构 MiT 模型，而是验证遥感预训练是否能给 rape/rice 带来
不相关增益。所有 checkpoint 按固定阈值 0.5 的逐图 IoU 选择；调优阈值仅记录，不能
用于 early stopping，避免在 664 张验证图上反复过拟合。

首轮准入标准：

- rape 单模 TTA 原始 IoU >= 0.825，或加入现有模型后空间分组 OOF 提升 >= 0.005；
- rice 单模 TTA 原始 IoU >= 0.860；
- 未达标时不追加相同模型的 seed，先切换预训练域或检查标签/尺度。

## 实验顺序

1. 等当前 mosaic2 rape/rice 完成并固定结果，作为同架构控制组。
2. 运行 `finetune_satlas_rape_s2tiny.json`。它适合 12GB 显存，是成本最低的域预训练探针。
3. 若 rape 达到 0.825，运行 rice 的 Swin-T 配置。
4. 若 Sentinel-2 Swin-T 有效，再运行 `Aerial_SwinB_SI`。官方航空权重只有 Swin-B，
   因此使用 batch=1、accumulation=4；不要把不存在的 Aerial Swin-T 当作实验项。
5. 只有单折提升成立后才构建 4 个连续栅格行 folds；先跑 2 折确认方向，再决定是否补齐 4 折。
6. 最终只融合空间 OOF 上有增益的模型。融合权重在 OOF 上确定，再用 train+val 重训。

## 三阶段训练

- `decoder_warmup`：冻结 backbone 3 轮，让随机初始化的融合头先适应掩膜。
- `full_finetune`：backbone 使用 head 学习率的 0.1 倍，避免灾难性遗忘。
- `low_lr_polish`：低学习率收尾；EMA checkpoint 作为部署权重。

rape 采样概率固定为空图/微小/小/大目标 `0.35/0.25/0.25/0.15`。这既强化小目标，
又保留足够空图来约束误检。损失使用逐图 BCE + Dice + Lovasz，额外加入轻量 Tversky
和 boundary 项；不再叠加高 pos_weight 或强 focal。

## 代码结构

```text
cropseg/
  config.py      配置加载和校验
  tasks.py       类别、影像域的唯一映射
  data.py        栅格拼图、数据集、面积分桶采样
  models.py      SMP/Satlas 统一模型工厂
  losses.py      区域损失和边界辅助损失
  metrics.py     与比赛一致的逐图 IoU、阈值搜索
  engine.py      AMP、梯度累积、EMA、训练/验证循环
scripts/
  finetune_v2.py             统一训练入口
  predict_v2.py              统一验证/testA 推理入口
  build_spatial_folds_v2.py  连续栅格行分折
```

旧脚本保留为历史实验，不再继续往其中复制新逻辑。新 checkpoint 带 `schema_version=2`、
模型配置、归一化、context size、EMA 和完整 history，推理不再依赖手写配置猜测模型结构。

## 命令

```bash
source /etc/network_turbo
pip install --no-deps -r requirements-satlas.txt

python scripts/finetune_v2.py \
  --config configs/finetune_satlas_rape_s2tiny.json

python scripts/predict_v2.py \
  --checkpoint weights/v2/satlas_s2tiny_rape_512_seed42/best.pth \
  --split val --tta --cache --output /root/satlas_s2tiny_rape_val.npz

python scripts/build_spatial_folds_v2.py \
  --task rape --folds 4 --output data/folds/rape_rows_k4.json
```

空间折训练时在配置的 `data` 中加入：

```json
"fold_file": "data/folds/rape_rows_k4.json",
"fold": 0
```

每个 fold 的验证区域是连续栅格行，不使用随机 tile K-fold。
