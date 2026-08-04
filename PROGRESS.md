# Rice_Segmentation 大赛进度记录

> 记录每次关键进展，及时 push 到云端 (GitHub: iakRulan/Rice_Segmentation)

## 任务
作物分割（水稻 rice / 小麦 wheat / 油菜 rape 三独立二分类），指标 = 每图 IoU 均值，目标 90。
数据：每类 train 4814 / val 664 / testA 664，全部 256×256。油菜目标小且散（中位占 7.7%）。

## 2026-08-04/05 当前状态
**最优（v3 集成 + 空图分类器，val 代理分）= 平均 0.8515**
| 类别 | best IoU | 空图 IoU | 非空图 IoU |
|------|---------|---------|-----------|
| 小麦 | 0.8907 | 0.973 | 0.797 |
| 油菜 | **0.8137** | 0.940 | **0.687** |
| 水稻 | 0.8500 | 0.972 | 0.816 |
| 平均 | **0.8515** | | |

**瓶颈**：三类非空图精度全偏低，油菜 0.687 最大短板（目标小）。空图处理已很好（0.94-0.97）。

## 优化动作（进行中）
1. **v4 集成**：加入 deeplabv3plus-mit_b2（WR 0.8314 / rice 0.8335，最强单模型）+ unetpp-eff-b3。旧 v3 的 unet-mit_b2 权重被 wave2 的 deeplabv3plus 同名覆盖（train_strong.py tag 不含 arch），v4 配置已按实际 arch 重建。**v4 raw（未加空图分类器）val：wheat 0.8867 / rape 0.8139**，vs v3 raw 0.8865/0.8090 —— 提升很小，集成接近饱和。
2. **train_strong.py 改造**：加 `--lovasz_w` 参数（默认 0.5，之前写死 0.0）；新增 `--mode wheat/rape` 单类训练。
3. **油菜/小麦专用单类模型**（s3，mit_b3 + lovasz）：00:36 已启动 rape seed42 训练（GPU 训练中），随后 wheat seed42、rape/wheat seed43。
4. 待办：v4 空图分类器重训、testA 伪标签自训练、水稻空图分类器应用。

## 关键文件
- 训练：`/root/train_strong.py`（lovasz 已开）
- 集成推理：`/root/infer_ensemble.py`；配置 `cfg_multi_v4.json` / `cfg_rice_v4.json`
- 评估：`/root/eval_final_sweep.py` / `eval_scores.py`；空图分类器 `train_empty_clf.py` / `ens_features.py`
- 权重：`/root/crop_segmentation/weights/*_best.pth`
