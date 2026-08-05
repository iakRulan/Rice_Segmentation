# Rice_Segmentation 大赛进度记录

> 记录每次关键进展，及时 push 到云端 (GitHub: iakRulan/Rice_Segmentation)

## 任务
作物分割（水稻 rice / 小麦 wheat / 油菜 rape 三独立二分类），指标 = 每图 IoU 均值，目标 90。
数据：每类 train 4814 / val 664 / testA 664，全部 256×256。油菜目标小且散（中位占 7.7%）。

## 线上得分（2026-08-05 首次提交）
**Final = 0.8674**（小麦 0.8955 / 油菜 0.8373 / 水稻 0.8695）。线上比 val 代理（0.853）高 ~0.015。距 90 还差 +0.0326。

## 当前状态
**最优（v3 集成 + 空图分类器，val 代理分）= 平均 0.8515**
| 类别 | best IoU | 空图 IoU | 非空图 IoU |
|------|---------|---------|-----------|
| 小麦 | 0.8907 | 0.973 | 0.797 |
| 油菜 | **0.8137** | 0.940 | **0.687** |
| 水稻 | 0.8500 | 0.972 | 0.816 |
| 平均 | **0.8515** | | |

**瓶颈**：三类非空图精度全偏低，油菜 0.687 最大短板（目标小）。空图处理已很好（0.94-0.97）。

## 优化动作（进行中）
1. **v4 集成**：加入 deeplabv3plus-mit_b2 + unetpp-eff-b3。v4 raw val ≈ v3（wheat 0.8867 / rape 0.8139 / rice 0.8495），**集成已饱和**。
2. **专用单类模型 + blend（当前最优）**：mit_b3 + lovasz 4 模型（rape 0.8154/wheat 0.8816 独立），blend alpha=0.5 → raw wheat 0.8878 / rape 0.8183，+clf ≈ **0.853 平均**。提升很小。
3. **高分辨率杠杆证伪**：TTA 加 384/448 反而掉分（wheat 0.8797 vs 0.8867）——模型在 256 原生最优。
4. **空图分类器 = 最大剩余杠杆（+0.04 潜力）**：诊断显示油菜 48 张图被空图 clf 误判（20 空图被预测有目标→IoU 0；28 非空图被清零→IoU 0）。形状分析：幻觉成分偏小（area 798 vs 2367），有部分可分性。
5. **testA 提交**（生成中）：v4 blend + 空图 clf（val 拟合）+ 逐类阈值 → /root/submission_final。
6. **伪标签自训练（计划）**：testA 置信伪标签 + train 重训，针对线上分。

## 关键文件
- 训练：`/root/train_strong.py`（lovasz 0.5、wheat/rape 单类、--data_root 支持伪标签重训）
- 集成：`/root/infer_ensemble.py`（TTA_SCALES 环境变量可配）+ cfg_multi_v4/cfg_rice_v4/cfg_ded_*
- 提交：`/root/make_testA_submission.py`；伪标签：`/root/gen_pseudo_train.py`
- 评估：`/root/eval_final_sweep.py` / `clf_th_fast.py` / `diag_empty.py`

## 关键文件
- 训练：`/root/train_strong.py`（lovasz 已开）
- 集成推理：`/root/infer_ensemble.py`；配置 `cfg_multi_v4.json` / `cfg_rice_v4.json`
- 评估：`/root/eval_final_sweep.py` / `eval_scores.py`；空图分类器 `train_empty_clf.py` / `ens_features.py`
- 权重：`/root/crop_segmentation/weights/*_best.pth`
