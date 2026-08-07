# Rice_Segmentation 大赛进度记录

> 记录每次关键进展，及时 push 到云端 (GitHub: iakRulan/Rice_Segmentation)

> **重要约束**：每天只有一次提交机会。策略 = 持续迭代训练/集成，直到诚实 val 明确足够（预估线上 ≥90）再提交。**教训：trainplus（train+85%val）模型的 val 分全部污染，只有 train-only 模型的 val 可信。**

## 线上实况（2026-08-06/07 实测）
| 提交 | 内容 | 线上 |
|------|------|------|
| 首次 | v4 + 专用（train-only），val 0.853 | **0.8674** |
| 第二次 | tp_wr/tp_r 主导（trainplus） | **0.8664**（小麦+0.002/油菜+0.009/水稻-0.014）|

**trainplus val 0.9043 是污染假象，线上不变。水稻退步由 tp_r 导致（已回退 v4）。**

## 用户 P0/P1 补丁后（2026-08-07）
逐图 loss、aux 分类头、三阈值后处理、Copy-Paste、SWA 全部落地。**诚实评估（train-only，holdout）：**
| 集成 | 诚实 val | 预估线上 |
|------|---------|---------|
| v4+专用基线 | 0.8539 | 0.8674 |
| **v2mix（v2+v4+专用）** | **0.8566** | ~0.870 |

**结论：v2（逐图loss）单模型追平旧 5 模型集成（真实修复），但所有 v2 变体（v2/v2cp/v2p/v2r）诚实 val 全部 ~0.856 水平，伪标签/Copy-Paste 无实质提升。冲 90 需诚实 val ~0.886（+0.03），本地 6GB 单卡已到瓶颈，需 OOF K-fold(train+val) 或服务器级算力。训练已按用户指示停止。**
**最佳提交：`submission_v2mix.zip`（train-only，诚实 0.8566，预估线上 ~0.870）。**

## 任务
作物分割（水稻 rice / 小麦 wheat / 油菜 rape 三独立二分类），指标 = 每图 IoU 均值，目标 90。
数据：每类 train 4814 / val 664 / testA 664，全部 256×256。油菜目标小且散（中位占 7.7%）。

## 本地复现 + 集成提升（2026-08-05，Windows + RTX 3060 6GB）
**全流程已在本地复现并大幅提升 val 代理分：**
| 阶段 | wheat | rape | rice | MEAN |
|------|-------|------|------|------|
| 基线复现 blend_repro (v4+ded42) | 0.8902 | 0.8194 | 0.8495 | **0.8530** |
| blend_v3 (v4+s7+ded42+ded43) fixed | 0.9066 | 0.8760 | 0.8495 | **0.8773** |
| blend_v3 阈值/后处理扫描 | 0.9078 | 0.8794 | 0.8495 | **0.8789** |
| blend_opt 权重优化 (val 拟合) | 0.9255 | 0.8851 | 0.8540 | **0.8882** |
| **诚实 holdout（一半拟合/一半评估）** | 0.908-0.917 | 0.874-0.883 | 0.842 | **≈0.8777** |

**关键教训**：blend_opt 的 0.8882 是 GBT 空图分类器 val 拟合的假象。诚实 holdout 下 blend_v3 和 blend_opt 都是 **0.8777**——权重优化没有真实增益。真实瓶颈是模型本身，只能靠训练提升。

**下一步**：trainplus(train+85%val) + focal 强路线（s7 证明有效），训练 tp_wr(deeplabv3plus/mit_b3) + tp_r(水稻)。

## 训练进展（2026-08-05 晚）
- **r1 (油菜 crop_zoom) 结论 = 负结果**：raw val 0.798（低于 s3 0.815），blend 优化器给它的权重仅 0.025（s7 占 0.885）。**弃用 r1**。smp 不支持 unetpp/mit_b3，trainplus 模型改用 deeplabv3plus/mit_b3。
- **tp_wr (trainplus wheat_rape, deeplabv3plus/mit_b3, focal) = 巨大成功**：valhold best **0.8980**（wheat 0.900 / rape 0.896），超过 s7 的 0.887。trainplus 路线再次验证。
- **blend_tp（加入 tp_wr）**：全 val fixed 0.8870 / sweep 0.8883 / **诚实 holdout 0.8856**（比 blend_v3 的 0.8777 +0.008）。权重：wheat tp_wr=0.905，rape tp_wr=0.693+s7=0.261。
- **tp_r (trainplus rice, unet/mit_b3, pos_weight 1.3) 训练中**：水稻是当前最弱类（0.8495），预计 +0.01。
- 提交设置（blend_tp 扫描）：wheat t=0.55 pp=(30,30)；rape t=0.55 pp=(0,60)；rice t=0.37 pp=(60,0)。

## 训练（进行中）
- `shell/train_rape_r1.sh`：油菜 unet/mit_b3 + crop_zoom 0.5（512 画布前景偏置裁剪）+ focal，目标击败 s3_rape 的 raw val 0.8154。

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

## 本地关键文件
- 环境：`pyproject.toml`（uv，缓存/解释器在项目 `.uv-cache`/`.uv-python`），python `.venv/Scripts/python.exe`
- 推理：`scripts/local_ensemble.py`（内存安全，适配 6GB，`--subdir` 指定图片目录）
- 评估：`scripts/local_blend_eval.py --spec configs/blend_v3.json [--fixed|--sweep] [--subset valhold]`
- 提交：`scripts/local_submission.py --spec <blend> --out_dir outputs/submission`
- 训练：`scripts/train_local.py`（crop_zoom 小目标增强、focal、梯度累积）
- 配置：`configs/blend_v3.json`（最优 blend），`configs/blend_repro.json`（基线）
- 权重：`weights/`（51 文件 6.2GB，从服务器拉到本地）
