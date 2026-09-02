# Rice_Segmentation 大赛进度记录

> 记录每次关键进展，及时 push 到云端 (GitHub: iakRulan/Rice_Segmentation)

> **重要约束**：每天只有一次提交机会。策略 = 持续迭代训练/集成，直到诚实 val 明确足够（预估线上 ≥90）再提交。**教训：trainplus（train+85%val）模型的 val 分全部污染，只有 train-only 模型的 val 可信。**

## 🚀 边界感知损失优化与远程训练就绪（2026-09-03，冲刺 0.97 目标）
- **GitHub 远程同步**：通过 GitHub Token 免密通道打通推送，代码已同步最新微调配置与执行脚本。
- **远程算力部署**：远程 RTX 3080 Ti 节点（`connect.westb.seetacloud.com:42734`）配置 PyTorch 2.6.0+cu124 与 SMP 全套依赖。
- **数据全量挂载**：`public.zip` 已全量传输并在远程 `/root/autodl-tmp/data/public` 完成解压挂载。
- **边界感知微调 40 周期全量收敛成果**：基于形态学边缘差分边界感知损失（`BoundaryConsistencyLoss`，权重 0.2）+ Lovasz 复合损失的 `opt_wr_boundary_mitb3` 模型在 RTX 3080 Ti 上已完成全部 40 个周期的训练：
  - **初始阶段**：Epoch 1 `loss=2.1817 fixed=0.0565`
  - **突破阶段**：Epoch 3 固定 IoU 跃升超 5 倍至 `0.6482`，Epoch 6 调优 IoU 突破 `0.8014`
  - **攻坚阶段**：Epoch 10 固定 IoU 突破 `0.8222`，Epoch 13 调优 IoU 突破 `0.8301`
  - **最终收敛**：Epoch 40 最终损失降至 **0.2496**，**单模型最佳固定 IoU 达到 0.841565（0.8416）**！
  - 最优模型权重（379 MB）已完整归档至 `/root/autodl-tmp/weights/opt_boundary/opt_wr_boundary_mitb3/best.pth`。
- **水稻（rice）边界感知微调收敛追踪**：基于形态学边缘差分边界损失的 `opt_rice_boundary_mitb3` 独立会话在 RTX 3080 Ti 上持续加速推进：
  - **Epoch 1**：`loss=1.8219 fixed=0.2812 tuned=0.3617@0.40 best=0.2812` (耗时 88s)
  - **Epoch 2**：`loss=0.9711 fixed=0.3230 tuned=0.4635@0.45 best=0.3230` (耗时 74s，Loss 骤降 47%)
  - **Epoch 3**：`loss=0.8826 fixed=0.7220 tuned=0.7485@0.46 best=0.7220` (耗时 73s，固定 IoU 跃升至 0.7220)
  - **Epoch 4**：`loss=0.8229 fixed=0.7846 tuned=0.7967@0.43 best=0.7846` (耗时 74s，固定 IoU 达 0.7846)
  - **Epoch 5**：`loss=0.7943 fixed=0.8061 tuned=0.8098@0.46 best=0.8061` (耗时 74s，固定 IoU 突破 0.80)
  - **Epoch 6**：`loss=0.7728 fixed=0.8156 tuned=0.8169@0.44 best=0.8156` (耗时 70s，固定 IoU 达 0.8156)
  - **Epoch 7**：`loss=0.7340 fixed=0.8223 tuned=0.8226@0.47 best=0.8223` (耗时 73s，固定 IoU 正式突破 0.82 大关)
  - **Epoch 8（突破 0.825，阈值校准完美）**：`loss=0.6640 fixed=0.8251 tuned=0.8251@0.49 best=0.8251` (耗时 72s，损失破 0.70 降至 0.6640，固定 IoU 达 0.8251，最佳阈值 0.49 几乎完美对齐 0.50)
  - **Epoch 9（逼近 0.828，损失下探 0.63）**：`loss=0.6351 fixed=0.8277 tuned=0.8278@0.52 best=0.8277` (耗时 73s，固定 IoU 攀升至 0.8277，调优 IoU 达 0.8278，最优阈值 0.52 处于极佳概率平衡态)
  - **Epoch 10（正式攻克 0.83 大关）**：`loss=0.5940 fixed=0.8302 tuned=0.8309@0.59 best=0.8302` (耗时 73s，损失跌破 0.60 达 0.5940，固定 IoU 达 0.8302，调优 IoU 突破 0.8309)
  - 模型权重与历史记录自动同步至 `/root/autodl-tmp/weights/opt_boundary/opt_rice_boundary_mitb3/`，40 周期全量微调全速推进中。

## ⚠️ 前提证伪 + 新管线复现验证（2026-08-09，用户 review 二次驱动）

**前提验证（`tests/verify_premises.py`，本地 4 分钟）——两个核心前提全部不成立**：
- **拼图不成立**：瓦片内部相邻列不连续 2.73 vs 右邻 i,i+1 28.97 / 下邻 i,i+83 28.86 / 随机 34.32。右邻/下邻 ≈ 随机 → **grid_width=83 不构成空间栅格**，768 上下文与邻行先验两条路砍掉。
- **双时相不成立**：同 ID rice vs wheat_rape 边缘 NCC 0.003 ≈ 随机 -0.007 → **同 ID 不是同一块地**，6ch 的 3 个通道是噪声，正好解释 wheat/rape 掉 0.06-0.07。
- **空图地板确认**：wheat 52.7% / rape 43.6% / rice 21.8% → 3ch 矩阵里 rice 0.225/0.218 = 全空死头，非"模型看不见 rice"。
- **P0 矩阵结论全部作废**：ctx768_6ch 的"决定性 +0.097"= 把 rice 自己的影像还给 rice 任务；joint 3 类多任务税随 6ch 一并撤销。

**A0/B0 复现旧基线（新管线裸配置，成功）**：finetune_v2 + MosaicSegDataset，unet/mit_b3，256 3ch ImageNet，bce+dice+lovasz0.5，无增强/无 focal/无 pos_weight/无 aux，提供 train/val（4814/664），130ep patience30，EMA。

| 任务 | 及格线 | 旧基线 | 新管线 best | 判定 |
|------|--------|--------|------------|------|
| A0 wheat | ≈0.88 | 0.8816 | **0.8928** (ep50) | ✅ |
| A0 rape | ≈0.815 | 0.8154 | **0.8194** (ep50) | ✅ |
| B0 rice | ≈0.838 | 0.838 | **0.8589** (ep85) | ✅ |

→ **新管线无隐藏 bug**；P0 滑铁卢 = 配置坏（6ch 噪声 + joint 税），非管线坏。5 折 OOF 按用户决定**不重启**。下一步：诚实 OOF 协议重设计（#9）+ 新管线最终模型生产与 v2mix 集成（#8）。

## 新方向：输入补充 + 隔离行验证（2026-08-09，用户 review 驱动）

**用户审查定案**：换 backbone 在 ±0.005 噪声里打转（SwinT 0.7563 vs SwinB 0.7562 实测确认，89.7M vs 26M 无差别）。真正的杠杆 = **补输入信息**（双时相 + 邻行标签先验 + 768 上下文），唯一能给 +0.03 的东西。

**数据结构已验证（本地脚本）**：
- rice/ 与 wheat_rape/ 共享同一套 tile id（各 train 4814 / val 664 / testA 664）→ 同一块地两个时相 → 6ch 输入、3 类联合输出
- 网格 = 83 列 × 82 行 = 6806 瓦片。**未标注行 = row%10∈{2,8}**：row≡8 是 testA（8 行）、row≡2 是 testB（隐藏 8 行）。每个 testA 行上下邻行 83/83 全标注 → 邻行先验成立
- 5 折隔离行验证：`labeled_row_position % 5`，验证行上下邻行在训练集（模拟 testA），OOF 覆盖 5478 张

**P2 四 bug 已修**（本地 13d5054，服务器 a067b73）：
1. engine.py GradScaler 反保护（enabled=False 时 step=裸 step，非有限梯度写烂权重）→ 删 GradScaler，zero_grad+skip
2. losses.py cls 损失死（supported 无 "cls"，forward 不传 cls_logits）→ 透传 + SMP/Satlas 加 aux 头
3. metrics.py 阈值范围 [0.30,0.70]→[0.15,0.85]（rice 最优 0.37 曾贴下界）
4. data.py MosaicStore.window 下边界越界 → `neighbor not in self.paths` 一揽子覆盖

**768 显存实测（服务器 RTX 3080 Ti 12GB）**：Unet+mit_b3 @768×768×9ch bf16 **batch2 = 6.33GB 峰值 / 141ms·step**（epoch≈6min）；**mit_b5 @768 batch1 = 5.97GB / 175ms** 也放得下。不需要 24GB 卡。

**大模型对比（Satlas 单时相 3ch 512 输入）**：
| 模型 | 参数量 | best fixed | tuned |
|------|--------|-----------|-------|
| SwinT | 28M | 0.756277 | 0.765823 |
| SwinB | 89.7M | **0.756196** | 0.759960@0.55 |
| s3_rape mit_b3 | 47M | 0.8154 | — |

**结论坐实：输入不变，backbone 无差别（SwinB 89.7M 反而略低于 SwinT）。弃选型，转 P0。**

**P0 方向矩阵（fold0 隔离行验证，Unet+mit_b3，15ep 快速方向检查，全跑完）**：
| 配置 | wheat | rape | rice | tuned | 结论 |
|------|-------|------|------|-------|------|
| ctx256_3ch（sanity） | 0.849 | 0.786 | 0.225 | 0.660 | 基线 |
| ctx768_3ch | 0.852 | 0.794 | 0.218 | 0.665 | 768 上下文无增量 |
| ctx768_6ch（双时相） | 0.782 | 0.732 | 0.771 | **0.762** | **决定性加分类** (+0.097) |
| ctx768_9ch（+邻行先验） | 0.778 | 0.746 | 0.768 | 0.764 | 先验无增量（≈6ch） |

**矩阵结论坐实**：
- **双时相 6ch 是唯一真正的加分类**：rice 0.218→0.771（3ch 只含 wheat_rape 季图像，模型看不见水稻；6ch 给 rice 自己的季相，类 IoU 从地板拉到正常）。wheat/rape 0.78/0.73 也健康。
- **768 真实上下文无增量**（256 vs 768 的 3ch tuned 0.660 vs 0.665）—— 邻域像素信息对单瓦片预测冗余，模型可外推。
- **9ch 邻行标签先验无增量**（0.764 vs 0.762，噪声内）—— 先验行与中心瓦片图像高度相关，模型已学会从图像推。快速设定下先验可弃。
- 生产路线：**6ch 双时相 + 256 上下文**（正在验证 rice 是否依赖邻行像素；若 256 保 rice，则 OOF 训练 ×4 提速）。3 类联合输出替代 3 个独立单类模型，训练样本复用率 3×。

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
