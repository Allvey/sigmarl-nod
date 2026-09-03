# PSB-MARL P5：单阶段联合训练

P5 支持两种初始化模式。原配置直接从已完成的 P3.3 Candidate warm-start；新的 scratch
配置则随机初始化 Candidate Base Actor、反对称控制网络、Branch Encoder/Adapter、
augmented central critic 和 differential critic，并在一个 run 内按课程联合训练。

Source Base 始终保持只读，承担 Candidate/Base 的 CRN 配对采样和验证通过前的部署回退。
warm-start 配置还将其用作 Candidate backbone 的分布 KL 锚点；scratch 配置关闭该锚点，
避免把随机 Candidate 拉回旧策略。两种模式都不会覆盖原始 Base Actor 或 Base Critic。

首版继续锁定 P3.3 的结构边界：`longitudinal_only`、mean-only、
`supported_sector_q_gate`，并仅启用 vehicle dual。P5 不同时开放 steering、scale 或 lane
dual；若联合训练效果不佳，再依据诊断结果定位原因。

## Transition PPO

P5 不再使用 Sequence PPO。环境 rollout 仍逐物理步维护真实的有状态分岔变量
`z_prev -> z_next`；优化时则把 rollout 展平，每个 transition 从采集时保存的
`z_prev_dense` 独立重算当前步近端更新和动作分布。梯度可以通过当前步的 proximal solver
传回 ControlNet 和 Actor，但不会跨 transition 传播，因此不再执行 16-step truncated
BPTT。

锁定设置为：

- `ppo_mode = transition`；
- 每批最多 `15` 个 PPO epochs；
- minibatch 为 `1024`；
- epoch 平均近似 KL 超过 `0.03` 时提前停止当前批次优化。

这项修改有意牺牲跨时间 credit assignment，以降低计算量并避免原先 `60` 个 Sequence
PPO epochs 对同一批数据的过度拟合。P2/P3 的历史 Sequence PPO 实现和产物保持不变。
训练指标仍保留若干 `sequence_*` 兼容字段供旧分析脚本读取；P5 新 run 中
`sequence_chunk_count=0`，应以 `ppo_mode` 和 `temporal_backpropagation_enabled` 判断实际
训练模式。

## Warm-start 联合目标与学习率

- Actor 使用同步配对 rollout 得到的 differential advantage；
- augmented central critic 使用绝对 augmented reward 的 GAE value target；
- differential critic 继续在线拟合三通道 Base-relative return；
- `b^2` 同时保留在 augmented reward 和显式可微正则路径；
- Candidate Base Actor 的学习率固定为主学习率的 `0.1`；
- Source Base 高斯动作分布 KL 的系数为 `0.01`。

## Scratch 课程

scratch 配置位于：

```text
configs/psb_marl/p5_scratch_joint_psb_marl.json
```

该配置保留为首个 monolithic scratch 实验的记录。推荐使用经过稳定性修正的 v2：

```text
configs/psb_marl/p5_scratch_v2_joint_psb_marl.json
```

它不是沿用 P3.3 权重后再重置优化器，而是随机初始化全部 Candidate 可学习模块。已训练的
Source Base 只用于 CRN 对照和部署回退；P3.3 parent 仍用于结构合同及来源追踪，不把 Actor、
absolute critic 或 differential critic 权重载入 Candidate。

v2 的 250 个 iteration 课程固定为：

- `1-60`：仅训练随机 Candidate Base Actor 和 absolute critic；不采集影子 Base、不训练
  differential critic，PSB 参数冻结；使用 `30 × 512` 的普通 transition PPO 更新，不应用
  KL early-stop；
- `61-100`：恢复 CRN 配对与 differential critic，Actor 仍只使用 absolute GAE；PSB
  重新可训练，并把 branch activity offset 从 `0.05` 线性衰减到零；vehicle dual 保持为 `0`；
- `101-200`：absolute 与 differential advantage 线性混合，differential 权重从 `0.01`
  增至 `1.0`；vehicle dual 开始更新；
- `201-250`：Actor 只使用 differential advantage。

scratch 下 Base Actor 使用完整主学习率，Source Base KL anchor 系数为 `0`。ControlNet 的
末层仍为小增益初始化，Adapter 输出层仍为零初始化；v2 仅在 PSB 启动阶段加入会消退的
activity offset，以避免“零 Adapter × 零 branch activity”使分支永久没有梯度。Base
backbone 本身始终是随机初始化。

启动完整 scratch 训练：

```bash
conda run -n sigmarl-nod python main_training.py \
  --config configs/psb_marl/p5_scratch_v2_joint_psb_marl.json
```

若要先隔离验证随机初始化的 Candidate Base Actor，可运行完整 250 轮的 Base-only
消融：

```bash
conda run -n sigmarl-nod python main_training.py \
  --config configs/psb_marl/p5_scratch_v2_base_only_250.json
```

该配置的 1-250 轮都处于 `base_actor_pretrain`：不采集影子 Base、不会启用 PSB、
差分 critic 或 dual。它复用原始 Base 的每批 `60 × 512` PPO 更新且不启用 KL
early-stop；它的用途是检验 Base-only 学习曲线，不会回答 PSB 是否带来收益。

训练记录会额外写入 `scratch_training_phase`、两类 advantage 权重和
`dual_update_enabled`，以及 `paired_learning_enabled`、`psb_learning_enabled` 和
`branch_activity_bootstrap_offset`。PPO 的 KL 指标使用非负的 k3 估计，产物 manifest
会明确记录三类 Candidate 模块的初始化方式。

运行配置位于：

```text
configs/psb_marl/p5_joint_psb_marl.json
```

单 iteration 冒烟命令：

```bash
conda run -n sigmarl-nod python main_training.py \
  --config configs/psb_marl/p5_joint_psb_marl.json \
  --iterations 1
```

## 关键诊断

训练记录除 P3.3 的同步边界、差分优势、对偶量和 proximal residual 外，还必须包含：

- `base_actor_gradient_norm`；
- `base_actor_parameter_drift_rms` 与 `base_actor_parameter_drift_max_abs`；
- `source_base_gaussian_kl` 与 `loss_base_anchor`；
- `absolute_critic_loss` 与 `differential_critic_loss`；
- Base Actor、absolute critic、differential critic 的实际学习率。
- `ppo_mode`、`temporal_backpropagation_enabled`；
- `ppo_epochs_configured`、`ppo_epochs_completed`、`ppo_early_stop_triggered`；
- `ppo_target_kl`、`ppo_update_count` 和 `transition_count`。

P5 训练产物仍处于 quarantine。只有通过 paired non-inferiority、绝对安全预算和 efficacy
gate 后，Candidate 才能替代 `base_fallback_policy.pth`。
