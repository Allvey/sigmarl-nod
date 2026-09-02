# PSB-MARL P5：解冻 Base 的单阶段联合训练

P5 直接从已完成的 P3.3 Candidate warm-start，不再拆分额外子阶段。训练从第一个
iteration 起同时更新 Candidate Base Actor、反对称控制网络、Branch Encoder/Adapter、
augmented central critic 和在线 differential critic。

Source Base 始终保持只读，承担三项职责：Candidate/Base 的 CRN 配对采样、Candidate
backbone 的分布 KL 锚点，以及验证通过前的部署回退。解冻的只是 Candidate checkpoint
内部的 Base backbone，不会覆盖原始 Base Actor 或 Base Critic。

首版继续锁定 P3.3 的结构边界：`longitudinal_only`、mean-only、
`supported_sector_q_gate`，并仅启用 vehicle dual。P5 不同时开放 steering、scale 或 lane
dual；若联合训练效果不佳，再依据诊断结果定位原因。

## 联合目标与学习率

- Actor 使用同步配对 rollout 得到的 differential advantage；
- augmented central critic 使用绝对 augmented reward 的 GAE value target；
- differential critic 继续在线拟合三通道 Base-relative return；
- `b^2` 同时保留在 augmented reward 和显式可微正则路径；
- Candidate Base Actor 的学习率固定为主学习率的 `0.1`；
- Source Base 高斯动作分布 KL 的系数为 `0.01`。

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

P5 训练产物仍处于 quarantine。只有通过 paired non-inferiority、绝对安全预算和 efficacy
gate 后，Candidate 才能替代 `base_fallback_policy.pth`。
