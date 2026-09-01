# PSB-MARL P2.1-G：Gain-Bounded Sector Gate

P2.1-G 修复 P2.1-P 暴露出的“分岔幅值缩小、下游网络增益放大”退化。它保留
mean-only 和 longitudinal-only 两个已验证的结构约束，并在动作修正外增加一个不可学习
的分岔活动包络：

\[
q_{ij}=\tanh(z_{ij}/z_0),\qquad
a_i=\sum_j\alpha_{ij}|q_{ij}|\in[0,1],
\]

\[
\mu_i
=
\mu_i^{\mathrm{Base}}
+P_{\parallel}B_\mu a_i
\tanh(r_{\mu,i}\odot g_{\mu,i}),
\qquad
P_{\parallel}=\operatorname{diag}(1,0),
\qquad
\sigma_i=\sigma_i^{\mathrm{Base}}.
\]

因此逐智能体满足固定扇区界：

\[
\boxed{
|\Delta\mu_i|
\le
P_{\parallel}B_\mu a_i
}
\]

无论 adapter 权重多大，都不能用下游增益补偿趋零的分岔状态；当全部 \(q_{ij}=0\)
时，候选策略逐元素精确恢复 Base。包络内的动作方向和相对大小仍由 MARL 学习，并未
预设正、负意见分别对应加速或减速。

## 1. 为什么 P2.1-P 未通过

正式 P2.1-P pilot：

```text
outputs/psb_marl/p2_1_p_projected_mean_only/runs/
psb-p2-seed0-20260831T151942755241Z-09067e18
```

30 轮训练中，后 10 轮相较前 10 轮：

- 活跃边 `|b|` 从 `0.0645` 降到 `0.0205`；
- 活跃边 `|z|` 从 `0.2496` 降到 `0.1011`；
- 活跃边 `|q|` 从 `0.3547` 降到 `0.1713`；
- 分支承诺率从 `0.330` 降到 `0.110`；
- 速度修正反而从 `0.00398` 增到 `0.00894`。

也就是说，能量项持续压低上游控制，而 adapter 通过增益放大维持动作影响。5-seed 正式
比较的奖励差下置信界为 `-0.002342`，低于 `-0.002` 门限；总碰撞差上置信界为
`+0.002296`，高于 `+0.002` 门限，因此 P2.1-P 不具备 promotion 资格。

## 2. 实现合同

新配置与输出目录为：

```text
configs/psb_marl/p2_1_g_sector_projected_mean_only.json
outputs/psb_marl/p2_1_g_sector_projected_mean_only/
```

关键配置：

```json
"conditioning_mode": "sector_q_gate",
"max_delta_log_scale": 0.0,
"action_projection": "longitudinal_only"
```

`branch_activity`、`rollout_branch_activity_mean`、
`rollout_branch_activity_max` 和 `rollout_sector_bound_max_violation` 会写入 rollout/训练
诊断。promotion 除原有非劣、近端、反对称、尺度冻结和 steering 投影门外，还要求：

```text
sector_bound_satisfied == true
rollout_sector_bound_max_violation <= 1e-7
```

P2.1-P 保留为无固定增益界的消融，不修改其配置或已有 checkpoint。P2.1-G 的 runtime
contract 与 P2.1-P 不同，必须创建新 run，不能从 P2.1-P checkpoint 恢复。

## 3. 代码级验证

```bash
conda activate sigmarl-nod
PYTHONNOUSERSITE=1 python -m unittest discover -s tests/psb_marl -v
```

测试覆盖：固定扇区界在极大 adapter 权重下仍成立、零分支精确恢复 Base、速度梯度仍
穿过近端层到达反对称控制器，以及 promotion 拒绝任何超出容差的扇区违反。

## 4. 下一次手动训练

先运行独立 30 轮 pilot：

```bash
python main_training.py \
  --config configs/psb_marl/p2_1_g_sector_projected_mean_only.json \
  --iterations 30
```

不要附加 `--resume`。训练完成后先检查：

```text
rollout_sector_bound_max_violation == 0
rollout_delta_steering_abs_max == 0
rollout_delta_log_scale_abs_max == 0
rollout_scale_matches_base_exactly == true
```

并比较前、后 10 轮的 `rollout_active_q_abs_mean`、
`rollout_branch_activity_mean` 与 `rollout_delta_speed_abs_mean`。本阶段的目标不是强迫
\(q\) 变大，而是保证动作影响不能在 \(q\) 消失时保持不变。

正式 5-seed 测试应在分析完 pilot 后再执行，避免对一个已经发生结构坍缩的候选浪费
完整评估预算。

## 5. P2.1-G 正式验证结果

run `psb-p2-seed0-20260831T160701321843Z-c4cabb50` 的 30 轮 pilot 满足所有硬约束，
奖励后 10 轮高于前 10 轮，总碰撞后 10 轮低于前 10 轮，因此进入正式评估。5-seed
结果中，奖励门通过，车辆碰撞平均改善，但车道碰撞增加使总碰撞置信上界达到
`+0.002099`，略高于 `+0.002` 门限。因此该候选保持 Base fallback，不能 promotion。

下一阶段见
[`P2_1_URGENCY_SUPPORTED_SECTOR.md`](P2_1_URGENCY_SUPPORTED_SECTOR.md)：它保留当前
固定扇区界，并要求分岔动作修正同时获得当前物理冲突强度的支持。
