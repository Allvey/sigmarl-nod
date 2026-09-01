# PSB-MARL P2.1-P：Projected Coordination Subspace

P2.1-P 的第一步是纯推理反事实，不重新训练，也不修改 P2.1-S checkpoint。它在已学习
均值修正之后施加固定控制投影：

\[
\mu_i
=
\mu_i^{\mathrm{Base}}
+P_{\parallel}\Delta\mu_{\theta,i},
\qquad
P_{\parallel}=\operatorname{diag}(1,0),
\qquad
\sigma_i=\sigma_i^{\mathrm{Base}}.
\]

SigmaRL 的原生动作是 `[speed, steering]`，因此该投影保留分支学习到的速度修正，并让
转向均值严格恢复 Base。它用于检验 P2.1-S 剩余的车道碰撞是否来自 steering 修正。

## 1. 隔离合同

- 使用原 P2.1-S `candidate_policy.pth`，checkpoint 字节不变；
- 投影掩码不进入 `state_dict`，旧 checkpoint 可以 strict load；
- 只允许在 testing mode 使用；
- 禁止使用 `--promote-if-noninferior`；
- 标准 `p2_manual_validation.json` 和 `comparison_to_base.json` 不会被覆盖；
- 结果单独写入：

```text
p2_counterfactual_longitudinal_only_validation.json
comparison_to_base_longitudinal_only.json
```

## 2. 自动验证

```bash
conda activate sigmarl-nod
python -m unittest discover -s tests/psb_marl -v
```

测试证明：

1. 速度修正保持非零，steering 修正逐元素严格为零；
2. steering 均值逐元素严格等于 Base；
3. 动作尺度仍逐元素严格等于 Base；
4. 投影前后使用同一个 checkpoint state dict；
5. 速度动作梯度仍能穿过因果门和近端层；
6. 反事实 promotion 被禁止。

## 3. 5-seed 反事实比较

```bash
python main_testing.py \
  --config configs/psb_marl/p2_1_s_mean_only.json \
  --run-dir <P2_1_S_RUN_DIR> \
  --checkpoint <P2_1_S_RUN_DIR>/candidate_policy.pth \
  --scenario intersection_2 \
  --max-steps 600 \
  --episodes 4 \
  --seeds 101 102 103 104 105 \
  --no-render \
  --compare-base \
  --psb-action-projection longitudinal_only
```

报告必须满足：

```text
rollout_delta_steering_abs_mean == 0
rollout_delta_steering_abs_p95 == 0
rollout_delta_steering_abs_max == 0
rollout_delta_log_scale_abs_max == 0
rollout_scale_matches_base_exactly == true
base_steering_mean_exactly_preserved == true
base_scale_exactly_preserved == true
```

若 longitudinal-only 在保持奖励非劣的同时明显减少 P2.1-S 的车道碰撞增量，再创建正式
的投影训练配置。正式阶段必须从 P1/Base 重新初始化，不得把测试期掩码伪装成已训练的
可部署策略。

## 4. 已完成的反事实结果

在 `intersection_2`、5 seeds、每个 seed 4 episodes、599 个有效 time steps 下：

| Candidate - Base | P2.1-S full mean | longitudinal-only |
|---|---:|---:|
| 奖励均值差 | +0.000534 | +0.001664 |
| 奖励置信下界 | -0.000490 | +0.000765 |
| 车辆碰撞均值差 | +0.000056 | -0.000640 |
| 车道碰撞均值差 | +0.000584 | +0.000751 |
| 总碰撞均值差 | +0.000640 | +0.000083 |
| 总碰撞置信上界 | +0.001623 | +0.000497 |

纵向投影显著提高奖励，并将车辆碰撞从轻微恶化变为改善；总碰撞几乎回到 Base。车道
碰撞没有改善，说明它并非单独由 steering 修正引起，后续仍需观察速度和路径动态之间的
耦合。综合奖励、车辆冲突和总碰撞，反事实支持进入正式投影训练阶段。

## 5. 正式训练合同

正式配置为：

```text
configs/psb_marl/p2_1_p_projected_mean_only.json
outputs/psb_marl/p2_1_p_projected_mean_only/
```

配置中的：

```json
"conditioning_mode": "causal_q_gate",
"max_delta_log_scale": 0.0,
"action_projection": "longitudinal_only"
```

共同进入 runtime 和 checkpoint 合同。正式训练时投影参与 PPO 前向及反向，steering
输出头的梯度被结构化屏蔽，而速度梯度仍穿过因果门、近端根与反对称控制器。P2.1-S
checkpoint 即使参数形状兼容，也会因 runtime contract 不同而被拒绝恢复。

首次 pilot 必须创建新 run：

```bash
python main_training.py \
  --config configs/psb_marl/p2_1_p_projected_mean_only.json \
  --iterations 30
```

训练完成前不存在可部署的 P2.1-P 候选，不能使用当前推理反事实执行 promotion。

## 6. 正式 P2.1-P 结果与后续修复

正式 30 轮 run `psb-p2-seed0-20260831T151942755241Z-09067e18` 的投影、尺度和零分支
结构检查均通过，但 5-seed 非劣门失败：奖励差下置信界为 `-0.002342`，总碰撞差上
置信界为 `+0.002296`。训练同时出现 `|b|`、`|z|`、`|q|` 持续减小，而速度修正持续
增大的可缩放增益退化。

因此 P2.1-P 保留为无固定增益界的消融，不再作为 promotion 候选。下一阶段采用
[`P2_1_GAIN_BOUNDED_GATE.md`](P2_1_GAIN_BOUNDED_GATE.md) 定义的固定扇区界，让动作影响
随分岔活动度同步趋零。
