# PSB-MARL P2.1-S：Scale-Frozen Mean-Only Adapter

P2.1-S 保留 P2.1-C 的分岔因果门控，但不再学习动作尺度。策略只学习分支条件下的
动作均值：

\[
\mu_i=\mu_i^{\mathrm{Base}}+\Delta\mu_{\theta,i}(o_i,c_i),
\qquad
\sigma_i=\sigma_i^{\mathrm{Base}}.
\]

其中 `c_i` 仍由带符号分支 `q_ij` 聚合得到，并经过无偏置因果门。因此同时满足：

```text
所有 q_ij = 0  => delta_loc = 0 => candidate == Base
任意 q_ij      => delta_log_scale = 0 and scale == base_scale
```

该阶段对应：

```text
configs/psb_marl/p2_1_s_mean_only.json
outputs/psb_marl/p2_1_s_mean_only/
```

## 1. 结构合同

P2.1-S 使用：

```json
"conditioning_mode": "causal_q_gate",
"max_delta_log_scale": 0.0
```

`max_delta_log_scale=0` 不是软惩罚。适配器会切换到结构化 mean-only 模式：

- 最后一层只输出 `action_dim` 个均值参数，不构造 log-scale 输出头；
- 因果门同样只输出 `action_dim` 个均值门；
- 前向直接返回 `scale = base_scale`；
- `delta_log_scale` 是逐元素精确为零的诊断张量；
- PPO 动作梯度仍可从均值经过因果门、分支编码器、近端根和反对称控制器传播；
- Base Actor 继续冻结。

P2.1-C 的均值+尺度适配器保留为消融组。由于网络输出维度和 runtime contract 不同，
P2.1-S 不能从 P2.1-C checkpoint 恢复，必须从已验证的 P1/Base 创建新 run。

## 2. 自动验证

```bash
conda activate sigmarl-nod

python -m unittest discover -s tests/psb_marl -v
```

测试覆盖：

1. P2.1-C 与 P2.1-S 配置和输出目录隔离；
2. 负的尺度上限会被配置层拒绝；
3. mean-only adapter 不包含尺度输出参数；
4. 非零分支能修改均值，但尺度逐元素等于 Base；
5. 均值动作损失仍能穿过近端层到达控制器；
6. promotion 结构门会拒绝任何尺度漂移；
7. 原 P0/P1/P2/P2.1-C 回归测试继续通过。

## 3. 单轮流程验证

```bash
python main_training.py \
  --config configs/psb_marl/p2_1_s_mean_only.json \
  --iterations 1
```

记录终端输出的 `<RUN_DIR>`，然后执行：

```bash
python main_testing.py \
  --config configs/psb_marl/p2_1_s_mean_only.json \
  --run-dir <RUN_DIR> \
  --checkpoint <RUN_DIR>/candidate_policy.pth \
  --scenario intersection_2 \
  --max-steps 64 \
  --episodes 1 \
  --seeds 101 \
  --no-render \
  --compare-base
```

单轮结构验证应满足：

```text
rollout_delta_log_scale_abs_mean == 0
rollout_delta_log_scale_abs_max == 0
rollout_scale_matches_base_exactly == true
rollout_zero_branch_bypass_loc_abs_mean == 0
rollout_zero_branch_bypass_log_scale_abs_mean == 0
rollout_branch_log_scale_effect_abs_mean == 0
base_actor_frozen == true
rollout_z_antisymmetry_error <= 1e-6
rollout_max_root_residual <= 1e-6
rollout_min_root_denominator > 0
```

第一轮 adapter 仍从零初始化，所以 `rollout_branch_loc_effect_abs_mean` 接近零是正常的。

## 4. 30 轮 pilot

不要从单轮 smoke checkpoint 恢复成 P2.1-C，也不要从已有 P2.1-C run 恢复。若希望
在同一 P2.1-S run 上继续：

```bash
python main_training.py \
  --config configs/psb_marl/p2_1_s_mean_only.json \
  --resume <RUN_DIR>/latest_checkpoint.pt \
  --iterations 30
```

30 轮后重点比较前 10 轮和后 10 轮：

- 尺度相关指标必须始终严格为零；
- `rollout_branch_loc_effect_abs_mean` 应从零增长；
- `rollout_zero_branch_bypass_loc_abs_mean` 必须始终为零；
- 后 10 轮碰撞率不应高于前 10 轮；
- 奖励不应出现持续下降趋势。

## 5. 与 Base 的正式配对比较

候选通过结构检查后，再执行至少 5 个 paired seeds：

```bash
python main_testing.py \
  --config configs/psb_marl/p2_1_s_mean_only.json \
  --run-dir <RUN_DIR> \
  --checkpoint <RUN_DIR>/candidate_policy.pth \
  --scenario intersection_2 \
  --max-steps 600 \
  --episodes 4 \
  --seeds 101 102 103 104 105 \
  --no-render \
  --compare-base
```

只有奖励下置信界、碰撞上置信界和 `base_scale_exactly_preserved` 全部通过，候选才具备
promotion 资格。确认报告后，再单独加 `--promote-if-noninferior`；默认部署始终保留
Base fallback。

若尺度冻结后剩余碰撞主要来自车道边界，下一步先执行
[`P2_1_PROJECTED_COUNTERFACTUAL.md`](P2_1_PROJECTED_COUNTERFACTUAL.md) 定义的
longitudinal-only 推理反事实。该测试复用当前 checkpoint，不需要重新训练，也不能用于
promotion。
