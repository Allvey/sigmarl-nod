# PSB-MARL P2.1-D：Bifurcation Diagnostics

P2.1-D 只增加诊断，不改变 P2 的策略前向、动作采样、奖励、PPO 损失、梯度路径、
optimizer 或 checkpoint schema。因此它仍使用：

```text
stage = p2_frozen_base_bifurcation
```

本阶段用于区分三种容易混淆的现象：冲突边太稀疏、分岔状态没有形成，以及动作适配器
绕过分岔状态直接修改 Base 动作。

## 1. 活跃边与临界区诊断

每次 rollout 只在无序边 `i < j` 上统计，避免同时计算 `(i,j)`、`(j,i)` 和对角线：

| 指标 | 含义 |
|---|---|
| `rollout_active_edge_fraction` | `rho > 0` 的边时刻占全部无序边时刻的比例 |
| `rollout_critical_edge_fraction` | `rho > rho_c` 的边时刻占全部无序边时刻的比例 |
| `rollout_critical_given_active_fraction` | 活跃边中进入超临界区的比例 |
| `rollout_active_b_abs_mean` | 仅活跃边上的平均 `abs(b)` |
| `rollout_active_z_abs_mean` | 仅活跃边上的平均 `abs(z)` |
| `rollout_active_q_abs_mean` | 仅活跃边上的平均 `abs(tanh(z/z_scale))` |
| `rollout_critical_*_abs_mean` | 仅超临界活跃边上的对应幅值 |

这些指标不能与旧的稠密矩阵全局均值直接互换。旧指标仍保留，以便兼容已有报告。

## 2. 分支承诺诊断

默认以 `abs(q) >= 0.5` 定义已形成连续分支承诺：

- `rollout_committed_given_active_fraction`：活跃边中已承诺的比例；
- `rollout_branch_switch_rate`：相邻状态均已承诺时发生符号翻转的比例；
- `rollout_branch_dwell_mean_steps`：同一符号承诺的平均连续步数；
- `rollout_branch_dwell_max_steps`：最长连续承诺步数。

当 `rollout_branch_switch_eligible_samples=0` 时，切换率记为 `0`，但不能解释为已经稳定，
因为此时可能根本没有形成承诺。

## 3. `q=0` 反事实旁路诊断

训练 rollout 完成后，在 `torch.no_grad()` 下保持相同 observation、冲突特征、`rho`、
Base 分布和网络参数，只将分支输入设为 `q=0`，重新计算一次 branch encoder 与 adapter。
该反事实前向不替换采样动作，也不进入 PPO 损失。

```text
bypass = abs(policy(q=0) - Base)
branch_effect = abs(policy(q_actual) - policy(q=0))
```

对应指标：

- `rollout_zero_branch_bypass_loc_abs_mean`；
- `rollout_zero_branch_bypass_log_scale_abs_mean`；
- `rollout_branch_loc_effect_abs_mean`；
- `rollout_branch_log_scale_effect_abs_mean`；
- `rollout_branch_loc_dependency_ratio`；
- `rollout_branch_log_scale_dependency_ratio`。

依赖比例定义为：

```text
branch_effect / (bypass + branch_effect)
```

当前通用 adapter 并不保证 `q=0` 时恢复 Base，因此 P2.1-D 的目标是测量问题，而不是
强行让这些指标通过。后续 P2.1-C 才负责施加分岔因果门控。

## 4. 自动验证

```bash
conda activate sigmarl-nod

python -m unittest tests.psb_marl.test_p2_frozen_base -v
```

测试检查：

1. 活跃边只统计一次，且不包含对角线；
2. 临界比例、切换率和驻留步数的口径正确；
3. `q=0` 反事实能够检出通用 adapter 的旁路；
4. 反事实前向不改变策略参数、rollout 分布参数或梯度；
5. 原 P2 的近端、反对称、序列重算、梯度和 promotion 测试继续通过。

## 5. 手动入口验证

先进行一次独立的单轮 smoke：

```bash
python main_training.py \
  --config configs/psb_marl/p2_frozen_base_bifurcation.json \
  --iterations 1
```

在输出的 `<RUN_DIR>/metrics.json` 最后一轮检查上述 `rollout_*` 指标均存在且为有限值。
训练完成后可运行结构测试：

```bash
python main_testing.py \
  --config configs/psb_marl/p2_frozen_base_bifurcation.json \
  --run-dir <RUN_DIR> \
  --checkpoint <RUN_DIR>/candidate_policy.pth \
  --scenario CPM_mixed \
  --max-steps 64 \
  --episodes 1 \
  --seeds 101 \
  --no-render
```

`p2_manual_validation.json` 会记录活跃边、临界区、承诺与切换诊断。测试入口不执行
adapter 的 `q=0` 反事实；该指标需要训练入口持有未封装的 policy bridge，因此以
`metrics.json` 为准。

P2.1-D 不定义性能通过门，也不允许据此 promotion。性能结论仍必须使用正式的多 seed
Base/P2 配对协议。
