# PSB-MARL P2.1-C：Causal Branch Gate

P2.1-C 将 P2 的通用分布适配器收紧为 Base 锚定的分岔因果策略类。核心合同是：

```text
所有局部分支 q = 0
    => branch context = 0
    => distribution gate = 0
    => delta_loc = delta_log_scale = 0
    => candidate distribution 与 Base distribution 逐元素完全相同
```

P2.1-D、原 P2 和 P2.1-C 使用不同的输出目录。原 P2 保留为允许旁路的消融组；
P2.1-C 使用：

```text
configs/psb_marl/p2_1_c_causal_branch.json
outputs/psb_marl/p2_1_c_causal_branch/
```

新配置显式包含：

```json
"conditioning_mode": "causal_q_gate"
```

该字段进入 runtime config 和 checkpoint 合同。因此旧 P2 checkpoint 不能作为 P2.1-C
恢复点或候选策略使用，P2.1-C 必须重新从已验证的 P1/Base 初始化。

## 1. 因果门控结构

首先计算连续分支坐标：

\[
q_{ij}=\tanh(z_{ij}/z_0).
\]

边编码器只用 `abs(q)` 描述分支强度，再显式乘以带符号的 `q`：

\[
e_{ij}=E_\theta(\chi_{ij},|q_{ij}|,\rho_{ij}),
\qquad
m_{ij}=q_{ij}e_{ij},
\]

\[
c_i=\sum_j\alpha_{ij}m_{ij}.
\]

动作分布适配器保留 observation 和 Base 分布作为分支内控制条件，但其输出必须经过
无偏置的分支门：

\[
r_i=N_\theta(o_i,c_i,\mu_i^{\mathrm{base}}),
\qquad
g_i=\tanh(W_gc_i),
\]

\[
\Delta\eta_i=B\tanh(r_i\odot g_i),
\qquad
\eta_i=\eta_i^{\mathrm{base}}+\Delta\eta_i,
\]

其中 `eta=(loc, log_scale)`。`W_g` 不使用 bias，所以 `c_i=0` 时门严格为零；即使
`N_theta` 学到了 observation 旁路或非零 bias，也不能修改 Base 分布。

该结构没有规定 `q>0` 必须加速或 `q<0` 必须减速。具体动作方向仍由共享 MARL 网络
根据 observation、Base 行为和分支上下文学习。

## 2. 梯度合同

非零分支上的动作策略梯度保持：

```text
action log-prob
  -> bounded distribution correction
  -> causal gate/context
  -> q = tanh(z/z_scale)
  -> differentiable proximal root
  -> antisymmetric b controller
```

adapter 最后一层仍为零初始化，因此训练初始候选分布等于 Base。第一批非零 `q` 样本
先更新 adapter 输出层；随后 PPO minibatch 和后续 rollout 可沿上述路径更新分支编码器
与控制器。能量正则的显式路径从开始即存在。

## 3. 自动验证

```bash
conda activate sigmarl-nod

python -m unittest discover -s tests/psb_marl -v
```

新增测试证明：

1. 旧 P2 与 P2.1-C runtime config 和输出目录严格不同；
2. 即使 adapter 参数全部非零，`q=0` 时仍逐元素恢复 Base；
3. 非零 `q` 能产生学习得到的互补动作变化；
4. 动作损失梯度能穿过门控和近端层到达控制器；
5. Base Actor 始终无梯度；
6. 原 P0/P1/P2 测试继续通过。

## 4. 单轮 smoke

```bash
python main_training.py \
  --config configs/psb_marl/p2_1_c_causal_branch.json \
  --iterations 1
```

完成后检查 `<RUN_DIR>/metrics.json`：

```text
rollout_zero_branch_bypass_loc_abs_mean == 0
rollout_zero_branch_bypass_log_scale_abs_mean == 0
base_actor_frozen == true
rollout_z_antisymmetry_error <= 1e-6
rollout_max_root_residual <= 1e-6
rollout_min_root_denominator > 0
```

第一轮因 adapter 零初始化，`branch_effect` 可能仍接近零；这不是失败。20–30 轮 pilot
后应同时满足旁路仍严格为零、`branch_effect>0`，并观察分支依赖比例趋向 1。

测试候选完整性：

```bash
python main_testing.py \
  --config configs/psb_marl/p2_1_c_causal_branch.json \
  --run-dir <RUN_DIR> \
  --checkpoint <RUN_DIR>/candidate_policy.pth \
  --scenario CPM_mixed \
  --max-steps 64 \
  --episodes 1 \
  --seeds 101 \
  --no-render
```

## 5. 正式训练与比较

正式训练必须创建新 run，不能从旧 P2 checkpoint 恢复：

```bash
python main_training.py \
  --config configs/psb_marl/p2_1_c_causal_branch.json
```

完成后按相同场景、episodes 和 seeds 与 Base 配对：

```bash
python main_testing.py \
  --config configs/psb_marl/p2_1_c_causal_branch.json \
  --run-dir <RUN_DIR> \
  --scenario CPM_mixed \
  --max-steps 600 \
  --episodes 4 \
  --seeds 101 102 103 104 105 \
  --no-render \
  --compare-base
```

只有配对非劣界和结构检查同时通过后才能使用 `--promote-if-noninferior`。单轮 smoke、
训练曲线或分支依赖比例都不能单独作为 promotion 依据。

若训练诊断显示策略主要通过扩大动作尺度而不是修正动作均值介入，则使用
[`P2_1_MEAN_ONLY.md`](P2_1_MEAN_ONLY.md) 定义的 P2.1-S。P2.1-C 保留为完整分布适配
消融组，不再继续扩大其尺度上限。
