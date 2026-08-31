# PSB-MARL P1：Zero-Control Equivalence

P1 将固定参数的近端饱和分岔状态接入真实道路 rollout，但不允许它改变 Base Actor：

```text
局部车辆对信息 ──> rho ──> Proximal PSB state z ──> 仅记录诊断
                                                    │
Base observation ──> 未修改的 Base Actor ──────────> action
```

本阶段固定：

- `b_max = 0`，不存在学习分岔控制；
- `actor_context_gain = 0`，Actor 不读取 `z`；
- Base policy/critic checkpoint 与 P0、Base 字节完全相同；
- PSB 可训练参数为零，不执行 PPO；
- 每条无序边只维护一个状态，稠密表示严格反对称。

P1 sidecar 中的 `n_agents` 只记录 Base 来源场景。Base Actor 使用共享参数的分散
策略，P1 状态 tracker 也按当前环境动态构造，因此测试场景可以使用不同车辆数；例如
从 4 车 `CPM_mixed` Base 部署到 6 车 `intersection_2`。

因此 P1 的非劣条件不是“平均奖励接近”，而是同一 seed 下 P1 与 Base 的 action 和
reward 张量逐元素完全一致。

## 1. 近端层

P1 求解：

\[
\frac{z^{t+1}-z^t}{h_z}
+\kappa z^{t+1}
-\rho^t\nu\tanh(\alpha z^{t+1})=0,
\qquad h_z=\frac{\Delta t}{\tau_z}.
\]

配置加载时强制检查：

\[
m_P=\frac1{h_z}+\kappa-\rho_{\max}\nu\alpha>0.
\]

前向使用理论显式根区间上的 safeguarded Newton–bisection，直到根残差不超过
`residual_tolerance`；反向使用收敛根处的隐式 Jacobian，而不是对固定次数 Newton
迭代展开求导。

`main_training.py` 会在不启动环境的情况下认证：

- 强凸裕度；
- 最大根残差；
- 零控制奇对称性；
- `rho=0` 时的解析收缩；
- 冻结输入下的近端能量不等式；
- 隐式梯度与理论公式的一致性。

认证结果写入 `p1_certification.json`。

## 2. 阶段封装

P1 只接受已经完成人工道路验证的 P0 run：

```bash
conda activate sigmarl-nod

python main_training.py \
  --config configs/psb_marl/p1_zero_control_equivalence.json
```

该命令不会执行 PPO，通常很快完成。终端会输出 `<RUN_DIR>`，其中应包含：

```text
final_policy.pth
final_critic.pth
final_psb_layer.pth
final_checkpoint.pt
p1_certification.json
p1_equivalence.json
deployment_manifest.json
```

检查：

```bash
cat <RUN_DIR>/p1_certification.json
cat <RUN_DIR>/p1_equivalence.json
```

要求 `passed=true`、所有 `checks=true`、`actor_context_gain=0`、`b_max=0`，且三个
checkpoint 的 manifest/equivalence 哈希链完整。

## 3. 快速配对测试

```bash
python main_testing.py \
  --config configs/psb_marl/p1_zero_control_equivalence.json \
  --run-dir <RUN_DIR> \
  --scenario intersection_2 \
  --max-steps 64 \
  --episodes 2 \
  --seeds 101 \
  --no-render \
  --compare-base
```

P1 和 Base 会分别使用相同 seed 运行一次。`p1_manual_validation.json` 必须满足：

```text
noninferiority_result = proven_by_exact_paired_actions
actions_exactly_equal = true
rewards_exactly_equal = true
max_abs_action_difference = 0
max_abs_reward_difference = 0
max_abs_b = 0
max_antisymmetry_error = 0
nonfinite_action_count = 0
nonfinite_reward_count = 0
nonfinite_z_count = 0
```

若 action 或 reward 任一元素不同，测试会写出失败报告并以非零状态退出。

## 4. 可视化与正式验证

单环境可视化：

```bash
python main_testing.py \
  --config configs/psb_marl/p1_zero_control_equivalence.json \
  --run-dir <RUN_DIR> \
  --scenario intersection_2 \
  --max-steps 600 \
  --episodes 1 \
  --seeds 101 \
  --render \
  --compare-base
```

多 seed 无渲染验证：

```bash
python main_testing.py \
  --config configs/psb_marl/p1_zero_control_equivalence.json \
  --run-dir <RUN_DIR> \
  --scenario CPM_mixed \
  --max-steps 600 \
  --episodes 4 \
  --seeds 101 102 103 104 105 \
  --no-render \
  --compare-base \
  --promote-if-noninferior
```

P1 通过后才能进入 P2。P2 将冻结 Base Actor，开放有界反对称控制网络及非零分岔
状态，但仍保留 Base 回退部署门。
