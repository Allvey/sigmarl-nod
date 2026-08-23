# SigmaRL 1.2.0 网络与训练结构

> 事实来源：tag `1.2.0` 的 `utilities/mappo_cavs.py`  
> 默认配置：4 agents、32 维单车观测、2 维连续动作  
> 对齐日期：2026-08-23

## 1. 总体结构

SigmaRL 1.2.0 默认采用 MAPPO：

```text
每辆车的局部 observation [32]
              │
              ▼
共享、分散的 Actor
              │
              ▼
loc [2] + scale [2]
              │
              ▼
TanhNormal → action [2]

训练期所有车辆 observation [4,32]
              │
              ▼
共享、中心化的 Critic
              │
              ▼
每辆车 state value [1]
```

Actor 与 Critic 都使用 `torchrl.modules.MultiAgentMLP`，参数在同类车辆间共享。

## 2. Actor

默认 Actor 配置：

```text
n_agent_inputs  = 32
n_agent_outputs = 2 × action_dim = 4
centralised     = false
share_params    = true
depth           = 2
num_cells       = 256
activation      = Tanh
```

可形象表示为：

```text
32
 ↓
Linear(32,256) + Tanh
 ↓
Linear(256,256) + Tanh
 ↓
Linear(256,4)
 ↓
NormalParamExtractor
 ├── loc   [2]
 └── scale [2]
 ↓
TanhNormal(action_low, action_high)
 ↓
action [2] + sample_log_prob
```

“分散”表示每辆车动作只由它自己的 32 维观测决定；“共享参数”表示四辆车调用同一
套 Actor 权重，而不是每辆车拥有独立网络。

## 3. Critic

默认 Critic 配置：

```text
n_agent_inputs  = 32
n_agent_outputs = 1
centralised     = true
share_params    = true
depth           = 2
num_cells       = 256
activation      = Tanh
```

中心化 Critic 在训练时组合四辆车的信息，单次价值估计的有效联合输入规模为
`4 × 32 = 128`。输出仍按智能体组织为每车一个 state value。

Critic 不参与执行期动作生成。

## 4. PPO

训练使用 TorchRL `ClipPPOLoss`：

```text
loss = loss_objective + loss_critic + loss_entropy
```

同时使用：

- GAE，`gamma=0.99`、`lambda=0.9`；
- PPO clip epsilon `0.2`；
- entropy coefficient `1e-4`；
- Adam，初始学习率 `2e-4`；
- gradient norm clip `1.0`；
- 线性学习率衰减至 `1e-5`。

每轮 rollout 先计算 GAE，再执行：

```python
data_view = tensordict_data.reshape(-1)
```

随后从 4096 frames 中无放回采样 512-frame minibatch。默认每 epoch 8 个 minibatch，
60 epochs，因此每轮 480 次向量化更新。

## 5. 1.2.0 中的可选 priority 代码

源码中包含可选 `PriorityModule`，但根配置为：

```text
is_using_prioritized_marl = false
is_prb = false
is_using_opponent_modeling = false
```

所以 SigmaRL Base-MAPPO 默认运行不构造 priority policy/critic，也不使用 prioritized
replay。新的 Opinion-MARL 应继续关闭这些路径。它们属于原仓库附带的可选功能，不是
新方法组件。

## 6. Opinion-MARL 的目标网络结构

重建后在 Base Actor 旁路增加 Opinion 模块：

```text
local observation ───────────────→ SigmaRL Base Actor ─→ base loc/scale
        │
        └→ pair features → EvidenceNet → b
                                      ↓
previous z → Fixed Opinion Dynamics → current z
                                      ↓
                             bounded speed residual
                                      ↓
base loc(speed) + residual ─────────→ final loc
base loc(steer) ────────────────────→ unchanged
base scale ─────────────────────────→ unchanged
                                      ↓
                                  TanhNormal
```

Opinion 第一版的模块职责：

| 模块 | 是否学习 | 是否有时间状态 |
|---|---|---|
| SigmaRL Base Actor | Base/Joint 学习 | 否 |
| Centralized Critic | 各阶段学习 | 否 |
| EvidenceNet | Evidence/Joint 学习 | 否 |
| OpinionDynamics | 固定 | `z` 由 Collector 持有 |
| Residual mapping | 固定或显式受限 | 否 |

第一版 Evidence/Joint Critic 仍只读取 SigmaRL 原始联合 observation，不追加 `z`，
因此其输入结构与 Base Critic 保持一致。Opinion-aware Critic 只作为后续消融，不是
主方法的必要组件。

## 7. 三阶段训练与速度边界

### Base

必须使用 1.2.0 的扁平化 MAPPO 路径。Base 不构造 pair features，不维护 `z`，不进入
sequence buffer。

### Evidence

加载 Base Actor，冻结 Actor；训练 EvidenceNet 和 Critic。由于 `z_t` 依赖
`z_{t-1}`，必须保留时间连续 chunk。

### Joint

解冻 Base Actor，与 EvidenceNet 和 Critic 联合优化；固定动力学仍不进入 optimizer。

Evidence/Joint 的高效实现应把多个 chunk 堆叠为 batch：chunk 维并行，只在 chunk
内部时间维循环。不能为 minibatch 中每个 chunk、每个时间步分别调用一遍完整网络。

## 8. checkpoint 兼容原则

Base checkpoint 应至少分别保存：

```text
base_actor_state
critic_state
resolved_config
iteration
```

Evidence/Joint checkpoint 再增加：

```text
evidence_state
stage
optimizer_states
checkpoint schema version
```

Base checkpoint 不应依赖 EvidenceNet 的隐藏层维度，这样修改意见模块时无需重新训练
完全相同的 Base。
