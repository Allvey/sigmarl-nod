# Opinion Dynamics + MARL：详细实施与跨 Session 交接指南

> 文档状态：实施前设计基线  
> 最后核对日期：2026-08-14  
> 适用仓库：`sigmarl-traffic`  
> 理论依据：`opinion_dynamics_marl_technical_route.md`  
> 用途：供后续 Codex/开发者在新 session 中快速恢复上下文、分阶段修改代码和验证结果。

---

## 0. 新 Session 最短启动说明

如果在新 session 中继续实现，请先把下面这段话发给 Codex：

```text
请阅读 OPINION_MARL_IMPLEMENTATION_GUIDE.md 和
opinion_dynamics_marl_technical_route.md，并严格按实施指南继续。

新方法是独立的“意见动力学 + MARL”理论，TSC 仅作为代码载体和外部基线，
不要把方法重新表述为 TSC 的扩展。请先检查 git status、当前依赖环境和文档中的
里程碑状态，然后只实施下一个尚未完成的里程碑。不要覆盖用户已有修改。
```

新 session 开始后必须依次执行：

1. 阅读本文件全文；
2. 阅读 `opinion_dynamics_marl_technical_route.md` 全文；
3. 执行 `git status --short`，保护用户已有修改；
4. 检查当前 Python/依赖是否已经修复；
5. 从第 16 节的里程碑清单中选择下一个未完成项；
6. 实施后更新本文件的里程碑状态和验证记录。

---

## 1. 方法定位

### 1.1 正确定位

本项目要实现的是一个独立的 **Opinion Dynamics + MARL** 框架：

\[
\boxed{
\text{MARL 学习瞬时意见证据}
+
\text{固定非线性动力学形成长期意见}
+
\text{意见显式修正动作分布}
}
\]

完整因果链为：

\[
o_i^t
\rightarrow
\chi_{ij}^t
\rightarrow
B_{\phi_b}
\rightarrow
b_{ij}^t
\rightarrow
z_{ij}^t
\rightarrow
q_{ij}^t
\rightarrow
\Delta\mu_{i,\mathrm{op}}^t
\rightarrow
\pi_i(a_i^t).
\]

其中：

- `MARL` 学习“什么物理情境应该产生什么方向和强度的证据”；
- `OpinionDynamics` 决定“证据如何积累、遗忘、自强化、切换和消退”；
- `Actor` 根据基础动作均值与意见残差形成最终动作分布；
- `CentralizedCritic` 只在训练期评价长期任务回报。

### 1.2 与 TSC 的关系

TSC 只承担以下角色：

- 提供 VMAS 道路交通环境；
- 提供车辆动力学、观测、奖励、并行环境和 MAPPO 基础代码；
- 提供可复用的局部车辆特征、mask、索引和 checkpoint 工程代码；
- 作为后续实验中的外部协调方法基线。

新方法在理论上不依赖：

- TSC priority labels；
- node priority score；
- priority Top-K；
- leader set；
- leader-action predictor；
- Stackelberg-conditioned Critic；
- total order；
- topology consistency loss。

代码实现时即使参考或复制 TSC 文件中的通用 MLP、mask、TensorDict 操作，也不能把这些 TSC 概念重新带入新方法表述。

---

## 2. 不可更改的理论约束

以下内容是实现合同。工程微调不能违反这些约束。

### 2.1 瞬时证据与动态意见必须分离

\[
b_{ij}^t=\text{当前帧的物理证据},
\qquad
z_{ij}^t=\text{跨时间的协调意见}.
\]

- `OpinionEvidenceNet` 不保存时间状态；
- `OpinionEvidenceNet` 禁止读取 `z` 或 `q`；
- 时间记忆只能由显式 `OpinionDynamics` 提供。

### 2.2 当前阶段只学习证据，不学习动力学参数

固定：

\[
\eta_z,\kappa_z,\nu_z,\alpha_z,z_0.
\]

这些参数：

- 使用 `register_buffer` 或普通不可训练配置保存；
- 不得进入任何 optimizer；
- 单元测试必须验证其 `requires_grad=False`。

### 2.3 证据必须有界并受物理门控

先计算未门控证据：

\[
\widetilde b_{ij}^t
=
b_{\max}
\tanh\left(\frac{\ell_{ij}^t}{T_b}\right),
\]

再计算最终证据：

\[
b_{ij}^t
=
\rho_{ij}^t c_{ij}^t\widetilde b_{ij}^t.
\]

其中：

- \(\rho\in[0,1]\)：固定物理冲突紧迫度；
- \(c\in[0,1]\)：可见性/预测置信度；
- 无冲突、不可见或无效边必须产生接近零的最终证据。

### 2.4 意见必须显式、有界地影响动作

\[
q_{ij}^t=\tanh(z_{ij}^t/z_0),
\]

\[
\Delta\mu_{i,\mathrm{op}}^t
=
c_{\mathrm{op}}
\sum_j\bar\rho_{ij}^tq_{ij}^t\mathbf d_{ij}^t.
\]

第一版：

\[
\mathbf d_{ij}=[1,0],
\]

即只修改速度动作的分布均值，不修改转向均值和动作方差。

### 2.5 Critic 梯度必须隔离

- 第一版 Critic 不读取 `z`；
- 后续若读取，只能读取 `detach(z)`；
- Critic optimizer 与 Actor/Evidence optimizer 分离；
- Critic loss 反向后，EvidenceNet 所有梯度必须为 `None` 或零。

### 2.6 训练必须保留连续序列

禁止在重放意见状态之前把 rollout 完全展平并随机采样单步。

必须：

- 保存每个片段起点的 `z_init`；
- 在片段内部重新计算 `b -> z -> q -> loc -> log_prob`；
- 对片段起点 `z_init` 停止梯度；
- 在片段内部执行 truncated BPTT。

---

## 3. 当前仓库事实快照

### 3.1 当前训练入口

- `main_training.py` 加载 `config.json`；
- 调用 `utilities/mappo_cavs.py::mappo_cavs()`；
- `mappo_cavs.py` 同时包含环境创建、Actor/Critic、TSC、PPO、日志和保存逻辑，长度较大；
- 不建议继续在该函数内部增加 Opinion-MARL 大分支。

### 3.2 当前 Actor

当前 Actor 结构：

```text
MultiAgentMLP
→ NormalParamExtractor
→ loc, scale
→ TanhNormal
→ action + sample_log_prob
```

关键位置：`utilities/mappo_cavs.py` 约第 208～251 行。

工程含义：

- 意见残差必须加到 `NormalParamExtractor` 输出的 `loc`；
- 必须在 `TanhNormal` 构造和 log-prob 计算之前加入；
- 不能采样后直接修改 action，否则 PPO 的 log-prob 与执行动作不一致。

### 3.3 当前训练数据问题

当前 PPO 在获得 rollout 后执行：

```python
data_view = tensordict_data.reshape(-1)
replay_buffer.extend(data_view)
```

随后使用随机单步 minibatch。

这会破坏：

\[
b^t\rightarrow z^t\rightarrow z^{t+1}.
\]

因此新训练管线必须保留时间维并按连续 chunk 采样。

### 3.4 当前 Collector

`SyncDataCollectorCustom.rollout()` 已逐时间步调用 policy、执行环境并堆叠时间维。可复用其主循环，但需要新建 `OpinionSyncDataCollector` 或一个最小子类来显式传递 `z_prev/z_next`。

### 3.5 当前邻车索引

场景已经能产生全局 neighbor index，例如：

```text
topology_neighbors_indices
neighbors_indices
```

但这些名字带有 TSC 语义或受现有配置条件影响。新方法应新增：

```text
opinion_neighbor_ids
```

其值可以复用现有索引构造代码，但接口必须独立。

### 3.6 当前 reset

`road_traffic.py::reset_world_at()` 支持：

- 整个环境 reset；
- 指定环境中的单个 agent reset。

因此 Opinion 状态必须支持：

- 环境 reset：清空对应环境全部 `z`；
- agent `k` reset：清空 `z[:,k,:]` 和 `z[:,:,k]`。

### 3.7 当前默认配置与新方法冲突

当前 `config.json` 中包括：

```json
{
  "is_using_opponent_modeling": true,
  "use_topology_neighbor_selection": true,
  "topology_loss_weight": 0.5,
  "topology_selection_threshold": 0.0
}
```

新 Opinion-MARL 配置必须关闭这些路径。

### 3.8 当前依赖环境状态

2026-08-14 检查结果：

- Python：3.9.13；
- `pytest`：7.1.2；
- `torch` 导入失败：当前 `typing_extensions` 缺少 `TypeIs`；
- `torchrl`、`tensordict` 同样导入失败；
- `vmas` 未安装；
- `requirements.txt` 要求 Torch 2.1.0、TorchRL 0.2.1、TensorDict 0.2.1、VMAS 1.4.1。

实际编码前必须先创建与项目隔离的兼容虚拟环境。不要在未知环境中边修代码边猜 API 行为。

### 3.9 工作区状态

检查时分支名为：`论文修改`。

以下设计文档是用户文件，必须保留：

- `Opinion_Dynamics_MARL_Design(1).md`；
- `opinion_dynamics_marl_technical_route.md`。

任何新 session 都必须先执行 `git status --short`，不得覆盖用户未提交修改。

---

## 4. 推荐文件架构

### 4.1 新增文件

```text
utilities/opinion/
├── __init__.py
├── config.py
├── conflict_graph.py
├── evidence_net.py
├── dynamics.py
├── residual.py
├── policy.py
├── collector.py
├── sequence_buffer.py
├── ppo_loss.py
├── trainer.py
├── checkpoint.py
└── diagnostics.py

main_training_opinion.py
main_testing_opinion.py
config_opinion.json
```

### 4.2 新增测试

```text
tests/opinion/
├── test_conflict_graph.py
├── test_evidence_net.py
├── test_opinion_dynamics.py
├── test_opinion_residual.py
├── test_opinion_policy.py
├── test_opinion_state_reset.py
├── test_sequence_buffer.py
├── test_recompute_log_prob.py
└── test_gradient_isolation.py
```

### 4.3 现有文件修改范围

| 文件 | 允许修改 | 禁止引入 |
|---|---|---|
| `utilities/helper_training.py` | 新增 `use_opinion_marl`、`opinion_config`；必要的通用 helper | Opinion 理论逻辑塞入现有 TSC 类 |
| `scenarios/road_traffic.py` | 新增局部车辆对特征、紧迫度、置信度、reset mask 输出 | 在 observation/info 中更新 `z` |
| `utilities/mappo_cavs.py` | 尽量不改；最多提取真正通用的环境 helper | 新增大段 Opinion 训练分支 |
| `main_training.py` | 不改 | 改成 Opinion 专用入口 |
| `main_testing.py` | 不改 | 继续堆叠 Opinion 日志 |
| `utilities/topology_*` | 保留 TSC 基线 | 改名后冒充新方法模块 |

---

## 5. 张量与 TensorDict 接口合同

设：

- \(E\)：并行环境数量；
- \(T\)：rollout 时间长度；
- \(N\)：车辆数量；
- \(K_c\)：Opinion 冲突候选数量；
- \(D_\chi\)：车辆对特征维度；
- \(A=2\)：动作维度（速度、转向）。

### 5.1 环境输出

当前时刻：

```text
agents.observation                         [E, N, Dobs]
agents.info.opinion_neighbor_ids           [E, N, Kc]       int64
agents.info.opinion_valid_mask             [E, N, Kc]       bool
agents.info.opinion_pair_features          [E, N, Kc, Dχ]
agents.info.opinion_urgency                 [E, N, Kc]
agents.info.opinion_confidence              [E, N, Kc]
agents.info.agent_reset_mask                [E, N]           bool
```

### 5.2 Policy 输入状态

```text
agents.opinion_z                            [E, N, N]
```

定义：

```text
opinion_z[e, i, j] = 环境 e 中车辆 i 对车辆 j 的意见
```

对角线恒为零。

### 5.3 Policy 输出

```text
agents.loc                                  [E, N, A]
agents.scale                                [E, N, A]
agents.action                               [E, N, A]
agents.sample_log_prob                      [E, N] 或 [E, N, 1]
agents.opinion_z_next                       [E, N, N]
agents.opinion_raw_b                        [E, N, Kc]
agents.opinion_b                            [E, N, Kc]
agents.opinion_q                            [E, N, Kc]
agents.opinion_residual                     [E, N]
```

### 5.4 Rollout 保存

```text
agents.opinion_z                            [E, T, N, N]
agents.opinion_z_next                       [E, T, N, N]
agents.info.opinion_*                       [E, T, N, ...]
agents.action                               [E, T, N, A]
agents.sample_log_prob                      [E, T, N]
agents.advantage                            [E, T, N, 1]
agents.value_target                         [E, T, N, 1]
```

### 5.5 Chunk 输出

```text
chunk_z_init                                [M, N, N]
chunk_observation                           [M, L, N, Dobs]
chunk_pair_features                         [M, L, N, Kc, Dχ]
chunk_actions                               [M, L, N, A]
chunk_old_log_prob                          [M, L, N]
chunk_advantage                             [M, L, N, 1]
chunk_reset_mask                            [M, L, N]
```

---

## 6. 数学实现细节

### 6.1 冲突候选

第一版：

\[
K_c=\min(N-1,K_{\max}).
\]

当前 `N=4` 时建议直接使用全部其他 3 辆车。

候选选择只决定需要计算哪些车辆对，不决定通行权，也不产生 leader/follower 结构。

### 6.2 车辆对特征

建议第一版：

\[
\chi_{ij}=
[r_x,r_y,
\Delta v_x,\Delta v_y,
\Delta\psi,
v_i,v_j,
d_{ij},
t_{\mathrm{CPA}},
d_{\mathrm{CPA}},
\Delta\tau_{ij},
r_{\mathrm{rule}},
c_{ij}].
\]

要求：

- 使用 ego 坐标系；
- 所有连续量先裁剪后归一化；
- `mask`、padding、不可见边不能形成非零物理证据；
- 不使用仿真真实未来轨迹；
- 不使用邻车未执行动作；
- 不使用中央 Critic 或全局状态。

### 6.3 固定紧迫度

相对位置和速度：

\[
r=p_j-p_i,
\qquad
v_{\mathrm{rel}}=v_j-v_i.
\]

最近接近时间：

\[
t_{\mathrm{CPA}}
=
\operatorname{clip}
\left(
-\frac{r^\top v_{\mathrm{rel}}}
{\|v_{\mathrm{rel}}\|^2+\epsilon},
0,H
\right).
\]

最近接近距离：

\[
d_{\mathrm{CPA}}
=
\|r+t_{\mathrm{CPA}}v_{\mathrm{rel}}\|.
\]

紧迫度初始实现：

\[
\rho_{ij}
=
m_{ij}
\exp(-t_{\mathrm{CPA}}/\tau_\rho)
\sigma\left(
\frac{d_{\mathrm{safe}}-d_{\mathrm{CPA}}}{T_d}
\right).
\]

若车辆已经远离，设置 \(\rho=0\)。

### 6.4 EvidenceNet

共享评分：

\[
s_{ij}=G_{\phi_b}(\xi_i,\xi_j,e_{ij}),
\]

角色交换：

\[
\ell_{ij}
=
s_{ij}
-G_{\phi_b}(\xi_j,\xi_i,\mathcal S(e_{ij})).
\]

`swap_roles()` 至少处理：

- 相对位置反号；
- 相对速度反号；
- 航向差按反向定义重算；
- 到达时间差反号；
- 规则优先差反号。

网络输出：

\[
\widetilde b_{ij}
=b_{\max}\tanh(\ell_{ij}/T_b),
\]

\[
b_{ij}=\rho_{ij}c_{ij}\widetilde b_{ij}.
\]

注意：不同车辆在部分观测和噪声下可能形成不同的局部证据，因此不得声称执行期一定满足全局 \(b_{ij}=-b_{ji}\)。反对称结构只保证同一局部重构车辆对在角色交换时的模型结构一致性。

### 6.5 OpinionDynamics

\[
z_{ij}^{t}
=
z_{ij}^{t-1}
+\Delta t\,\eta_z
\left[
-\kappa_z z_{ij}^{t-1}
+\rho_{ij}^t\nu_z\tanh(\alpha_z z_{ij}^{t-1})
+b_{ij}^t
\right].
\]

分岔阈值：

\[
\rho_c=\frac{\kappa_z}{\nu_z\alpha_z}.
\]

必须满足：

\[
\nu_z\alpha_z>\kappa_z,
\]

否则在 \(\rho\le1\) 范围内不会发生中性态失稳。

离散衰减至少满足：

\[
0<\Delta t\,\eta_z\kappa_z<2.
\]

工程保护：

- `n_substeps >= 1`；
- 每个子步使用 `dt / n_substeps`；
- 每步之后 `clamp(-z_clip, z_clip)`；
- 发现 NaN/Inf 时将对应边回退至零并记录计数；
- 对角线每步强制清零。

### 6.6 OpinionResidual

\[
q_{ij}=\tanh(z_{ij}/z_0),
\]

\[
\bar\rho_{ij}
=
\frac{\rho_{ij}}
{\epsilon+\sum_k\rho_{ik}},
\]

\[
\Delta\mu_{i,\mathrm{op}}
=
c_{\mathrm{op}}
\sum_j\bar\rho_{ij}q_{ij}.
\]

由 \(|q|\le1\) 且 \(\sum_j\bar\rho_{ij}\le1\)，应满足：

\[
|\Delta\mu_{i,\mathrm{op}}|\le c_{\mathrm{op}}.
\]

最终均值：

```python
loc_final = loc_base.clone()
loc_final[..., 0] += opinion_residual
# loc_final[..., 1] unchanged
```

---

## 7. 推荐初始配置

`config_opinion.json` 建议使用嵌套配置：

```json
{
  "scenario_name": "road_traffic",
  "scenario_type": "CPM_mixed",
  "n_agents": 4,
  "dt": 0.05,
  "device": "cpu",

  "n_iters": 250,
  "frames_per_batch": 4096,
  "max_steps": 128,
  "num_epochs": 5,
  "minibatch_size": 512,
  "lr": 0.0002,
  "gamma": 0.99,
  "lmbda": 0.9,
  "clip_epsilon": 0.2,
  "entropy_eps": 0.0001,
  "max_grad_norm": 1.0,

  "is_prb": false,
  "is_using_opponent_modeling": false,
  "is_using_prioritized_marl": false,
  "use_topology_neighbor_selection": false,
  "topology_loss_weight": 0.0,

  "use_opinion_marl": true,
  "opinion_config": {
    "stage": "base",
    "n_candidates": 3,
    "chunk_length": 16,
    "chunks_per_minibatch": 16,

    "evidence_hidden_dim": 128,
    "evidence_num_layers": 2,
    "b_max": 0.5,
    "b_temperature": 1.0,

    "kappa": 1.0,
    "nu": 1.0,
    "alpha": 2.0,
    "eta": 1.0,
    "z0": 1.0,
    "z_clip": 2.0,
    "n_substeps": 1,

    "residual_scale_start": 0.0,
    "residual_scale_target": 0.1,
    "residual_warmup_fraction": 0.15,

    "lr_actor": 0.0002,
    "lr_evidence": 0.00002,
    "lr_critic": 0.0002,

    "neutral_loss_weight": 0.001,
    "magnitude_loss_weight": 0.0001,

    "ttc_horizon": 5.0,
    "safe_distance": 1.0,
    "urgency_time_scale": 2.0,
    "urgency_distance_temperature": 0.2,

    "include_z_in_critic": false,
    "log_pair_diagnostics": true
  }
}
```

说明：以上只是数值稳定的第一轮起点，不是最终论文参数。所有动力学参数调整必须记录分岔阈值和离散稳定性。

---

## 8. 详细实施步骤

## 步骤 0：建立可复现环境

### 修改

不修改算法代码。创建独立虚拟环境并安装 `requirements.txt`。

### 检查

```bash
python -c "import torch, torchrl, tensordict, vmas; print(torch.__version__)"
pytest -q
```

### 验收

- 所有核心依赖可以导入；
- 现有测试可运行；
- 能创建 road traffic env 并执行一个随机动作 rollout。

### 禁止

- 不为绕过依赖错误而修改算法代码；
- 不在系统/用户 Python 中覆盖安装未知版本。

---

## 步骤 1：固化原始基线

### 修改

新增纯 Base-MAPPO 配置，关闭所有 TSC/priority/opponent 路径。

### 输出

```text
outputs/baselines/base_mappo/
outputs/baselines/tsc/
```

### 验收

- Base-MAPPO 能训练至少 2～5 iteration；
- TSC 原入口仍能运行；
- 保存初始性能、碰撞率和配置副本；
- 不要求两者性能相同，只要求后续有可回归基线。

---

## 步骤 2：增加配置类与新入口

### 文件

- 新增 `utilities/opinion/config.py`；
- 新增 `main_training_opinion.py`；
- 新增 `main_testing_opinion.py`；
- 新增 `config_opinion.json`；
- `Parameters` 只增加：

```python
use_opinion_marl: bool = False
opinion_config: dict | None = None
```

### `OpinionConfig` 职责

- 提供默认值；
- 验证 `chunk_length > 0`；
- 验证 `n_candidates <= n_agents - 1`；
- 验证 `0 <= residual_scale`；
- 计算并记录 `rho_c`；
- 验证 `nu * alpha > kappa`；
- 验证离散衰减条件；
- 验证旧 TSC 路径全部关闭。

### 验收

- 旧 `config.json` 加载行为不变；
- 新 `config_opinion.json` 能加载；
- 配置非法时在训练开始前明确报错。

---

## 步骤 3：实现独立数学模块

### 文件

- `evidence_net.py`；
- `dynamics.py`；
- `residual.py`。

### 顺序

1. 实现 `swap_roles()`；
2. 实现共享相对评分和 `raw_b/b`；
3. 实现无状态的 `OpinionDynamics.forward()`；
4. 实现候选边到稠密 \([E,N,N]\) 的 scatter；
5. 实现稠密状态到候选边的 gather；
6. 实现 residual 聚合。

### 单元测试

- `raw_b` 幅值不超过 `b_max`；
- `b` 不超过 `rho * confidence * b_max`；
- 本地角色交换时证据反号；
- mask/padding 输出严格为零；
- 无冲突时 `z` 衰减；
- 高紧迫度时微小正负扰动进入不同分支；
- 强反向证据可以翻转意见；
- residual 有界；
- 对角线永远为零；
- 动力学参数无梯度。

### 验收

此步骤不接环境、不接 PPO，只完成纯张量模块及测试。

---

## 步骤 4：增加 ConflictGraph 数据接口

### 文件

- 新增 `utilities/opinion/conflict_graph.py`；
- 修改 `scenarios/road_traffic.py`。

### 实现原则

- `ConflictGraph` 是固定几何模块，不是 TSC topology；
- 场景只输出当前物理量、mask 和固定紧迫度；
- 场景不保存、不更新 `z`；
- 每个 ego 使用稳定全局 neighbor ID；
- 当前 N=4 时可为全部其他车辆输出候选。

### 推荐实现

在场景完成当前时刻位置、速度、距离缓存后，构造所有车辆对：

```python
pair_data = build_opinion_pair_data(
    positions=current_positions,
    velocities=current_velocities,
    headings=current_headings,
    visibility_mask=...,
    agent_sizes=...,
    local_map_features=...,
)
```

将对应 ego 行写入 `info()`。

### Reset 接口

新增 `agent_reset_mask`。如果当前环境不能直接在 `info()` 提供该事件，则在 `reset_world_at()` 设置一次性标志，并在下一次读取后清除。

### 测试

- 迎面接近；
- 平行同速；
- 背离；
- 交叉接近；
- 静止邻车；
- 不可见邻车；
- 单 agent reset。

### 验收

- 环境 rollout 中所有 Opinion info key 形状固定；
- 没有 NaN/Inf；
- 不使用真实未来标签；
- 不受 `use_topology_neighbor_selection` 开关影响。

---

## 步骤 5：实现 OpinionAugmentedPolicy

### 文件

- 新增 `utilities/opinion/policy.py`。

### 模块

```text
BaseGaussianActor
OpinionEvidenceNet
OpinionDynamics
OpinionResidual
OpinionAugmentedPolicyCore
ProbabilisticActor(TanhNormal wrapper)
```

### 关键顺序

```python
base_loc, scale = base_actor(observation)
raw_b, b = evidence_net(pair_features, rho, confidence, mask)
z_next = dynamics(z_prev, ids, rho, b, mask, reset_mask)
q = gather_and_squash(z_next, ids)
delta = residual(q, rho, mask)
loc = base_loc.clone()
loc[..., 0] = loc[..., 0] + delta
```

### Stage base

在 `stage="base"` 时：

- `delta=0`；
- EvidenceNet 不计算或停止梯度；
- 仍使用同一 Policy 外壳，确保后续 checkpoint 结构兼容。

### 测试

- `residual_scale=0` 时结果等于 BaseActor；
- `scale` 不受 Opinion 分支影响；
- 正 `q` 增加速度 loc，负 `q` 降低；
- 转向 loc 不变；
- 计算图能从 loc 回到 EvidenceNet；
- BaseActor 不读取 `z`。

### 验收

给定伪造 TensorDict，完整 policy 可以采样 action 并返回 log-prob 和 `z_next`。

---

## 步骤 6：实现 Opinion Collector

### 文件

- 新增 `utilities/opinion/collector.py`。

### 状态生命周期

初始化：

```python
z_prev = torch.zeros(num_envs, n_agents, n_agents)
```

每步：

```python
td["agents", "opinion_z"] = z_prev
policy(td)  # produces action and z_next
transition, next_td = env.step_and_maybe_reset(td)
z_next = td["agents", "opinion_z_next"]
z_next = apply_agent_and_env_resets(z_next, transition, next_td)
next_td["agents", "opinion_z"] = z_next
z_prev = z_next
```

### 注意

- collection 本身使用 `torch.no_grad()`；
- rollout 保存 action 生成时实际使用的 `z_prev`；
- 不允许在 `observation()`、`info()` 或 `_observe_other_agents()` 更新意见；
- done 和 reset 的时序必须通过测试确定，不能凭猜测清空前后状态。

### 测试

- 单环境多步；
- 多并行环境；
- 不同环境在不同时间 done；
- partial reset；
- full reset；
- 车辆 ID/候选槽位切换。

### 验收

- 每个物理步只更新一次；
- 并行环境无状态串扰；
- reset 后相应状态为零；
- rollout 时间维完整。

---

## 步骤 7：实现 Sequence Buffer

### 文件

- 新增 `utilities/opinion/sequence_buffer.py`。

### 禁止

不要执行当前 PPO 的全局 `reshape(-1)` 后单步随机采样。

### 实现

1. GAE 在完整 \([E,T,N]\) rollout 上计算；
2. 沿每个环境时间轴按 `chunk_length=L` 切片；
3. 每个 chunk 保存第一个时间步更新之前的 `z_init`；
4. chunk 内保留 done/reset mask；
5. minibatch 采样单位是 chunk，不是 frame；
6. minibatch 可在 chunk 维打乱，但 chunk 内时间顺序不能打乱。

### 边界

- 若 episode 在 chunk 内结束，后续状态按 reset mask 清零；
- 可使用有效时间 mask 处理不足 L 的尾段；
- 第一版也可要求 `T % L == 0` 并在配置层验证，降低复杂度。

### 测试

- chunk 起点 `z_init` 正确；
- chunk 内顺序不变；
- env 之间不混合；
- terminal mask 正确；
- 拼接 chunks 可恢复原 rollout。

---

## 步骤 8：实现序列重算和 PPO Loss

### 文件

- 新增 `utilities/opinion/ppo_loss.py`。

### 重算伪代码

```python
z = chunk.z_init.detach()
new_log_probs = []
entropies = []

for t in range(chunk_length):
    z = apply_reset_before_step(z, chunk.reset_mask[:, t])
    outputs = policy_core(
        observation=chunk.obs[:, t],
        pair_features=chunk.pair_features[:, t],
        neighbor_ids=chunk.neighbor_ids[:, t],
        rho=chunk.rho[:, t],
        confidence=chunk.confidence[:, t],
        valid_mask=chunk.valid_mask[:, t],
        z_prev=z,
    )
    dist = make_tanh_normal(outputs.loc, outputs.scale)
    new_log_probs.append(dist.log_prob(chunk.action[:, t]))
    entropies.append(dist.entropy())
    z = outputs.z_next
```

### PPO

\[
r_t=\exp(\log\pi_{\mathrm{new}}-\log\pi_{\mathrm{old}}),
\]

\[
L_{\mathrm{clip}}
=
-\mathbb E
\left[
\min(r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)
\right].
\]

Actor 总损失：

\[
L_\pi
=
L_{\mathrm{clip}}
-\lambda_HH
+\lambda_{\mathrm{neutral}}L_{\mathrm{neutral}}
+\lambda_{\mathrm{mag}}L_{\mathrm{mag}}.
\]

辅助损失建议作用于未门控证据：

\[
L_{\mathrm{neutral}}
=
\mathbb E[(1-\rho)\widetilde b^2],
\]

\[
L_{\mathrm{mag}}
=
\mathbb E[m_{ij}\widetilde b^2].
\]

### Optimizer

使用独立参数组：

```text
base_actor_optimizer
evidence_optimizer
critic_optimizer
```

动力学参数不进入 optimizer。

### 必做一致性测试

参数完全不更新时：

\[
\max|
\log\pi_{\mathrm{recomputed}}
-\log\pi_{\mathrm{rollout}}
|<10^{-5}\sim10^{-4}.
\]

如果失败，停止实现 PPO，先检查：

- `z_init` 是否正确；
- reset 时序；
- action 是否与 log-prob 对应；
- TanhNormal 参数和上下界；
- pair candidate 顺序；
- policy 是否在 collection/recompute 使用相同输入；
- 是否存在随机 dropout/BatchNorm。

### 梯度测试

- Actor/PPO loss 对 EvidenceNet 梯度非零；
- Critic loss 对 EvidenceNet 无梯度；
- OpinionDynamics 参数无梯度；
- chunk 起点以前无梯度；
- 片段内早期 `b` 能通过 `z` 影响后期 loss。

---

## 步骤 9：实现三阶段 Trainer

### 文件

- 新增 `utilities/opinion/trainer.py`。

### Stage 1：Base

```text
stage = base
residual_scale = 0
EvidenceNet frozen
BaseActor trainable
Critic trainable
```

目标：获得基础驾驶、车道保持和一般避碰能力。

### Stage 2：Evidence

```text
stage = evidence
load Base checkpoint
BaseActor frozen
EvidenceNet trainable
Critic trainable
OpinionDynamics fixed
residual_scale warm-up: 0 → target
```

目标：证明长期任务优势能训练局部证据网络。

### Stage 3：Joint

```text
stage = joint
BaseActor trainable with controlled lr
EvidenceNet trainable with smaller lr
Critic trainable
OpinionDynamics fixed
```

建议初始学习率：

```text
actor:    2e-4
evidence: 2e-5
critic:   2e-4
```

### BaseActor 绕过问题

“BaseActor 不读取 z”不能严格保证其不从物理观测复制或抵消协调行为。

因此必须通过：

- Stage 2 冻结 BaseActor；
- Stage 3 控制学习率；
- `q=0` 干预；
- shuffle `q`；
- remove residual；
- gradient/输出日志；

来验证意见通道是否真正具有因果作用。

---

## 步骤 10：Checkpoint 与恢复训练

### 文件

- 新增 `utilities/opinion/checkpoint.py`。

### 保存

```text
base_actor_state_dict
evidence_net_state_dict
critic_state_dict
actor_optimizer_state
evidence_optimizer_state
critic_optimizer_state
training_stage
iteration
opinion_config
environment_config
normalization_config
metrics_summary
code/checkpoint schema version
```

### 不保存

默认不保存 episode 中间的运行时 `z`。评估和新 episode 从零意见开始。

如果未来需要真正无缝恢复 collector 中间状态，再单独增加 runtime checkpoint，不与第一版混合。

### 文件命名

```text
<run_name>_base_actor.pt
<run_name>_evidence.pt
<run_name>_critic.pt
<run_name>_training_state.pt
<run_name>_config.json
```

不要继续使用 `_topology.pth`、`_priority_policy.pth` 等 TSC 名称。

---

## 步骤 11：测试与诊断入口

### 文件

- 新增 `main_testing_opinion.py`；
- 新增 `utilities/opinion/diagnostics.py`。

### 每条有效边可记录

```text
step
time_sec
env_id
ego_id
neighbor_id
valid
rho
confidence
raw_b
b
z
q
opinion_residual
base_loc_speed
final_loc_speed
executed_speed
collision
done
reset
```

### 训练指标

```text
opinion/mean_abs_raw_b
opinion/mean_abs_b
opinion/b_saturation_rate
opinion/mean_abs_z
opinion/z_clip_rate
opinion/neutral_edge_ratio
opinion/sign_switch_rate
opinion/mean_residual
opinion/max_residual
opinion/effective_conflict_edges
grad/evidence_norm
grad/base_actor_norm
debug/recompute_log_prob_max_error
debug/opinion_nan_count
```

### 注意

这些日志可以在集中评估脚本中读取所有车辆状态，但论文必须明确：集中日志不等于执行期通信。

---

## 9. Rollout 与训练伪代码

### 9.1 执行/采集

```python
z = zeros(num_envs, n_agents, n_agents)

for t in range(rollout_length):
    td[("agents", "opinion_z")] = z

    with torch.no_grad():
        td = policy(td)

    transition, next_td = env.step_and_maybe_reset(td)

    z_next = td[("agents", "opinion_z_next")]
    z_next = reset_full_env_rows(z_next, transition)
    z_next = reset_agent_rows_and_columns(z_next, next_td)

    next_td[("agents", "opinion_z")] = z_next
    save(transition)
    z = z_next
```

### 9.2 更新

```python
rollout = collect()
compute_gae_on_full_rollout(rollout)
chunks = split_into_contiguous_chunks(rollout)

for epoch in range(num_epochs):
    for chunk_batch in sample_chunk_batches(chunks):
        replay = recompute_opinion_and_policy(chunk_batch)

        actor_loss = clipped_ppo_loss(replay, chunk_batch)
        auxiliary_loss = neutral_loss(replay) + magnitude_loss(replay)
        critic_loss = value_loss(chunk_batch)

        actor_optimizer.zero_grad()
        evidence_optimizer.zero_grad()
        (actor_loss + auxiliary_loss).backward()
        clip_actor_and_evidence_gradients()
        step_enabled_actor_optimizers_for_current_stage()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_critic_gradients()
        critic_optimizer.step()
```

---

## 10. 必须通过的测试矩阵

### 10.1 数学单元测试

| 测试 | 预期结果 |
|---|---|
| `rho=0,b=0,z!=0` | `|z|` 逐步下降 |
| `rho<rho_c,b=0` | 小扰动回到零附近 |
| `rho>rho_c,b=0` | 正负微扰进入不同非零分支 |
| 正 `b` | 推动 `z` 增大 |
| 负 `b` | 推动 `z` 减小 |
| 强反向 `b` | 可克服原意见并翻转 |
| 无效 mask | `b=0`，状态仅衰减 |
| 对角线 | 永远为零 |
| 角色交换 | 本地未门控证据反号 |
| residual | 绝对值不超过 `c_op` |

### 10.2 生命周期测试

| 场景 | 预期结果 |
|---|---|
| 环境 done | 对应环境全部 `z=0` |
| agent k reset | 第 k 行、第 k 列清零 |
| neighbor 槽位换人 | 状态仍跟随全局 agent ID |
| neighbor 消失 | 对应状态衰减，不转移给新槽位 |
| 多环境异步结束 | 环境之间状态不串扰 |

### 10.3 PPO 测试

| 测试 | 预期结果 |
|---|---|
| 未更新参数重算 log-prob | 与 rollout 值误差小于容差 |
| Actor loss backward | EvidenceNet 有非零梯度 |
| Critic loss backward | EvidenceNet 无梯度 |
| dynamics parameters | 不在 optimizer，无梯度 |
| `z_init.detach()` | chunk 之前无梯度 |
| 片段内时序 | 早期证据能影响后期状态和 loss |

### 10.4 集成测试

- 1 env × 2 agents × 16 steps；
- 4 env × 4 agents × 128 steps；
- CPU smoke training 2 iterations；
- 无 NaN/Inf；
- 能保存和重新加载 checkpoint；
- 加载后相同输入产生相同 policy distribution；
- 原 TSC 测试仍通过。

---

## 11. 分阶段验收门

### Gate A：数学模块完成

- Evidence/Dynamics/Residual 单测全部通过；
- 未接环境；
- 未接 PPO。

### Gate B：环境接口完成

- rollout 中 Opinion info key 齐全；
- neighbor ID 稳定；
- reset mask 正确；
- 无未来信息泄漏。

### Gate C：单步 Policy 完成

- 可以采样 action；
- log-prob 正确；
- residual 只影响速度 loc；
- `z_next` 有限。

### Gate D：Collector 完成

- 状态只更新一次；
- reset 正确；
- rollout 连续。

### Gate E：Sequence PPO 完成

- log-prob 重算一致；
- 梯度隔离正确；
- chunk BPTT 正确。

### Gate F：训练完成

- Base stage 达到可用性能；
- Evidence stage 中 EvidenceNet 梯度和输出非零；
- Joint stage 无崩溃；
- 意见模块干预实验能改变策略行为。

只有通过前一个 Gate，才能进入下一个 Gate。

---

## 12. 消融实验计划

新方法必须证明收益来自非线性意见动力学，而不只是增加了参数或历史状态。

| ID | 方法 | 设置 |
|---|---|---|
| A | Base-MAPPO | `c_op=0` |
| B | Direct Evidence | `q=tanh(raw_b/b0)`，无动态状态 |
| C | EMA Evidence | 使用普通指数平滑 |
| D | Linear Opinion | `nu=0`，只有遗忘 + evidence |
| E | GRU-MAPPO | 黑盒循环记忆 |
| F | Handcrafted Evidence + NOD | 固定几何 evidence，不学习 B |
| G | Full Opinion-MARL | 完整方法 |
| H | TSC | 外部协调基线 |

### 扰动设置

- 原始干净观测；
- 位置/速度噪声；
- 1～3 步观测延迟；
- 短时遮挡；
- 对称到达；
- 邻车突然加速/减速；
- 初始错误倾向；
- 不同车辆数量和密度。

### 任务指标

- agent-agent collision rate；
- agent-map collision rate；
- average speed；
- smoothness；
- episode reward；
- success rate；
- deadlock duration。

### 意见指标

- evidence saturation；
- opinion sign switch rate；
- commitment formation time；
- wrong-opinion reversal time；
- simultaneous-go rate；
- simultaneous-yield rate；
- no-conflict residual；
- `q=0` 干预性能下降。

---

## 13. 常见错误与禁止事项

### 13.1 把 `z` 拼进 observation 后继续用原 MLP

这无法保证意见具有显式作用，也不能训练 EvidenceNet 的时序路径。禁止作为最终实现。

### 13.2 在场景 observation 函数中更新 `z`

观察函数可能被多次调用，会导致一个物理步更新多次。禁止。

### 13.3 用 Top-K 槽位保存状态

槽位会换人，造成意见错误转移。必须按全局 agent ID 保存。

### 13.4 执行动作后再添加 residual

这会导致保存的 log-prob 不对应真实动作，PPO 数学失效。必须改 distribution `loc`。

### 13.5 全部展平后随机单步训练

这会切断 BPTT。必须使用连续 chunk。

### 13.6 让 Critic loss 更新 EvidenceNet

会让 evidence 追逐 value-fitting 误差，而不是策略优势。必须分离计算图和 optimizer。

### 13.7 同时学习动力学参数和 evidence

第一版禁止。否则策略、证据和动力学同时漂移，无法解释收益来源。

### 13.8 使用真实未来轨迹计算执行期冲突

训练标签可使用未来信息，但本方案第一版不需要标签。执行数据流禁止未来泄漏。

### 13.9 宣称安全保证

本方案没有 VO/CBF/安全投影，因此只能报告经验安全指标，不能声称形式化无碰撞保证。

### 13.10 宣称全局意见一致

每辆车使用局部观测维护自己的意见。部分观测下 `z_ij` 和 `z_ji` 不一定严格互补，不能声称存在通信共享的全局意见场。

---

## 14. Checkpoint、回滚与兼容策略

### 14.1 保持 TSC 基线不变

- 原 `main_training.py` 和 `main_testing.py` 保持可用；
- 新方法使用独立入口；
- 新配置不覆盖 `config.json`；
- 新 checkpoint 不覆盖原 outputs。

### 14.2 每个里程碑独立提交

建议提交顺序：

```text
1. environment/baseline
2. opinion config and module skeleton
3. pure mathematical modules and tests
4. scenario conflict interface
5. opinion policy
6. opinion collector
7. sequence buffer
8. recurrent PPO loss
9. staged trainer/checkpoints
10. evaluation/ablations
```

未得到用户授权时，不自动 commit 或 push；这里只规定推荐的逻辑拆分。

### 14.3 兼容性

- `use_opinion_marl=false` 时旧配置行为必须不变；
- 新配置加载失败时要在训练前报错；
- checkpoint 中保存 schema version；
- 不使用 `strict=False` 静默吞掉关键权重缺失，除非显式记录迁移结果。

---

## 15. 完成定义

同时满足以下条件才算第一版实现完成：

- [ ] 依赖环境可复现；
- [ ] 原 TSC 入口仍可运行；
- [ ] 纯 Base-MAPPO 基线已固化；
- [ ] 新方法不调用 TSC priority/leader/Stackelberg 路径；
- [ ] EvidenceNet 输出有界；
- [ ] EvidenceNet 不读取 `z/q`；
- [ ] OpinionDynamics 参数固定；
- [ ] `z` 按全局 agent ID 保存；
- [ ] `z` 每个物理步只更新一次；
- [ ] partial/full reset 正确；
- [ ] residual 只修改速度 loc；
- [ ] residual 有界；
- [ ] rollout 保留连续时间；
- [ ] chunk 重放 log-prob 一致；
- [ ] Actor loss 能训练 EvidenceNet；
- [ ] Critic loss不能训练 EvidenceNet；
- [ ] 训练无 NaN/Inf；
- [ ] checkpoint 可保存/加载；
- [ ] Opinion 诊断日志完整；
- [ ] Direct/EMA/Linear/GRU/Full 消融可配置；
- [ ] TSC 作为外部基线完成比较。

---

## 16. 里程碑状态（后续 Session 必须维护）

> 状态值使用：`未开始`、`进行中`、`已完成`、`阻塞`。

| 里程碑 | 状态 | 验证记录 |
|---|---|---|
| M0 依赖环境修复 | 已完成 | 见下方“M0 验证记录”；隔离环境的核心依赖导入、TanhNormal 与 3-step road_traffic smoke 均通过 |
| M1 Base-MAPPO/TSC 基线固化 | 未开始 |  |
| M2 Opinion 配置与文件骨架 | 未开始 |  |
| M3 Evidence/Dynamics/Residual 单测 | 未开始 |  |
| M4 ConflictGraph 环境接口 | 未开始 |  |
| M5 OpinionAugmentedPolicy | 未开始 |  |
| M6 Opinion Collector 与 reset | 未开始 |  |
| M7 Sequence Buffer | 未开始 |  |
| M8 PPO 重算与梯度隔离 | 未开始 |  |
| M9 三阶段 Trainer/Checkpoint | 未开始 |  |
| M10 测试入口与诊断日志 | 未开始 |  |
| M11 消融与完整实验 | 未开始 |  |

### M0 验证记录

- 修改文件：新增 `scripts/check_runtime_environment.py`、`tests/test_runtime_environment_check.py`；更新本指南的里程碑状态。检查脚本以 `requirements.txt` 为唯一版本真源，读取并规范化核心包名后校验四个 `==` pin。算法、场景和依赖锁定文件均未修改。
- 虚拟环境：`.python-version` 当前保存的是既有 pyenv 环境名 `vmas`，而不是 Python 版本号；本次使用兼容的 Python 3.9.13 创建仓库内 `.venv`。可移植安装命令如下：

  ```bash
  python3.9 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m pip install pytest==7.1.2
  ```

  `pytest` 单独安装是因为当前 `requirements.txt` 未声明测试运行器；没有在系统或用户 Python 中安装/覆盖包。
- 实际版本：Python 3.9.13、torch 2.1.0、torchrl 0.2.1、tensordict 0.2.1、vmas 1.4.1。
- TDD 记录：首轮先新增运行时检查测试并执行 `.venv/bin/python -m pytest -q tests/test_runtime_environment_check.py`，因 `scripts.check_runtime_environment` 尚不存在而按预期 RED；添加最小实现后 GREEN 为 `3 passed`。审查修正轮先加入版本漂移/缺 pin、真实 2-step 环境和 main 错误边界测试，旧实现按预期为 `2 failed, 6 passed`（尚无锁定校验接口）；补充最小实现后同一命令为 `8 passed`。
- 完整验证命令：

  ```bash
  .venv/bin/python -m pytest -q
  .venv/bin/python scripts/check_runtime_environment.py --steps 3
  .venv/bin/python scripts/check_runtime_environment.py --steps 1
  .venv/bin/python -m pip check
  .venv/bin/python -m compileall -q scripts/check_runtime_environment.py tests/test_runtime_environment_check.py
  ```

- 实际结果：完整测试 `10 passed`，其中包括不 mock 核心环境的 2-step 集成测试；运行时检查成功导入四个核心依赖，并确认其版本精确匹配 `requirements.txt`，`TanhNormal.rsample()`/`log_prob()` 均为有限值，`CPM_mixed` road traffic 环境完成 reset 和 3 步随机动作 rollout，检查到的浮点张量均为有限值；非法的 `--steps 1` 明确报告范围错误并返回非零状态，导入异常同样由 main 统一转换为 `[FAIL]` 和状态码 1；`pip check` 未发现依赖冲突，目标文件 compileall 成功。
- 尚存问题：Gym 0.26.2 会输出其上游弃用提示，但不影响本次 smoke；`intersection_1` 的既有地图路径常量为 `intersection_1.osm`，从仓库根目录构造时找不到文件，因此 M0 使用项目训练配置中的 `CPM_mixed`，未越界修改场景代码；测试运行器尚未进入锁定依赖，后续环境重建须执行上述单独安装命令。
- 下一步入口：执行 M1，分别固化纯 Base-MAPPO 与现有 TSC 的可回归训练基线；开始前先在 `.venv` 重新执行上述测试和 runtime smoke。

每完成一个里程碑，应在“验证记录”中写入：

- 修改文件；
- 执行的测试命令；
- 测试结果；
- 仍存在的问题；
- 下一步入口。

---

## 17. 新 Session 交接模板

完成阶段性工作后，可把以下内容附在新 session 提示中：

```text
项目继续实施 Opinion Dynamics + MARL。

请先完整阅读：
1. OPINION_MARL_IMPLEMENTATION_GUIDE.md
2. opinion_dynamics_marl_technical_route.md

理论定位：新方法不是 TSC 扩展。TSC 只作为代码载体和外部基线。

当前已完成里程碑：<填写 M 编号>
当前验证命令：<填写>
当前验证结果：<填写>
当前阻塞：<填写，没有则写无>

请先检查 git status 和指南第 16 节，然后只继续下一个未完成里程碑。
实现时保留原 TSC 入口，不覆盖用户文件，不在 observation/info 中更新 z，
不按 Top-K 槽位保存 z，不破坏序列 PPO 的时间维。
```

---

## 18. 第一轮实施建议

第一次正式编码建议只完成 M0～M3：

1. 修复并验证依赖环境；
2. 固化 Base-MAPPO/TSC 基线；
3. 建立 `utilities/opinion/` 和独立配置；
4. 实现纯数学 Evidence/Dynamics/Residual；
5. 完成所有纯张量单元测试。

在 M3 完成前，不应修改 Collector 或 PPO。这样可以先确认非线性动力学、边界、分岔条件、mask 和动作残差在数学与数值上正确，再进入高风险的序列训练改造。
