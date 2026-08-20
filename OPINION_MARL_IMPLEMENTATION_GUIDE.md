# Opinion Dynamics + MARL：实施方案与跨 Session 交接指南

> 最后核对日期：2026-08-20  
> 当前仓库：`/Users/zhangxiaotong/Code/sigmarl-nod`  
> 当前阶段：R0/M0、R1/M1、M2、M3、M4、M5 已完成；下一步执行 M6 Stateful Collector  
> 理论真源：`opinion_dynamics_marl_technical_route.md`  
> 用途：让新的开发 Session 不依赖历史聊天，也能准确继续 Opinion Dynamics + MARL 的实现

---

## 0. 新 Session 先读这里

新的 Session 开始后，必须先完整阅读：

1. `OPINION_MARL_IMPLEMENTATION_GUIDE.md`；
2. `opinion_dynamics_marl_technical_route.md`；
3. 与当前里程碑直接相关的源码和测试。

本项目的工作规则：

- 只在 `/Users/zhangxiaotong/Code/sigmarl-nod` 中工作；
- 直接修改原目录中的代码和文档；
- 使用用户已经建立的 `.venv`；
- 不创建或管理 Git 分支、worktree、commit、merge、push、pull；
- 不修改旧目录 `/Users/zhangxiaotong/Code/sigmarl-traffic`；
- 旧目录只能在确有必要时作为只读实现参考；
- 每个里程碑必须先测试、再实现、再运行完整回归；
- 不得因为训练入口可以运行，就宣称 Opinion 方法已经实现。

如果本文件与理论路线冲突：

- 方法定义、数学含义以 `opinion_dynamics_marl_technical_route.md` 为准；
- 文件组织、接口、测试和实施顺序以本文件为准；
- 用户最新指令始终拥有最高优先级。

### 0.1 当前最重要结论

`sigmarl-nod` 在开始修改前已确认与原始工程一致；当前已在此基础上完成
R0/M0、R1/M1、M2、M3、M4 和 M5：

- 原始 `config.json`、`requirements.txt`、`main_training.py`、`main_testing.py`
  仍保持兼容；
- `utilities/mappo_cavs.py` 已加入 Base/TSC 门控；
- `utilities/helper_training.py` 已加入默认关闭的 Opinion 配置字段；
- `scenarios/road_traffic.py` 仅在 `use_opinion_marl=true` 时输出 M4
  current-physics ConflictGraph 信息；
- 理论路线文件已经存在且内容完整；
- M0 runtime checker、M1 基线、M2 配置入口、M3 数学模块、M4 环境接口和
  M5 单步 Policy 均已通过验证；
- 本文件是新仓库重新建立的工程实施真源。

当前环境已经验证：

```text
Python       3.9.13
torch        2.1.0
torchrl      0.2.1
tensordict   0.2.1
vmas         1.4.1
现有测试      201 passed
pip check    无依赖冲突
```

当前恢复顺序为：

```text
R0：恢复并验证 M0 运行时检查（已完成）
→ R1：恢复并验证 M1 Base-MAPPO/TSC 基线（已完成）
→ M2：Opinion 配置与独立入口（已完成）
→ M3：Evidence、Dynamics、Residual 纯数学模块（已完成）
→ M4：ConflictGraph 当前物理量接口（已完成）
→ M5：OpinionAugmentedPolicy（已完成）
→ M6：Stateful Opinion Collector（当前下一步）
```

---

## 1. 方法定位：必须与 TSC 理论分离

新方法是独立的 **Opinion Dynamics + MARL**，不是在 TSC 理论上增加一个意见模块。

核心因果链为：

```text
局部物理观测
→ 冲突车辆对特征 χ_ij
→ EvidenceNet 输出瞬时意见证据 b_ij
→ 固定非线性 OpinionDynamics 更新 z_ij
→ q_ij = tanh(z_ij / z0)
→ 有界 Opinion residual
→ 修改基础 Actor 的速度均值 loc[..., 0]
→ TanhNormal 采样动作
→ MAPPO 长期回报训练 Actor、Critic 和 EvidenceNet
```

其中：

- `b_ij` 是当前时刻的瞬时证据，不承担记忆；
- `z_ij` 是带时间连续性的意见状态，承担积累、衰减、自强化和翻转；
- OpinionDynamics 第一版参数固定，不加入 optimizer；
- EvidenceNet 通过 PPO Actor loss 接收长期任务回报的梯度；
- Critic loss 不得更新 EvidenceNet；
- 执行阶段不在线更新网络参数。

### 1.1 TSC 可以被如何使用

TSC 只允许作为：

- VMAS 道路环境和现有 MAPPO 工程载体；
- TensorDict、collector、checkpoint 等工程实现的参考；
- 实验中的外部协调基线。

Opinion 方法禁止依赖：

- priority labels；
- leader set；
- Stackelberg leader/follower 结构；
- total order 或 topology consistency loss；
- TSC topology learner；
- TSC action predictor；
- opponent-modeling critic 输入；
- 真实未来轨迹标签。

### 1.2 Base-MAPPO、TSC、Opinion 三条路径

| 路径 | 用途 | 允许组件 |
|---|---|---|
| Base-MAPPO | 纯 MARL 对照与 Opinion 的基础策略 | 共享 Actor、集中 Critic、MAPPO |
| TSC | 外部协调基线 | 原 topology、action predictor、opponent 路径 |
| Opinion-MARL | 新方法 | Base-MAPPO + Evidence + 固定 Dynamics + residual |

三条路径必须使用不同配置和输出目录，不能通过同一个含糊开关混在一起。

---

## 2. 新方法的状态、张量和因果接口

### 2.1 有向意见状态

对车辆 `i` 和候选冲突车辆 `j`：

```text
z_ij > 0：i 倾向于相对 j 继续通行
z_ij < 0：i 倾向于相对 j 让行
z_ij ≈ 0：尚未形成明确意见
```

`z_ij` 与 `z_ji` 是两个有向状态，不要求互为相反数。

### 2.2 状态存储

内部状态必须按全局 agent ID 保存：

```text
z_dense: [num_envs, n_agents, n_agents]
```

约束：

- 对角线始终为零；
- 不按 Top-K 邻居槽位保存状态；
- neighbor 槽位换人时，旧状态不能转移给新车辆；
- agent reset 时清零对应行和列；
- environment reset 时清零该环境全部状态；
- 多环境异步 reset 不能串扰。

候选边输入建议形状：

```text
pair_features:       [E, N, K, F_pair]
neighbor_ids:        [E, N, K]
pair_mask:           [E, N, K]
urgency:             [E, N, K]
confidence:          [E, N, K]
agent_reset_mask:    [E, N]
environment_done:    [E]
```

### 2.3 每个物理步的固定顺序

```text
1. 读取当前 observation 和 ConflictGraph 信息
2. EvidenceNet 计算 raw_b
3. 通过 urgency/confidence/mask 得到 b
4. 从 z_dense gather 当前候选边状态
5. OpinionDynamics 更新 z_next
6. scatter 回 z_dense
7. q = tanh(z_next / z0)
8. OpinionResidual 聚合成速度残差
9. 修改 Actor 的 loc[..., 0]
10. 使用最终 loc 和原 scale 构造 TanhNormal
11. 采样 action 并保存 log_prob
```

意见状态每个物理步只能更新一次。禁止在 `observation()`、`info()` 或其他可能被重复调用的环境回调中更新 `z`。

### 2.4 动作耦合边界

第一版只修改速度通道的分布均值：

```python
loc_final = loc_base.clone()
loc_final[..., 0] = loc_final[..., 0] + opinion_residual
# loc_final[..., 1] 保持不变
# scale 保持 Base Actor 输出
```

必须先修改 `loc`，再构造分布和采样。禁止在动作采样后直接对 action 加 residual，否则 rollout log-prob 与 PPO 重算不一致。

---

## 3. 数学模块的第一版合同

### 3.1 EvidenceNet

EvidenceNet 使用共享的车辆对评分结构：

```text
s_i = f_local(role=i, pair context)
s_j = f_local(role=j, pair context)
raw_b_ij = b_max * tanh((s_i - s_j) / b_temperature)
b_ij = raw_b_ij * urgency_ij * confidence_ij * mask_ij
```

必须满足：

- `|raw_b| <= b_max`；
- `|b| <= urgency * confidence * b_max`；
- mask/padding 边严格输出零；
- 本地角色交换时，未门控证据近似反号；
- EvidenceNet 不读取 `z` 或 `q`；
- 不使用真实未来轨迹和 TSC 标签。

### 3.2 固定 OpinionDynamics

第一版采用固定参数的非线性离散动力学。具体方程以技术路线为准，工程上必须暴露并验证：

```text
kappa
nu
alpha
eta
dt
z_clip
n_substeps
```

关键阈值：

```text
rho_c = kappa / (nu * alpha)
```

配置必须满足：

```text
nu * alpha > kappa
0 < dt * eta * kappa < 2
```

动力学参数：

- 不是 `nn.Parameter`；
- 不出现在 optimizer；
- backward 后无梯度；
- 每次更新后对 `z` 做有限范围裁剪；
- 输入无效时只允许状态自然衰减，不得产生新证据。

### 3.3 OpinionResidual

建议：

```text
q_ij = tanh(z_ij / z0)
delta_i = aggregate(q_ij, urgency_ij, mask_ij)
opinion_residual = residual_scale * bounded(delta_i)
```

约束：

- residual 有明确绝对上界；
- 没有有效冲突边时严格为零；
- 只影响速度 loc；
- warm-up 期间 residual scale 从 0 平滑增加；
- residual 不负责安全保证，安全仍由任务回报和环境约束共同塑造。

---

## 4. 推荐文件架构

最终目标结构：

```text
sigmarl-nod/
├── config.json
├── config_opinion.json
├── configs/
│   └── baselines/
│       ├── base_mappo.json
│       └── tsc.json
├── main_training.py
├── main_testing.py
├── main_training_baseline.py
├── main_training_opinion.py
├── main_testing_opinion.py
├── scripts/
│   └── check_runtime_environment.py
├── utilities/
│   ├── baseline_config.py
│   ├── mappo_cavs.py
│   └── opinion/
│       ├── __init__.py
│       ├── config.py
│       ├── evidence_net.py
│       ├── dynamics.py
│       ├── residual.py
│       ├── conflict_graph.py
│       ├── policy.py
│       ├── collector.py
│       ├── sequence_buffer.py
│       ├── ppo_loss.py
│       ├── trainer.py
│       ├── checkpoint.py
│       └── diagnostics.py
└── tests/
    ├── test_runtime_environment_check.py
    ├── test_baseline_config.py
    ├── test_mappo_baseline_gating.py
    ├── test_main_training_baseline.py
    └── opinion/
        ├── test_opinion_config.py
        ├── test_evidence_net.py
        ├── test_dynamics.py
        ├── test_residual.py
        ├── test_conflict_graph.py
        ├── test_policy.py
        ├── test_collector.py
        ├── test_sequence_buffer.py
        ├── test_ppo_loss.py
        └── test_checkpoint.py
```

模块职责必须单一。不要把所有 Opinion 逻辑塞进 `mappo_cavs.py` 或 `road_traffic.py`。

---

## 5. 里程碑总览

| 里程碑 | 当前状态 | 目标 |
|---|---|---|
| R0 / M0 运行环境基线 | 已完成 | runtime checker、依赖、TanhNormal、road rollout |
| R1 / M1 Base/TSC 基线 | 已完成 | 纯 Base-MAPPO 与现有 TSC 可回归基线 |
| M2 Opinion 配置/入口 | 已完成 | 强类型配置和独立入口骨架 |
| M3 数学模块 | 已完成 | Evidence、Dynamics、Residual 纯张量实现 |
| M4 ConflictGraph | 已完成 | 当前物理量到 pair data 的环境接口 |
| M5 Opinion Policy | 已完成 | Base Actor + Opinion residual + TanhNormal |
| M6 Stateful Collector | 未开始 | 全局 ID 状态、单步更新、reset |
| M7 Sequence Buffer | 未开始 | 保留连续时间的 chunk 数据 |
| M8 Sequence PPO | 未开始 | chunk 内重算意见和 log-prob |
| M9 Trainer/Checkpoint | 未开始 | 三阶段训练和恢复 |
| M10 测试/诊断 | 未开始 | 测试入口、日志和可解释指标 |
| M11 消融/完整实验 | 未开始 | 多 seed 和方法比较 |

状态只能使用：`未开始`、`进行中`、`已完成`、`阻塞`。每完成一个里程碑必须更新本表及对应验证记录。

---

## 6. R0 / M0：恢复可复现运行时检查

### 6.1 目标

不修改算法，只把当前可用环境固化为可重复验证的工程合同。

### 6.2 新增文件

```text
scripts/check_runtime_environment.py
tests/test_runtime_environment_check.py
```

### 6.3 检查内容

- 从 `requirements.txt` 读取精确版本，不复制第二套硬编码版本；
- 导入 `torch`、`torchrl`、`tensordict`、`vmas`；
- 版本漂移、缺 pin、重复 pin、非精确 pin 必须失败；
- `TanhNormal.rsample()` 和 `log_prob()` 必须有限；
- 创建 `CPM_mixed` road traffic 环境；
- 执行 2～10 个随机动作 step；
- reset 和每个 step 的嵌套 TensorDict 浮点张量必须有限；
- 错误路径返回非零状态并打印清晰 `[FAIL]`。

### 6.4 验收命令

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_runtime_environment.py --steps 3
.venv/bin/python scripts/check_runtime_environment.py --steps 1
.venv/bin/python -m pip check
```

预期：

- 完整测试通过；
- 3-step smoke 返回 0；
- 非法 `--steps 1` 返回 1；
- 不修改 `requirements.txt`、scenario 和算法。

### 6.5 完成结果

R0/M0 已于 2026-08-19 完成：

- 新增 `scripts/check_runtime_environment.py`；
- 新增 `tests/test_runtime_environment_check.py`；
- 完整测试为 `10 passed`；
- 3-step `CPM_mixed` smoke 成功；
- 非法 `--steps 1` 返回 1 并输出 `[FAIL]`；
- `pip check` 和新增文件编译检查通过；
- 未修改 `requirements.txt`、scenario 或算法实现。

---

## 7. R1 / M1：固化 Base-MAPPO 与 TSC 基线

### 7.1 为什么不能只加配置

原始 `mappo_cavs.py` 会无条件构造和训练 `TopologyManager`。即使关闭 opponent、priority 和 topology neighbor selection，仅靠 JSON 仍不是纯 Base-MAPPO。

因此必须加入最小门控：

```python
uses_tsc = (
    is_using_opponent_modeling
    or use_topology_neighbor_selection
    or topology_loss_weight > 0
)

topology_manager = TopologyManager(...) if uses_tsc else None
```

并保护所有 topology 的 load、BCE、zero_grad、step、action predictor、日志和保存访问。

### 7.2 新增文件

```text
configs/baselines/base_mappo.json
configs/baselines/tsc.json
utilities/baseline_config.py
main_training_baseline.py
tests/test_baseline_config.py
tests/test_mappo_baseline_gating.py
tests/test_main_training_baseline.py
```

### 7.3 Base-MAPPO 必须关闭

```text
is_using_opponent_modeling = false
is_using_prioritized_marl = false
prioritization_method = "none"
use_topology_neighbor_selection = false
topology_loss_weight = 0.0
is_append_current_pos_to_short_refs_for_topology = false
n_topology_nearing_agents_observed = n_nearing_agents_observed
```

Base 运行时：

- 不构造 `TopologyManager`；
- 不计算 topology BCE；
- 不训练 action predictor；
- 不生成 topology/action-predictor/priority checkpoint；
- 仍使用原共享 Actor、集中式 Critic、collector、GAE 和 PPO。

### 7.4 TSC 必须保持原行为

```text
is_using_opponent_modeling = true
is_using_prioritized_marl = false
prioritization_method = "soft_label"
use_topology_neighbor_selection = true
topology_loss_weight = 0.5
is_append_current_pos_to_short_refs_for_topology = true
n_topology_nearing_agents_observed = 3
```

TSC 必须生成 topology 和 action-predictor checkpoint，但当前配置禁止 priority checkpoint。

### 7.5 配置和运行合同

- 固定 `road_traffic`、`CPM_mixed`、4 agents、CPU、seed 7；
- 两份正式配置除方法开关和输出目录外保持一致；
- 配置 schema 覆盖全部 committed 字段；
- bool 必须是真实 bool；
- int 必须是真实 int；
- 浮点超参必须有限并满足范围；
- 概率向量长度、范围和总和必须正确；
- 拒绝未知字段和缺失字段；
- baseline 必须从零训练，禁止 load/continue；
- W&B 默认强制 disabled；
- 命令必须从仓库根目录执行。

每次训练使用独立目录：

```text
outputs/baselines/base_mappo/runs/<run-id>/
outputs/baselines/tsc/runs/<run-id>/
```

`run-id` 必须是安全的单一路径段；`.`、`..`、隐藏段和已存在 run 必须拒绝。旧 run 的 checkpoint 或 metrics 不能满足新 run 的验收。

每个 run 至少包含：

```text
resolved_config.json
metrics.json
final_policy.pth
final_critic.pth
```

TSC 额外包含：

```text
final_topology.pth
final_action_predictor.pth
```

### 7.6 Smoke 验收

```bash
MPLBACKEND=Agg .venv/bin/python main_training_baseline.py --baseline base_mappo --smoke
MPLBACKEND=Agg .venv/bin/python main_training_baseline.py --baseline tsc --smoke
.venv/bin/python -m pytest -q
```

Smoke resolved copy：

```text
n_iters = 2
max_steps = 8
frames_per_batch = 16
num_epochs = 1
minibatch_size = 8
device = cpu
```

验收：

- 四组 reward/collision metrics 长度精确等于 2；
- reward 是非 bool 的有限数；
- collision rate 位于 `[0,1]`；
- snapshot 与本次 resolved config 精确一致；
- Base 和 TSC 产物合同分别满足；
- 两条路径训练中无 NaN/Inf。

### 7.7 完成结果

R1/M1 已于 2026-08-19 完成：

- Base-MAPPO 在运行时不构造或访问 `TopologyManager`；
- TSC 保留 topology learner 和 action predictor 原路径；
- 两份正式配置除方法字段和输出目录外保持一致；
- 每次运行使用唯一 `runs/<run-id>`，保存 resolved snapshot 和精确指标；
- Base 产物不含 topology、action-predictor 或 priority checkpoint；
- TSC 产物包含 topology/action-predictor，且不含 priority checkpoint；
- 两条 2-iteration CPU smoke 和完整测试均通过。

### 7.8 日常执行与验证

所有命令必须从仓库根目录执行：

```bash
cd /Users/zhangxiaotong/Code/sigmarl-nod
```

#### 7.8.1 Base-MAPPO smoke

```bash
MPLBACKEND=Agg .venv/bin/python main_training_baseline.py \
  --baseline base_mappo \
  --smoke
```

成功时命令末尾必须同时出现：

```text
[PASS] baseline=base_mappo snapshot=...
[PASS] metrics=...
```

输出目录：

```text
outputs/baselines/base_mappo/runs/<run-id>/
```

Base-MAPPO run 必须包含：

```text
resolved_config.json
metrics.json
final_policy.pth
final_critic.pth
```

允许保存中间 policy/critic checkpoint，但任何 `.pth` 文件名都不得包含：

```text
topology
action_predictor
priority
```

这项约束用于证明 Base-MAPPO 没有构造、训练或保存 TSC/priority 组件。

#### 7.8.2 TSC smoke

```bash
MPLBACKEND=Agg .venv/bin/python main_training_baseline.py \
  --baseline tsc \
  --smoke
```

成功时命令末尾必须同时出现：

```text
[PASS] baseline=tsc snapshot=...
[PASS] metrics=...
```

输出目录：

```text
outputs/baselines/tsc/runs/<run-id>/
```

除公共产物外，TSC run 还必须包含：

```text
final_topology.pth
final_action_predictor.pth
```

当前 TSC 基线关闭 prioritized MARL，因此不得出现文件名包含 `priority` 的
checkpoint。

#### 7.8.3 自动验收内容

`main_training_baseline.py` 会在训练结束后自动验证本次新 run，而不是从父目录
或旧 run 中查找文件。只有以下条件全部满足才会输出 `[PASS]`：

- `resolved_config.json` 与本次解析后的配置精确一致；
- policy、critic 及该方法要求的附加 checkpoint 齐全；
- Base/TSC 禁止的 checkpoint 不存在；
- `metrics.json` 包含四组指标；
- 每组指标长度精确等于 resolved `n_iters`；
- reward 为非 bool 的有限数；
- collision rate 为 `[0,1]` 内的有限数；
- 四组指标长度一致。

Smoke 模式的 `n_iters=2`，因此以下列表都必须恰好包含两个值：

```text
episode_reward_mean_list
collision_agents_rate_list
collision_lanelets_rate_list
collision_total_rate_list
```

#### 7.8.4 手工查看某次运行

列出 Base-MAPPO 和 TSC 的 run：

```bash
ls -td outputs/baselines/base_mappo/runs/*/
ls -td outputs/baselines/tsc/runs/*/
```

将下例中的目录替换为实际 run：

```bash
baseline_run_dir="outputs/baselines/base_mappo/runs/实际的run-id"
find "$baseline_run_dir" -maxdepth 1 -type f -print | sort
.venv/bin/python -m json.tool "$baseline_run_dir/resolved_config.json"
.venv/bin/python -m json.tool "$baseline_run_dir/metrics.json"
```

每次命令都会自动生成新的安全 `run-id`。不要手工复用已有 run 目录，也不要把
旧 checkpoint 复制进新 run。

#### 7.8.5 自动测试

只运行基线合同测试：

```bash
.venv/bin/python -m pytest -q \
  tests/test_baseline_config.py \
  tests/test_mappo_baseline_gating.py \
  tests/test_main_training_baseline.py
```

当前预期为 `63 passed`。

运行完整回归和依赖检查：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
```

当前完整工程预期分别为 `201 passed` 和 `No broken requirements found.`。

#### 7.8.6 正式训练

确认两条 smoke 和完整测试都成功后，移除 `--smoke` 启动正式训练：

```bash
MPLBACKEND=Agg .venv/bin/python main_training_baseline.py --baseline base_mappo
MPLBACKEND=Agg .venv/bin/python main_training_baseline.py --baseline tsc
```

正式配置使用 `250` 次迭代、每批 `4096` frames、每批 `60` 个优化 epoch，
计算量远高于 smoke。Opinion-MARL 训练链路完成前，日常开发验证只运行 smoke；
正式多 seed 对照实验留到 M11 统一执行。

---

## 8. M2：Opinion 配置与独立入口骨架

M2 只创建强类型配置和独立入口，不实现 EvidenceNet、Dynamics、Collector 或 PPO。

### 8.1 新增文件

```text
utilities/opinion/__init__.py
utilities/opinion/config.py
config_opinion.json
main_training_opinion.py
main_testing_opinion.py
tests/opinion/test_opinion_config.py
tests/opinion/test_opinion_entrypoints.py
```

### 8.2 对旧代码的最小修改

`Parameters` 只增加：

```python
use_opinion_marl: bool = False
opinion_config: Optional[dict] = None
```

要求：

- 旧 `config.json` 不包含新字段时行为完全不变；
- `config_opinion.json` 明确关闭所有 TSC/opponent/priority 路径；
- Opinion 入口不调用 `mappo_cavs()` 冒充新方法；
- 尚未实现训练核心时，入口必须明确返回“当前只完成 M2 骨架”。

### 8.3 OpinionConfig 字段

```text
stage
n_candidates
chunk_length
chunks_per_minibatch
evidence_hidden_dim
evidence_num_layers
b_max
b_temperature
kappa
nu
alpha
eta
z0
z_clip
n_substeps
residual_scale_start
residual_scale_target
residual_warmup_fraction
lr_actor
lr_evidence
lr_critic
neutral_loss_weight
magnitude_loss_weight
ttc_horizon
safe_distance
urgency_time_scale
urgency_distance_temperature
include_z_in_critic
log_pair_diagnostics
```

配置阶段必须计算并记录 `rho_c`，且在构造环境/网络前拒绝非法配置。

### 8.4 推荐初值

```json
{
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
```

这些值只是数值稳定的第一轮起点，不是论文最终调参结果。

### 8.5 已实现的配置合同

`utilities/opinion/config.py` 当前提供：

- 冻结的强类型 `OpinionConfig`；
- `stage` 只允许 `base`、`evidence`、`joint`；
- bool、正整数、有限浮点数和范围的严格校验；
- 缺失字段和未知字段拒绝；
- `rho_c = kappa / (nu * alpha)` 的派生计算；
- `nu * alpha > kappa` 检查；
- `0 < dt * eta * kappa < 2` 检查；
- `n_candidates <= n_agents - 1` 检查；
- resolved `to_dict()`，其中记录派生后的 `rho_c`；
- Opinion 根配置与 Base-MAPPO 公共实验条件的一致性校验；
- Opinion、opponent、priority、topology 开关的互斥隔离。

原始 `config_opinion.json` 不允许手工填写 `rho_c`，避免派生值和动力学参数
不一致；加载后的 resolved 配置会自动包含该值。推荐参数当前得到：

```text
rho_c = 1.0 / (1.0 * 2.0) = 0.5
```

### 8.6 使用方式

所有命令必须从仓库根目录执行：

```bash
cd /Users/zhangxiaotong/Code/sigmarl-nod
```

只验证默认 Opinion 训练配置：

```bash
.venv/bin/python main_training_opinion.py --validate-only
```

只验证默认 Opinion 测试配置：

```bash
.venv/bin/python main_testing_opinion.py --validate-only
```

当前成功输出分别为：

```text
[PASS] Opinion training configuration valid: stage=base rho_c=0.5 source=...
[PASS] Opinion testing configuration valid: stage=base rho_c=0.5 source=...
```

验证自定义配置：

```bash
.venv/bin/python main_training_opinion.py \
  --config path/to/opinion_config.json \
  --validate-only
```

配置合法时返回 0；配置非法时返回 1 并输出 `[FAIL]`。校验会在构造环境、
网络或 collector 之前完成。

当前若不带 `--validate-only`：

```bash
.venv/bin/python main_training_opinion.py
.venv/bin/python main_testing_opinion.py
```

入口会先验证配置，然后输出 `[NOT IMPLEMENTED]` 并返回状态码 2。这个返回值是
刻意设计的：M2 只完成配置和入口，不允许把 Base-MAPPO/TSC 训练冒充为
Opinion-MARL。只有后续里程碑完成相应训练/测试核心后，默认入口才能返回成功。

运行 M2 目标测试：

```bash
.venv/bin/python -m pytest -q \
  tests/opinion/test_opinion_config.py \
  tests/opinion/test_opinion_entrypoints.py
```

当前预期为 `63 passed`。完整回归：

```bash
.venv/bin/python -m pytest -q
```

当前预期为 `201 passed`。

### 8.7 完成边界

M2 已完成配置、解析、隔离验证和 CLI 骨架，但仍未实现：

```text
OpinionAugmentedPolicy
stateful collector
sequence PPO
真实 Opinion 训练和测试
```

因此 `--validate-only` 成功只表示配置合同成立，不表示 Opinion 方法已经训练、
推理或产生性能结果。

---

## 9. M3：纯数学模块

新增：

```text
utilities/opinion/evidence_net.py
utilities/opinion/dynamics.py
utilities/opinion/residual.py
```

本阶段禁止接环境、collector 和 PPO。

实现顺序：

1. `swap_roles()`；
2. 共享相对评分；
3. `raw_b` 和门控后的 `b`；
4. 无状态 `OpinionDynamics.forward()`；
5. candidate edge 与 dense state 的 gather/scatter；
6. residual 聚合。

必须测试：

- Evidence 上界；
- mask/padding 为零；
- 角色交换反号；
- 无冲突时 `z` 衰减；
- `rho < rho_c` 时小扰动回零；
- `rho > rho_c` 时正负扰动进入不同分支；
- 强反向证据可以翻转意见；
- residual 有界；
- 对角线为零；
- Dynamics 参数不参与梯度。

完成后达到 Gate A。

### 9.1 已实现接口

`utilities/opinion/evidence_net.py`：

- `swap_roles()` 交换车辆角色并反转 antisymmetric context；
- `OpinionEvidenceNet` 使用同一个 shared scorer 分别计算双方评分；
- `raw_b = b_max * tanh((s_i-s_j)/b_temperature)`；
- `b = raw_b * urgency * confidence * mask`；
- 被 mask 的 padding 即使含 NaN 也会在进入网络前安全归零；
- 有效边上的非有限特征或越界 gate 会明确失败；
- forward 接口不接受 `z` 或 `q`。

`utilities/opinion/dynamics.py`：

- `OpinionDynamics` 以 `dt/n_substeps` 做固定 Euler 子步更新；
- 每个子步后将状态裁剪到 `[-z_clip,z_clip]`；
- invalid edge 不注入证据或自强化，只进行自然衰减；
- 类中没有 `nn.Parameter`，但梯度可从 `z_next` 传播到 `b`；
- `gather_candidate_opinions()` 按 global neighbor ID 读取 dense state；
- `scatter_candidate_opinions()` 返回新的 dense state 并强制 diagonal 为零；
- active 越界 ID、自环和重复 candidate 会明确失败。

`utilities/opinion/residual.py`：

- `q = tanh(z/z0)`；
- urgency 在 candidate 维归一化；
- 使用 `[−1,1]` 内的 signed direction 聚合；
- 最终通过 `residual_scale * tanh(aggregate)` 保证绝对上界；
- 无有效边时 `q`、权重和 residual 严格为零；
- 候选车辆数量增加不会放大理论上界。

三个模块可直接从 `utilities.opinion` 导入：

```python
from utilities.opinion import (
    OpinionDynamics,
    OpinionEvidenceNet,
    OpinionResidual,
    gather_candidate_opinions,
    scatter_candidate_opinions,
    swap_roles,
)
```

### 9.2 张量接口约定

Evidence 输入采用统一 batch 前缀：

```text
ego_features             [..., K, D_individual]
neighbor_features        [..., K, D_individual]
symmetric_context        [..., K, D_symmetric]
antisymmetric_context    [..., K, D_antisymmetric]
urgency/confidence/mask  [..., K]
raw_b/b                  [..., K]
```

Dynamics 对 `z_prev`、`b`、`urgency`、`mask` 做逐边同形更新。

全局 ID 状态转换采用：

```text
z_dense          [..., n_agents, n_agents]
candidate_ids    [..., n_agents, n_candidates]
candidate_mask   [..., n_agents, n_candidates]
candidate_z      [..., n_agents, n_candidates]
```

Residual 在最后一个 candidate 维聚合：

```text
z/urgency/direction/mask  [..., K]
residual                  [...]
```

当前 `direction` 是 Opinion 对速度均值的有符号标量方向。M5 才负责把该标量
只加到 `loc[...,0]`；M3 不接 Actor 或动作分布。

### 9.3 验证方式和完成边界

只运行 M3 测试：

```bash
.venv/bin/python -m pytest -q \
  tests/opinion/test_evidence_net.py \
  tests/opinion/test_dynamics.py \
  tests/opinion/test_residual.py \
  tests/opinion/test_math_pipeline.py
```

当前预期为 `28 passed`。Opinion 测试全集和完整工程回归分别为：

```bash
.venv/bin/python -m pytest -q tests/opinion
.venv/bin/python -m pytest -q
```

当前预期分别为 `128 passed` 和 `201 passed`。

M3 已达到 Gate A，M4 已达到 Gate B，但仍未连接 collector、Actor、PPO 或
checkpoint。`main_training_opinion.py` 不带 `--validate-only` 时仍必须返回
`[NOT IMPLEMENTED]` 和状态码 2。

---

## 10. M4：ConflictGraph 环境接口

M4 已完成。新增：

```text
utilities/opinion/conflict_graph.py
tests/opinion/test_conflict_graph.py
tests/opinion/test_road_traffic_conflict_info.py
```

修改：

```text
scenarios/road_traffic.py
utilities/opinion/__init__.py
```

### 10.1 当前物理量与候选边

`ConflictGraph` 是无状态纯张量模块。输入为：

```text
positions:       [E, N, 2]
velocities:      [E, N, 2]
headings:        [E, N]
visibility_mask: [E, N, N]，bool，方向为 ego i → neighbor j
```

它只利用当前 `pos/vel/rot` 做 constant-velocity closest-approach 几何计算：

```text
r_ij     = p_j - p_i
v_ij     = v_j - v_i
t_cpa    = clamp(-<r_ij,v_ij> / ||v_ij||², 0, ttc_horizon)
d_cpa    = ||r_ij + t_cpa v_ij||
```

这里的 CPA 是从当前物理量得到的解析量，不读取仿真器未来状态、真实未来轨迹、
short-term reference path 或 topology label。

每条 pair feature 固定为 12 维，顺序由
`utilities.opinion.conflict_graph.PAIR_FEATURE_NAMES` 唯一定义：

```text
0  relative_position_longitudinal（ego 坐标系）
1  relative_position_lateral
2  relative_velocity_longitudinal
3  relative_velocity_lateral
4  distance
5  closing_speed
6  time_to_closest_approach
7  distance_at_closest_approach
8  heading_difference_sin
9  heading_difference_cos
10 ego_speed
11 neighbor_speed
```

候选排序是确定性的：`urgency` 降序、当前距离升序、global agent ID 升序。
无效槽位统一输出 `neighbor_id=-1`、`pair_mask=false`，其 features、urgency、
confidence 均为 0。当前仿真器使用精确状态，因此可见有效 pair 的
`confidence=1`，不可见/填充 pair 为 0。

### 10.2 urgency 定义

仅当车辆正在接近，或当前距离已经不大于 `safe_distance` 时激活冲突：

```text
time_score     = exp(-t_cpa / urgency_time_scale)
distance_score = sigmoid((safe_distance - d_cpa)
                         / urgency_distance_temperature)
urgency        = active_conflict * time_score * distance_score * confidence
```

输出严格位于 `[0,1]`。平行同速和明确背离车辆在安全距离外的 urgency 为 0；
迎面、交叉和接近静止车辆可得到正 urgency。

### 10.3 road_traffic info 合同

仅当 `parameters.use_opinion_marl=true` 时，`road_traffic.info(agent)` 增加：

```text
pair_features:    raw per agent [E,K,12] → TorchRL [E,N,K,12]
neighbor_ids:     raw per agent [E,K]    → TorchRL [E,N,K]
pair_mask:        raw per agent [E,K]    → TorchRL [E,N,K]
urgency:          raw per agent [E,K]    → TorchRL [E,N,K]
confidence:       raw per agent [E,K]    → TorchRL [E,N,K]
agent_reset_mask: raw per agent [E]      → TorchRL [E,N,1]
```

`environment_done [E]` 继续使用 VMAS/TorchRL 顶层 `done/terminated`，不在每个
agent 的 info 中重复复制；M6 必须同时读取顶层 done 和上述 partial reset mask。

抽象方法接口中的 reset mask 仍是 `[E,N]`；TorchRL 0.2.1 会自动为标量 info
增加最后一个 metric 维度，因此环境 TensorDict 中实际为 `[E,N,1]`，M5/M6
读取时应对最后一维执行 `squeeze(-1)`。

全局 ID 在纯模块和 raw scenario info 中为 `int64`。TorchRL 0.2.1 的
`read_info()` 会把全部 info 转成 float32，因此后续从 TensorDict 读取
`neighbor_ids` 时必须显式 `.long()`，不能把候选槽位号当成车辆 ID。

reset 事件满足：

- 全环境 reset 标记该环境全部 agent；
- 直接 single-agent reset 只标记对应全局 ID；
- `done()` 内部的 partial reset 会在 reset 前一条 transition 的 info 中报告；
- 该事件只报告一次，不会在物理 reset 后重复报告。

Base-MAPPO/TSC 在开关关闭时既不构造 `ConflictGraph`，也不增加上述 info key。
ConflictGraph 的构造和输出不读取 `use_topology_neighbor_selection` 或
`n_topology_nearing_agents_observed`。

### 10.4 验证方式

只运行 M4 测试：

```bash
.venv/bin/python -m pytest -q \
  tests/opinion/test_conflict_graph.py \
  tests/opinion/test_road_traffic_conflict_info.py
```

当前预期：`20 passed`。Opinion 全集和完整回归：

```bash
.venv/bin/python -m pytest -q tests/opinion
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
```

当前预期分别为 `128 passed`、`201 passed, 13 warnings` 和
`No broken requirements found`。

### 10.5 完成边界

环境禁止：

- 保存或更新 `z`；
- 调用 EvidenceNet；
- 使用未来真实轨迹生成执行期输入；
- 依赖 TSC topology selection 开关。

已覆盖：迎面接近、平行同速、背离、交叉接近、静止邻车、不可见邻车、
稳定全局 ID、直接 single-agent reset、自动 partial reset、未来 reference
隔离、TSC 开关隔离和 Base 路径零新增输出。

M4 已达到 Gate B，但没有保存/更新 `z`，没有调用 EvidenceNet，没有改变 Actor
或动作分布，也没有解除训练/测试入口的 `[NOT IMPLEMENTED]`。这些属于 M5/M6
及后续里程碑。

---

## 11. M5：OpinionAugmentedPolicy

M5 已完成。新增：

```text
utilities/opinion/policy.py
tests/opinion/test_policy.py
```

### 11.1 模块和职责

```text
BaseGaussianActor
    共享参数的分散式 MLP，输出 base loc/scale

PairInteractionEncoder
    将 M4 的 12 维 pair feature 拆成角色明确的 EvidenceNet 输入

OpinionEvidenceNet
    生成 raw_b 和经过 urgency/confidence/mask 门控的 b

OpinionDynamics
    单步执行 z_prev → z_next，但不持有跨步状态

OpinionResidual
    聚合有向意见并输出有界标量速度 residual

OpinionAugmentedPolicyCore
    组合 Base Actor、Evidence、Dynamics 和 Residual，先生成 final loc

OpinionTanhNormalPolicy
    使用 final loc 和原 scale 构造 TanhNormal，采样或重算 log-prob
```

M5 没有修改旧 `mappo_cavs.py`，也没有接管 `z_dense`。Base/TSC 继续使用原有
Policy；Opinion Policy 是独立模块。

### 11.2 InteractionEncoder 固定映射

M4 的 12 维特征全部保留，并按 EvidenceNet 的相对评分接口拆分：

```text
ego_features [1]:
  ego_speed

neighbor_features [1]:
  neighbor_speed

symmetric_context [5]:
  distance
  closing_speed
  time_to_closest_approach
  distance_at_closest_approach
  heading_difference_cos

antisymmetric_context [5]:
  relative_position_longitudinal
  relative_position_lateral
  relative_velocity_longitudinal
  relative_velocity_lateral
  heading_difference_sin
```

EvidenceNet 仍使用同一个 scorer 做角色交换差分，因此网络不能读取 `z/q`，
且瞬时证据继续满足有界和 mask 合同。

### 11.3 单步 Policy 因果顺序

```text
observation ───────────────→ BaseGaussianActor → base_loc, scale
pair_features → InteractionEncoder → EvidenceNet → b
z_prev + b + urgency ──────→ OpinionDynamics → z_next
z_next + urgency ──────────→ OpinionResidual → residual
base_loc[...,0] + residual → final_loc[...,0]
base_loc[...,1] ───────────→ final_loc[...,1]（原样）
final_loc + 原 scale ──────→ TanhNormal → action/log_prob
```

固定约束：

- Base stage residual 严格为零；
- Base stage 不调用 EvidenceNet，并输出全零 `b/z_next`；
- residual 只改变速度 loc；
- 转向 loc 和 scale 不变；
- 必须先修改 loc，再构造 TanhNormal；
- `residual_scale=0` 是精确 Base Policy 消融；
- `z_next` 有限；
- invalid/padding 输入不能改变动作；
- `direction` 默认固定为 `+1`，即正意见增加前进倾向、负意见降低前进倾向；
- 相同输入和 `z_prev` 必须重算出相同 loc/scale/log-prob；
- VMAS 的 `[N,A]` per-agent action bounds 和普通 `[A]` bounds 均支持。

### 11.4 直接构造方式

```python
core = OpinionAugmentedPolicyCore.from_config(
    observation_dim=observation.shape[-1],
    action_dim=2,
    config=opinion_config,
    dt=parameters.dt,
)

policy = OpinionTanhNormalPolicy(
    core=core,
    action_low=action_spec.space.low,
    action_high=action_spec.space.high,
)

result = policy(
    observation=observation,
    pair_features=pair_features,
    urgency=urgency,
    confidence=confidence,
    pair_mask=pair_mask,
    z_prev=z_prev,
    residual_scale=current_residual_scale,
)
```

输出包括：

```text
action、log_prob
base_loc、final_loc、scale
raw_b、b、z_next、q
normalized_weights、aggregate、residual
```

如果传入 rollout 已保存的 `action`，wrapper 不重新采样，而是使用同一个
final distribution 重算 log-prob；这为 M8 PPO ratio 重算提供单步基础。

### 11.5 验证方式

```bash
.venv/bin/python -m pytest -q tests/opinion/test_policy.py
.venv/bin/python -m pytest -q tests/opinion
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
```

当前预期分别为 `17 passed`、`128 passed`、`201 passed, 13 warnings` 和
`No broken requirements found`。

### 11.6 完成边界

M5 已达到 Gate C，但仍然只是单步纯 Policy。它要求调用方显式传入
`z_prev [E,N,K]` 并返回 `z_next`；当前没有模块持有 `[E,N,N]` 的跨时间状态，
也没有处理 global neighbor ID gather/scatter、done/reset、rollout buffer、PPO、
optimizer 或 checkpoint。

因此 Opinion 训练/测试入口仍必须返回 `[NOT IMPLEMENTED]` 和状态码 2。下一步
M6 由 Stateful Collector 成为 `z_dense` 生命周期的唯一拥有者。

---

## 12. M6：Stateful Opinion Collector

新增：

```text
utilities/opinion/collector.py
```

Collector 是 `z` 生命周期的唯一拥有者：

- 初始化 `[E,N,N]`；
- 每个物理步更新一次；
- rollout 保存 `z_prev`、`z_next`、`b`、candidate IDs/mask；
- done/reset 后正确清零；
- 不跨环境共享状态。

必须测试异步环境结束、agent partial reset、neighbor 换槽、neighbor 消失和重新出现。

完成后达到 Gate D。

---

## 13. M7：Sequence Buffer

新增：

```text
utilities/opinion/sequence_buffer.py
```

标准 MAPPO 可以展平 rollout，但本方法的 `z` 有时序依赖，因此必须按连续 chunk 采样：

```text
rollout [T,E,N,...]
→ 连续 chunk [B_chunk,L,N,...]
→ 保存每个 chunk 的 z_init
→ chunk 内顺序重放
```

约束：

- `chunk_length > 0`；
- chunk 不跨 done 边界；
- `z_init.detach()` 截断 chunk 之前的梯度；
- chunk 内梯度保持；
- 重新拼接能恢复原时间顺序。

---

## 14. M8：Sequence PPO 重算与梯度隔离

新增：

```text
utilities/opinion/ppo_loss.py
```

PPO 更新时不能只读取 rollout 中存好的 `z`。必须从 `z_init` 开始在 chunk 内重新计算：

```text
pair features
→ EvidenceNet
→ b_t
→ Dynamics(z_{t-1}, b_t)
→ residual
→ loc_t
→ log_prob_t
```

必须验证：

- 参数未更新时，重算 log-prob 与 rollout log-prob 在容差内一致；
- Actor loss backward 后 EvidenceNet 有非零梯度；
- Critic loss backward 后 EvidenceNet 无梯度；
- Dynamics 参数无梯度；
- 早期证据能影响 chunk 后期状态和 loss；
- invalid/padding/done 部分不产生梯度污染。

完成后达到 Gate E。

---

## 15. M9：三阶段 Trainer 与 Checkpoint

新增：

```text
utilities/opinion/trainer.py
utilities/opinion/checkpoint.py
```

训练阶段：

```text
Stage base:
  residual = 0
  训练 Base Actor/Critic

Stage evidence:
  可冻结或低学习率更新 Base Actor
  主要训练 EvidenceNet
  residual 从 0 warm-up

Stage joint:
  Actor、EvidenceNet、Critic 使用独立学习率联合微调
```

Optimizer 约束：

- Actor、Evidence、Critic 参数集合明确且不重复；
- Dynamics 参数不在任何 optimizer；
- Critic optimizer 不包含 EvidenceNet；
- gradient clipping 分模块执行；
- checkpoint 保存 schema version、配置、stage、网络和 optimizer；
- 第一版不要求恢复 collector 中间 `z`，但必须明确 episode 边界恢复规则。

完成后达到 Gate F。

---

## 16. M10：测试入口和诊断日志

新增：

```text
main_testing_opinion.py
utilities/opinion/diagnostics.py
```

至少记录：

- reward；
- agent/lane/total collision rate；
- `raw_b`、`b` 均值/方差/饱和率；
- `z` 均值/方差/绝对值/翻转率；
- opinion residual 幅值和饱和率；
- conflict edge 数量、mask 比例；
- reset 次数；
- EvidenceNet 梯度范数；
- Actor/Critic 梯度范数；
- 不同 stage 当前 residual scale。

测试 checkpoint 保存/加载后，相同输入和 `z` 产生相同 distribution 参数。

---

## 17. M11：消融和正式实验

需要比较：

```text
Base-MAPPO
TSC
Direct evidence residual（无动态记忆）
EMA evidence
Linear dynamics
GRU memory
Handcrafted evidence + fixed dynamics
Learned evidence + fixed dynamics（Full）
```

实验维度：

- 多随机种子；
- 训练场景和未见场景；
- 感知噪声；
- 观测延迟；
- 邻车遮挡；
- 不同交通密度；
- 不同冲突类型。

任务指标：reward、碰撞率、完成率、通行时间、舒适性。  
意见指标：形成时间、翻转次数、饱和率、冲突边持续时间、残差大小。  
统计必须报告均值、标准差和 seed 数量。

---

## 18. 全局测试门

### Gate A：数学模块

- Evidence/Dynamics/Residual 全部纯张量测试通过；
- 未接环境和 PPO。

### Gate B：环境接口

- Opinion info key 形状固定；
- global neighbor ID 稳定；
- reset mask 正确；
- 无未来信息泄漏；
- 不受 TSC 开关影响。

### Gate C：单步 Policy

- 可采样 action；
- log-prob 正确；
- residual 只影响速度 loc；
- `z_next` 有限。

### Gate D：Collector

- 每物理步只更新一次；
- partial/full reset 正确；
- 多环境无串扰。

### Gate E：Sequence PPO

- 重算 log-prob 一致；
- Actor→Evidence 梯度存在；
- Critic→Evidence 梯度不存在；
- chunk 内时序梯度正确。

### Gate F：训练与恢复

- 三阶段 smoke 可运行；
- checkpoint 可保存加载；
- 无 NaN/Inf；
- Base/TSC 回归不破坏。

---

## 19. 永久禁止事项

- 不把新方法写成 TSC 的理论扩展；
- 不复用 priority、leader、Stackelberg、topology label 作为 Opinion 信号；
- 不在环境回调中更新 `z`；
- 不按邻居槽位保存 `z`；
- 不在 action 采样后添加 residual；
- 不随机打散全部时间步做单步 PPO；
- 不让 Critic loss 更新 EvidenceNet；
- 不学习第一版 Dynamics 参数；
- 不使用真实未来轨迹作为执行期输入；
- 不把 `b` 当作长期记忆；
- 不宣称形式化安全保证、全局意见一致或全局最优；
- 不提交 `.venv`、`outputs`、checkpoint 或训练二进制产物。

---

## 20. 每个里程碑的标准执行方式

用户负责版本控制，开发 Session 只负责文件修改和验证：

```text
1. 确认当前目录是 sigmarl-nod
2. 阅读本文件和当前里程碑源码
3. 记录修改前测试结果
4. 先写会因缺少行为而失败的测试
5. 确认 RED 原因正确
6. 编写最小实现
7. 确认目标测试 GREEN
8. 运行全部历史测试
9. 运行该里程碑 smoke/integration
10. 检查 NaN/Inf、输出和接口合同
11. 更新本文件的状态和验证记录
12. 报告修改文件、命令、结果和遗留问题
```

禁止执行任何 Git 写操作。不要创建分支、worktree、commit 或合并；这些由用户处理。

---

## 21. 完成定义

第一版实现只有同时满足以下条件才算完成：

- [x] R0/M0 runtime 基线已恢复并通过；
- [x] R1/M1 Base-MAPPO/TSC 基线已恢复并通过；
- [x] 新 Opinion 入口不调用 TSC coordination 路径；
- [x] EvidenceNet 输出有界且不读取 `z/q`；
- [x] Dynamics 固定且无梯度；
- [ ] `z` 按全局 agent ID 保存；
- [ ] `z` 每物理步只更新一次；
- [ ] partial/full reset 正确；
- [ ] residual 只修改速度 loc 且有界；
- [ ] rollout 保留连续时间；
- [ ] chunk 重算 log-prob 一致；
- [ ] Actor loss 能训练 EvidenceNet；
- [ ] Critic loss 不能训练 EvidenceNet；
- [ ] 训练无 NaN/Inf；
- [ ] checkpoint 可保存/加载；
- [ ] Opinion 诊断日志完整；
- [ ] 主要消融可配置；
- [ ] TSC 只作为外部基线完成比较。

---

## 22. 当前验证记录

### 2026-08-19：新仓库初始核对

仓库：`/Users/zhangxiaotong/Code/sigmarl-nod`

结果：

- 核心源码与开始修改前版本一致；
- 理论路线存在；
- M0/M1 实现文件不存在；
- `.venv` 可用；
- Python 3.9.13；
- torch 2.1.0；
- torchrl 0.2.1；
- tensordict 0.2.1；
- vmas 1.4.1；
- `.venv/bin/python -m pytest -q`：`2 passed`；
- `.venv/bin/python -m pip check`：无冲突。

当时下一步：执行 R0/M0，不直接进入 M2。

### 2026-08-19：R0/M0 运行时基线完成

新增文件：

- `scripts/check_runtime_environment.py`；
- `tests/test_runtime_environment_check.py`。

测试先行证据：

- 只加入测试后运行目标测试，因缺少
  `scripts.check_runtime_environment` 得到预期 `ModuleNotFoundError`；
- 加入最小实现后，目标测试为 `8 passed`；
- `.venv/bin/python -m pytest -q`：`10 passed, 13 warnings`；
- `.venv/bin/python scripts/check_runtime_environment.py --steps 3`：成功；
- `.venv/bin/python scripts/check_runtime_environment.py --steps 1`：返回 1，
  且输出明确的 `[FAIL]` 与参数范围错误；
- `.venv/bin/python -m pip check`：`No broken requirements found`；
- 新增脚本和测试的 `compileall` 检查通过。

运行时锁定结果：Python 3.9.13、torch 2.1.0、torchrl 0.2.1、
tensordict 0.2.1、vmas 1.4.1。已知提示仅包括上游 Gym 停止维护消息和
matplotlib/pyparsing 弃用警告，不影响本阶段 smoke。

下一步：只执行 R1/M1，不直接进入 M2。

### 2026-08-19：R1/M1 Base-MAPPO 与 TSC 基线完成

新增文件：

- `configs/baselines/base_mappo.json`；
- `configs/baselines/tsc.json`；
- `utilities/baseline_config.py`；
- `main_training_baseline.py`；
- `tests/test_baseline_config.py`；
- `tests/test_mappo_baseline_gating.py`；
- `tests/test_main_training_baseline.py`。

修改文件：

- `utilities/mappo_cavs.py`：增加 `uses_tsc_components()` 和显式
  `TopologyManager` 构造/训练/加载/保存门控。

测试先行证据：

- 首批测试因缺少 `utilities.baseline_config`、`uses_tsc_components` 和
  `main_training_baseline` 得到预期的 3 个 collection errors；
- 配置、入口、门控和产物合同目标测试：`63 passed`；
- `.venv/bin/python -m pytest -q`：`73 passed, 13 warnings`；
- `.venv/bin/python -m pip check`：`No broken requirements found`；
- M1 新增/修改 Python 文件的 `compileall` 检查通过。

真实 Base-MAPPO smoke：

```text
run: outputs/baselines/base_mappo/runs/
     smoke-20260819T061355664526Z-e9444045/
reward:    [-0.3993280529975891, -0.5993180274963379]
collision: [0.0625, 0.125]
```

该 run 只包含 policy/critic 类型 checkpoint，不含 topology、
action-predictor 或 priority checkpoint。

真实 TSC smoke：

```text
run: outputs/baselines/tsc/runs/
     smoke-20260819T061406329807Z-e14ee076/
reward:    [-0.3367869555950165, -0.27221325039863586]
collision: [0.0625, 0.0625]
```

该 run 包含 `final_topology.pth` 和 `final_action_predictor.pth`，不含
priority checkpoint。两个 run 的 snapshot、四组精确 2-iteration 指标和
产物身份均已再次独立验证。

已知提示仍仅包括上游 Gym 停止维护消息和 matplotlib/pyparsing 弃用警告。

下一步：执行 M2，只建立 Opinion 强类型配置与独立入口骨架，不提前实现
EvidenceNet、OpinionDynamics、Collector 或 PPO。

### 2026-08-19：M2 Opinion 配置与独立入口骨架完成

新增文件：

- `utilities/opinion/__init__.py`；
- `utilities/opinion/config.py`；
- `config_opinion.json`；
- `main_training_opinion.py`；
- `main_testing_opinion.py`；
- `tests/opinion/test_opinion_config.py`；
- `tests/opinion/test_opinion_entrypoints.py`。

修改文件：

- `utilities/helper_training.py`：以 Python 3.9 兼容形式增加默认关闭的
  `use_opinion_marl` 和 `opinion_config`。

测试先行证据：

- 首批测试因缺少 `utilities.opinion` 和两个 Opinion 入口得到预期的
  2 个 collection errors；
- 增加 resolved `rho_c` 记录接口前，单测得到预期 `AttributeError`；
- M2 配置和入口目标测试：`63 passed`；
- `.venv/bin/python -m pytest -q`：`136 passed, 13 warnings`；
- `.venv/bin/python -m pip check`：`No broken requirements found`；
- M2 新增文件的 `compileall` 检查通过。

入口实测：

- `main_training_opinion.py --validate-only`：返回 0，`stage=base`，
  `rho_c=0.5`；
- `main_testing_opinion.py --validate-only`：返回 0，`stage=base`，
  `rho_c=0.5`；
- 两个入口不带 `--validate-only`：均返回 2 并输出 `[NOT IMPLEMENTED]`；
- 非法 TSC 开关会在任何环境或网络构造前返回 1 和 `[FAIL]`。

M2 没有实现或调用 `mappo_cavs()`、EvidenceNet、OpinionDynamics、Collector
或 PPO，也没有生成 Opinion 训练结果。

下一步：执行 M3，只实现并测试 EvidenceNet、固定 OpinionDynamics 和
OpinionResidual 三个纯张量数学模块；禁止接入环境、collector 或 PPO。

### 2026-08-19：M3 纯数学模块完成

新增文件：

- `utilities/opinion/evidence_net.py`；
- `utilities/opinion/dynamics.py`；
- `utilities/opinion/residual.py`；
- `tests/opinion/test_evidence_net.py`；
- `tests/opinion/test_dynamics.py`；
- `tests/opinion/test_residual.py`；
- `tests/opinion/test_math_pipeline.py`。

修改文件：

- `utilities/opinion/__init__.py`：公开 M3 类、输出类型和状态转换函数。

测试先行证据：

- 只加入 M3 测试时，因缺少 `evidence_net`、`dynamics`、`residual` 得到
  预期的 3 个 collection errors；
- 三个模块首轮目标测试：`27 passed`；
- 加入完整数学链梯度测试后，M3 目标测试：`28 passed`；
- Opinion 测试全集：`91 passed`；
- `.venv/bin/python -m pytest -q`：`164 passed, 13 warnings`；
- `.venv/bin/python -m pip check`：`No broken requirements found`；
- M3 模块和测试的 `compileall` 检查通过。

关键验证结果：

- `EvidenceNet` trainable parameters：497；
- `OpinionDynamics` trainable parameters：0；
- `OpinionResidual` trainable parameters：0；
- `rho_c=0.5`；
- 角色交换后 `raw_b` 和 `b` 精确反号；
- `rho<rho_c` 时小扰动回零，`rho>rho_c` 时进入正负分支；
- `b=-0.5` 可以翻转已建立的正意见；
- residual 不超过 `residual_scale`；
- `EvidenceNet → Dynamics → Residual` 端到端梯度有限且只更新 EvidenceNet；
- M3 源文件不引用 road traffic、collector、PPO 或 TSC 组件。

M3 没有修改 `scenarios/road_traffic.py`，没有持久保存跨时间 `z`，也没有解除
Opinion 训练/测试入口的 `[NOT IMPLEMENTED]` 状态。

下一步：执行 M4，建立只依赖当前物理量的 ConflictGraph 和 road-traffic
环境信息接口；环境禁止保存或更新 `z`，禁止调用 EvidenceNet。

### 2026-08-19：M4 ConflictGraph 环境接口完成

新增文件：

- `utilities/opinion/conflict_graph.py`；
- `tests/opinion/test_conflict_graph.py`；
- `tests/opinion/test_road_traffic_conflict_info.py`。

修改文件：

- `scenarios/road_traffic.py`：增加默认关闭的 current-physics info 路径和
  agent reset 事件接口；
- `utilities/opinion/__init__.py`：公开 ConflictGraph 接口；
- `main_training_opinion.py`、`main_testing_opinion.py`：保持状态码 2，仅将
  `[NOT IMPLEMENTED]` 边界说明更新到 M4；
- `OPINION_MARL_IMPLEMENTATION_GUIDE.md`：记录接口、维度、使用方式和验证证据。

测试先行证据：

- 纯模块测试首次运行因缺少 `utilities.opinion.conflict_graph` 得到预期
  `ModuleNotFoundError`；
- 环境接口测试首次运行得到预期 `5 failed, 1 passed`，失败原因为 info key、
  graph builder 和 reset mask 尚未实现；
- M4 目标测试：`20 passed`；
- Opinion 测试全集：`111 passed`；
- `.venv/bin/python -m pytest -q`：`184 passed, 13 warnings`；
- `.venv/bin/python -m pip check`：`No broken requirements found`；
- 2 个并行环境的真实 Opinion VMAS random rollout 连续 5 步通过，所有新增
  浮点 info 有限且形状稳定。

关键验证结果：

- pair feature 固定为 12 维，并由 `PAIR_FEATURE_NAMES` 明确语义和顺序；
- 候选 global ID 稳定，padding ID 为 `-1`；
- head-on、crossing 和 approaching-stationary 得到正 urgency；
- parallel-same-speed、diverging 和 invisible pair 不产生有效冲突证据；
- 修改 short-term reference future points 不改变任何 ConflictGraph 输出；
- 切换 TSC topology selection 参数不改变 ConflictGraph 输出；
- direct single-agent reset 和 `done()` 自动 partial reset 均只报告一次；
- Base/TSC 开关关闭时不构造 graph、不增加 M4 info key。

M4 没有保存或更新 `z`，没有调用 EvidenceNet，没有改 Actor、collector、PPO
或 checkpoint。Opinion 训练/测试入口仍保持 `[NOT IMPLEMENTED]` 和状态码 2。

下一步：执行 M5，实现单步 `OpinionAugmentedPolicy`，保证 residual 只修改速度
通道的分布均值，并在修改 loc 后构造 TanhNormal 和计算 log-prob。

### 2026-08-20：M5 OpinionAugmentedPolicy 完成

新增文件：

- `utilities/opinion/policy.py`；
- `tests/opinion/test_policy.py`。

修改文件：

- `utilities/opinion/__init__.py`：公开 M5 Actor、Encoder、Core、wrapper 和输出类型；
- `main_training_opinion.py`、`main_testing_opinion.py`：继续保持状态码 2，并将
  `[NOT IMPLEMENTED]` 边界说明更新到 M5；
- `OPINION_MARL_IMPLEMENTATION_GUIDE.md`：记录 M5 接口、映射、用法和验证证据。

测试先行证据：

- 首轮 M5 测试因缺少 `utilities.opinion.policy` 得到预期
  `ModuleNotFoundError`；
- 真实 VMAS smoke 首次发现 action bounds 是 `[N,A]` 而非仅 `[A]`，加入
  per-agent bounds 测试后旧实现按预期失败，再扩展为严格 broadcast 合同；
- Base-stage 非法 residual scale 测试首次得到预期 `4 failed`，随后将严格
  scale 验证移动到所有 stage 的公共入口；
- M5 目标测试：`17 passed`；
- Opinion 测试全集：`128 passed`；
- `.venv/bin/python -m pytest -q`：`201 passed, 13 warnings`；
- `.venv/bin/python -m pip check`：`No broken requirements found`；
- M5 源码和测试 `compileall` 通过。

真实环境单步 smoke：

```text
num_envs=2
n_agents=4
observation=[2,4,32]
pair_features=[2,4,3,12]
action=[2,4,2]
log_prob=[2,4]
EvidenceNet trainable parameters=18305
max |residual|=0.0001648313（随机初始化单步样本）
VMAS next observation finite
```

关键验证结果：

- Base stage 精确复现 Base Actor 的 loc/scale，且不调用 EvidenceNet；
- joint/evidence stage 的 residual 只加到 `loc[...,0]`；
- steering `loc[...,1]` 和全部 scale 精确不变；
- invalid/padding NaN 被隔离，不产生 evidence、opinion 或 residual；
- 采样 action 和 log-prob 有限且位于动作边界；
- 使用保存 action 重算得到相同 loc/scale 和容差内相同 log-prob；
- Actor loss 路径可向 EvidenceNet 产生有限非零梯度；
- Dynamics 和 Residual 仍无可训练参数；
- M5 没有依赖 TSC topology、priority、opponent modeling 或 action predictor。

M5 没有持有跨物理步状态，也没有实现 collector、sequence buffer、PPO、trainer、
checkpoint 或 evaluation。Opinion 训练/测试入口仍保持 `[NOT IMPLEMENTED]`
和状态码 2。

下一步：执行 M6，让 Stateful Collector 独占 `[E,N,N]` 的 `z_dense` 生命周期，
完成 global-ID gather/scatter、每步恰好一次更新、整环境/单 agent reset，以及
neighbor 消失时的自然衰减。

---

## 23. 可复制给新 Session 的提示词

```text
请继续实施 sigmarl-nod 中的 Opinion Dynamics + MARL。

唯一工作目录：
/Users/zhangxiaotong/Code/sigmarl-nod

不要进行任何 Git/版本控制操作；用户会手动处理版本。
直接修改原目录中的代码和文档，使用现有 .venv。

开始前完整阅读：
1. OPINION_MARL_IMPLEMENTATION_GUIDE.md
2. opinion_dynamics_marl_technical_route.md

方法定位：这是独立的 Opinion Dynamics + MARL，不是 TSC 扩展。
TSC 只能作为工程参考和外部基线，Opinion 路径禁止依赖 priority、leader、
Stackelberg、topology labels、topology learner、action predictor 或 opponent modeling。

先查看第 5 节里程碑状态和第 22 节验证记录。
R0/M0、R1/M1、M2、M3、M4 和 M5 已完成。当前下一步是 M6：实现 Stateful
Opinion Collector。M5 已提供单步 OpinionAugmentedPolicy，但不拥有跨步状态；
M6 必须成为 `z_dense [E,N,N]` 的唯一拥有者，按 global neighbor ID gather/scatter，
每个物理步只更新一次，正确处理 environment done、single-agent reset、neighbor
换槽/消失/重现，并让非候选旧边自然衰减。禁止把 z 放进 environment callback。

每一步必须先测试、再最小实现、运行全部回归、更新本指南验证记录，
并报告修改文件、测试命令、实际结果和遗留问题。
```
