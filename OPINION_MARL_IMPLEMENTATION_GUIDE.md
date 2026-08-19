# Opinion Dynamics + MARL：实施方案与跨 Session 交接指南

> 最后核对日期：2026-08-19  
> 当前仓库：`/Users/zhangxiaotong/Code/sigmarl-nod`  
> 当前阶段：R0/M0、R1/M1、M2 已完成；下一步执行 M3 纯数学模块  
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
R0/M0、R1/M1 和 M2：

- 原始 `config.json`、`requirements.txt`、`main_training.py`、`main_testing.py`
  仍保持兼容；
- `utilities/mappo_cavs.py` 已加入 Base/TSC 门控；
- `utilities/helper_training.py` 已加入默认关闭的 Opinion 配置字段；
- `scenarios/road_traffic.py` 尚未为 Opinion 修改；
- 理论路线文件已经存在且内容完整；
- M0 runtime checker、M1 基线和 M2 Opinion 配置入口均已通过验证；
- 本文件是新仓库重新建立的工程实施真源。

当前环境已经验证：

```text
Python       3.9.13
torch        2.1.0
torchrl      0.2.1
tensordict   0.2.1
vmas         1.4.1
现有测试      136 passed
pip check    无依赖冲突
```

当前恢复顺序为：

```text
R0：恢复并验证 M0 运行时检查（已完成）
→ R1：恢复并验证 M1 Base-MAPPO/TSC 基线（已完成）
→ M2：Opinion 配置与独立入口（已完成）
→ M3：Evidence、Dynamics、Residual 纯数学模块（当前下一步）
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
| M3 数学模块 | 未开始 | Evidence、Dynamics、Residual 纯张量实现 |
| M4 ConflictGraph | 未开始 | 当前物理量到 pair data 的环境接口 |
| M5 Opinion Policy | 未开始 | Base Actor + Opinion residual + TanhNormal |
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

当前完整工程预期分别为 `136 passed` 和 `No broken requirements found.`。

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

当前预期为 `136 passed`。

### 8.7 完成边界

M2 已完成配置、解析、隔离验证和 CLI 骨架，但仍未实现：

```text
EvidenceNet
OpinionDynamics
OpinionResidual
ConflictGraph
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

---

## 10. M4：ConflictGraph 环境接口

新增：

```text
utilities/opinion/conflict_graph.py
```

修改：

```text
scenarios/road_traffic.py
```

环境只负责输出当前物理量和固定几何计算：

- pair features；
- global neighbor ID；
- pair mask；
- urgency；
- confidence；
- agent reset mask。

环境禁止：

- 保存或更新 `z`；
- 调用 EvidenceNet；
- 使用未来真实轨迹生成执行期输入；
- 依赖 TSC topology selection 开关。

测试场景：迎面接近、平行同速、背离、交叉接近、静止邻车、不可见邻车、单 agent reset。

完成后达到 Gate B。

---

## 11. M5：OpinionAugmentedPolicy

新增：

```text
utilities/opinion/policy.py
```

模块：

```text
BaseGaussianActor
OpinionEvidenceNet
OpinionDynamics
OpinionResidual
OpinionAugmentedPolicyCore
ProbabilisticActor/TanhNormal wrapper
```

关键测试：

- Base stage residual 严格为零；
- sample/log-prob 有限；
- residual 只改变速度 loc；
- 转向 loc 和 scale 不变；
- `z_next` 有限；
- invalid mask 不改变意见；
- 同一输入和状态可重复计算相同分布参数。

完成后达到 Gate C。

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
- [ ] EvidenceNet 输出有界且不读取 `z/q`；
- [ ] Dynamics 固定且无梯度；
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
R0/M0、R1/M1 和 M2 已完成。当前下一步是 M3：实现 EvidenceNet、固定
OpinionDynamics 和 OpinionResidual 三个纯张量数学模块；本阶段禁止接入
环境、collector 或 PPO。

每一步必须先测试、再最小实现、运行全部回归、更新本指南验证记录，
并报告修改文件、测试命令、实际结果和遗留问题。
```
