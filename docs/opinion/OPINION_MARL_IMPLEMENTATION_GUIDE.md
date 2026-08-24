# Opinion Dynamics + MARL：SigmaRL 1.2.0 重建指南

> 文档状态：R0、R1、M2、M3 实现已完成，训练与性能由用户手动验证  
> 唯一代码底座：SigmaRL tag `1.2.0`  
> 基线 commit：`5fe715bdfba4ff3e33d901d69dfa220f1222c060`  
> 理论真源：[`opinion_dynamics_marl_technical_route.md`](opinion_dynamics_marl_technical_route.md)  
> 对齐日期：2026-08-24

## 0. 新 Session 先读这里

本项目已将根目录源码恢复为 SigmaRL 1.2.0，将在此基础上重新实现独立的
Opinion Dynamics + MARL。旧 TSC 代码不作为载体，也不恢复旧 Opinion 实现。

新 Session 必须先完成：

1. 阅读本文件；
2. 阅读技术路线全文；
3. 阅读 [`M2_CONFIG_AND_ENTRYPOINTS.md`](M2_CONFIG_AND_ENTRYPOINTS.md)；
4. 阅读 [`M3_MATH_MODULES.md`](M3_MATH_MODULES.md)；
5. 阅读 `docs/sigmarl_1_2_0/` 下的核对记录、R1 使用说明和三份事实文档；
6. 确认当前代码以 tag 1.2.0 为底座，且只包含本表已经完成的阶段修改；
7. 查看本文件第 10 节，只执行下一个未完成阶段；
8. 每次实现后更新阶段状态、验证命令和真实结果。

可复制给新 Session：

```text
请在以 SigmaRL 1.2.0 为底座、已完成 R0/R1 的代码上继续实现独立
Opinion Dynamics + MARL。

先完整阅读：
1. docs/opinion/opinion_dynamics_marl_technical_route.md
2. docs/opinion/OPINION_MARL_IMPLEMENTATION_GUIDE.md
3. docs/opinion/M2_CONFIG_AND_ENTRYPOINTS.md
4. docs/opinion/M3_MATH_MODULES.md
5. docs/sigmarl_1_2_0/CODEBASE_AUDIT.md
6. docs/sigmarl_1_2_0/ 下的环境、观测和网络说明

不要恢复 docs/archive_tsc 中的代码设计。TSC 只作为外部实验基线；新方法禁止依赖
TopologyLearner、priority、leader、Stackelberg、action predictor 或 opponent
modeling。Base 必须走 SigmaRL 1.2.0 原始向量化 MAPPO；Evidence/Joint 才使用
连续 chunk。先检查里程碑状态，只实现下一个阶段；保证训练/测试入口完整，实际训练
和性能判断由用户手动完成。
```

### 0.1 当前解释器约定

当前使用 Conda 环境 `sigmarl-nod`，不存在项目内 `.venv/`。激活环境后直接
使用 `python`；未激活时使用：

```bash
conda run -n sigmarl-nod python <command>
```

禁止在新文档、测试或脚本中写死 `.venv/bin/python`。R0 已在现有环境中设置
user-site 隔离并补齐固定依赖；详见
[`../sigmarl_1_2_0/CODEBASE_AUDIT.md`](../sigmarl_1_2_0/CODEBASE_AUDIT.md)。

后续继续使用现有 `sigmarl-nod` Conda 环境，不新建其他环境、不改环境名，也不
切换回以前的 `.venv`。实际训练和性能验证由用户手动执行。

## 1. 方法和代码底座的关系

```text
SigmaRL 1.2.0
├── VMAS 道路环境
├── observation/action/reward/reset
├── 共享分散 Actor
├── 中心化 Critic
└── 向量化 MAPPO Base 训练

Opinion-MARL 新增
├── ConflictGraph
├── EvidenceNet
├── Fixed OpinionDynamics
├── Stateful Collector
├── bounded residual
└── Evidence/Joint Sequence PPO

TSC
└── 外部比较仓库和实验基线，不是运行时依赖
```

### 1.1 不得引入的 TSC 概念

Opinion 主路径禁止调用或重新包装：

- topology learner/labels/loss；
- priority score、priority policy 或 priority critic；
- leader/follower、leader set、total order；
- Stackelberg-conditioned Critic；
- topology action predictor；
- opponent modeling；
- priority Top-K 或 action propagation。

候选车辆筛选只表示“当前可能发生物理冲突的车辆对”，不表示通行权或领导关系。

## 2. 1.2.0 不变量

加入 Opinion 前后，下列合同必须保持：

| 合同 | 1.2.0 默认值 |
|---|---|
| 场景 | `road_traffic / CPM_mixed` |
| agents | 4 |
| observation | `[E,4,32]` |
| action | `[E,4,2]` |
| Actor | shared decentralized MLP，2×256 Tanh |
| Critic | shared centralized MLP，2×256 Tanh |
| episode | 128 steps |
| rollout | 4096 team frames |
| minibatch | 512 |
| PPO epochs | 60 |

Opinion 关闭时必须满足：

- 原始 observation 数值和 shape 不变；
- 原始 action spec 不变；
- reward 和 done/reset 不变；
- 原始 Base checkpoint 可训练和测试；
- 不创建任何 Opinion 状态或附加网络。

## 3. 核心张量合同

记：

```text
E = 并行环境数
N = 4 agents
K = 每个 ego 的 Opinion 候选数（第一版固定为 2）
F = pair feature 维度（第一版固定为 10）
A = 2 actions
```

### 3.1 环境输出

只在 Opinion 开启时额外输出：

```text
pair_features     float [E,N,K,F]
neighbor_ids      long  [E,N,K]
pair_mask         bool  [E,N,K]
urgency           float [E,N,K]
confidence        float [E,N,K]
agent_reset_mask  bool  [E,N]
```

环境不输出 `b` 或 `z`。

第一版的 `K=2`，与原始 `n_nearing_agents_observed=2` 保持一致。
`neighbor_ids` 必须复用环境当前最近邻选择对应的 global agent ID，不得额外使用
第三个未被原始局部观测覆盖的车辆。如果以后将 `K` 扩展为 3，必须作为
“感知范围变更”的独立实验，不能与原始 Base 直接宣称公平。

第一版 ConflictGraph 仅使用当前可观测物理量和短时常速运动学外推，计算最近接近
时间和最小间距。不读取未来真值轨迹、其他车辆真实未来动作或 TSC action
predictor。

第一版有向车辆对特征固定为 10 维，全部在 ego 坐标系中归一化：

```text
relative_position_x, relative_position_y          2
relative_velocity_x, relative_velocity_y          2
ego_speed, neighbor_speed                         2
sin(relative_yaw), cos(relative_yaw)               2
time_to_closest_approach                           1
distance_at_closest_approach                       1
----------------------------------------------------
F                                                   10
```

常速最近接近量使用：

```text
r = p_neighbor - p_ego
u = v_neighbor - v_ego
t_cpa = clamp(-dot(r,u) / (dot(u,u)+eps), 0, horizon)
d_cpa = norm(r + t_cpa*u)
```

`pair_mask` 由最近邻有效性、预测时间窗和安全间距阈值共同决定；`urgency` 必须随
`t_cpa` 与 `d_cpa` 单调减小，`confidence` 第一版只由当前可见性和感知距离确定。
具体阈值全部进入 typed Opinion 配置，禁止散落在环境代码中。后续若改为参考路径
交点或学习型轨迹预测，必须新建版本/消融，不能静默改变这 10 维合同。

### 3.2 Opinion 状态

Collector 按 global agent ID 持有：

```text
z_dense [E,N,N]
```

`z_dense[:,i,j]` 表示车辆 `i` 相对车辆 `j` 的有向意见；对角线始终为零。

### 3.3 Policy 输出

每步至少返回：

```text
base_loc, scale, raw_b, b, z_prev, z_next,
q, normalized_weights, aggregate, residual,
final_loc, action, log_prob
```

所有 rollout 存储张量必须 detach；PPO 重算时从 chunk 起点状态重新展开计算图。

第一版 Evidence/Joint Critic 继续只读取 SigmaRL 原始联合 observation，不把
`z` 追加到 Critic 输入。这样可以保持原始 Critic 结构和 checkpoint 边界，也天然阻断
Critic loss 对 EvidenceNet 的梯度。使用 `stop_gradient(z)` 的 Opinion-aware
Critic 只能作为后续消融。

## 4. 每个物理步的唯一因果顺序

```text
1. 获取 observation 和当前物理 pair info
2. 根据 reset mask 清理 z_dense
3. 用 neighbor_ids gather z_prev
4. EvidenceNet(pair_features) → raw_b
5. urgency × confidence 物理门控 → b
6. Fixed OpinionDynamics(z_prev,b) → z_next
7. OpinionResidual(z_next, urgency, mask) → residual
8. Base Actor observation → base loc/scale
9. 只修正速度 loc，转向 loc 和 scale 不变
10. TanhNormal 采样 action/log_prob
11. 将 z_next scatter 回 z_dense
12. env.step(action)
```

每个环境步只能积分一次 `z`。环境、render 和 PPO loss 不得偷偷再次更新状态。

## 5. 模块边界

按阶段新增：

```text
utilities/experiment_artifacts.py     # R1 已新增
configs/base/pilot.json               # R1 已新增
scripts/check_runtime_environment.py
scripts/run_milestone_validation.py
main_training_opinion.py
main_testing_opinion.py
configs/validation/smoke.json
configs/validation/pilot.json
config_opinion.json
configs/opinion/pilot.json

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
├── diagnostics.py
├── evaluation.py
└── artifacts.py

tests/opinion/
└── 与模块一一对应的测试

tests/test_runtime_environment.py
tests/test_base_entrypoint.py
```

原始文件允许的最小修改：

| 原始文件 | 允许修改 |
|---|---|
| `main_training.py` | R1 独立 Base run 包装，保持标准无参数入口 |
| `main_testing.py` | R1 定位最近成功 Base run，保持标准无参数入口 |
| `utilities/mappo_cavs.py` | R1 种子、指标、计时和 Base 产物钩子；不得改变 PPO 数据流 |
| `utilities/helper_training.py` | R1 seed 与 artifact metadata |
| `config.json` | Base seed 和稳定输出根目录 |
| `scenarios/road_traffic.py` | gated pair info、reset event、detached render |
| `requirements.txt` | 可视化确需的固定依赖 |

Opinion 实现不得为了迁就新方法修改：

```text
utilities/helper_scenario.py
utilities/constants.py
```

R1 对原始训练文件的改动只服务于 Base 产物合同。M2 已选择更严格的并列配置边界：
Opinion loader 用被引用的 Base JSON 构造原始 `Parameters`，同时单独返回 typed
`OpinionExperimentConfig`。不把 Opinion/TSC 字段塞回 `Parameters`，也不把 Opinion
分支塞入原始 Base PPO 主循环。关闭 Opinion 时，独立入口复用同一个 `train_base()`。

## 6. Base 训练必须使用原始快速路径

Base 阶段直接复用 1.2.0：

```text
rollout → GAE → reshape(-1) → 512-frame minibatch → ClipPPOLoss
```

Base 阶段禁止：

- 构造 ConflictGraph；
- 运行 EvidenceNet；
- 分配 `z_dense`；
- 使用 SequenceBuffer；
- 按 chunk/time step 调用 loss。

Base Actor 最好直接复用 1.2.0 的 `MultiAgentMLP + NormalParamExtractor`，使
`final_policy.pth` 可无歧义地进入 Evidence 阶段。

Base checkpoint 合同与 Opinion checkpoint 分离：

```text
base checkpoint:
  base_actor_state
  critic_state
  resolved_config
  iteration

opinion checkpoint:
  base_actor_state
  evidence_state
  critic_state
  stage
  optimizer_states
  resolved_config
  schema_version
```

修改 EvidenceNet 结构时，不应迫使用户重训完全相同的 Base。

## 7. Evidence/Joint 的高效 Sequence PPO

时间依赖要求保留连续 chunk，但不意味着逐 chunk 串行：

```text
错误实现：
for chunk in minibatch:
    for t in chunk:
        policy(one chunk, one time)

目标实现：
stack all chunks on batch dimension
for t in chunk_length:
    policy(all chunks at time t)
```

即：

- chunk batch 维并行；
- 只在时间维循环；
- chunk 起点保存 `z_init`；
- 不跨 done/environment 边界；
- log-prob 必须由相同最终分布重算；
- old rollout tensors 全部 detach；
- Critic loss 不得反向训练 EvidenceNet；
- 固定动力学参数不进入 optimizer。

## 8. 三阶段优化合同

| 阶段 | Actor | EvidenceNet | Critic | Dynamics | residual |
|---|---|---|---|---|---|
| Base | 训练 | 不构造 | 训练 | 不构造 | 0 |
| Evidence | 冻结 | 训练 | 训练 | 固定 | warm-up |
| Joint | 训练 | 训练 | 训练 | 固定 | warm-up/target |

“Evidence 阶段冻结 Base”不代表最终模型永久冻结。Joint 阶段会解冻 Actor 共同微调。
永久冻结版本只能作为单独消融。

## 9. 训练产物合同

每个 stage 使用独立目录：

```text
<run>/<stage>/
├── config_source.json
├── config_resolved.json
├── metrics.json
├── training_curves.pdf
├── training_status.json
├── latest_checkpoint.pt
├── final_checkpoint.pt
├── final_base_actor.pth
├── final_evidence_net.pth      # Base 阶段无此文件
├── final_critic.pth
└── artifacts_manifest.json
```

每轮原子更新 metrics/status；中断后至少能判断最后完整 iteration。PDF 至少包含：

- reward；
- agent/lane/total collision；
- PPO loss；
- Actor/Evidence/Critic gradient norm；
- `raw_b/b/z/residual`；
- mask、reset 和饱和率。

每个里程碑的验证训练还必须保存：

```text
validation_protocol.json   # 预算、seed、场景和比较 checkpoint
timing.json                # rollout、优化和总 wall time
comparison_to_base.json    # 与 R1 基线的同预算差值
```

必须区分：

- stage 初始化：从上阶段权重开始新的 optimizer；
- exact resume：恢复 optimizer、iteration、随机状态和调度器。

若尚未实现 exact resume，CLI 和文档必须明确说明。

## 10. 重建里程碑

状态只使用：`未开始 / 进行中 / 已完成 / 阻塞`。

| 阶段 | 状态 | Gate |
|---|---|---|
| R0：恢复 SigmaRL 1.2.0 原始代码 | 已完成 | 环境原地隔离，标准训练/测试入口已统一 |
| R1：原始 Base 速度与产物基线 | 已完成 | 固定小预算与 wall time；用户手动训练 |
| M2：Opinion 配置与独立入口 | 已完成 | 不导入 TSC，不改变 Base；用户手动训练 |
| M3：Evidence/Dynamics/Residual | 已完成 | 数学、边界、梯度测试；用户手动运行 |
| M4：ConflictGraph 环境接口 | 未开始 | gated info/reset，Base 不变 |
| M5：Policy 与 Base checkpoint bridge | 未开始 | 分布、边界和权重一致 |
| M6：Stateful Collector | 未开始 | global ID、每步一次、reset |
| M7：Sequence Buffer | 未开始 | 不跨 env/done，保存 z_init |
| M8：批量 chunk Sequence PPO | 未开始 | log-prob/梯度正确且性能达标 |
| M9：三阶段 Trainer/Checkpoint | 未开始 | Base→Evidence→Joint smoke |
| M10：评估、诊断、PDF、可视化 | 未开始 | 无参数更新、产物完整 |
| M11：消融与正式实验 | 未开始 | 多 seed、公平预算、统计报告 |

### R0 Gate

当前证据见 [`../sigmarl_1_2_0/CODEBASE_AUDIT.md`](../sigmarl_1_2_0/CODEBASE_AUDIT.md)。

重置后先验证：

```text
main_training.py                 与 tag 1.2.0 一致
utilities/mappo_cavs.py          与 tag 1.2.0 一致
utilities/helper_training.py     与 tag 1.2.0 一致
scenarios/road_traffic.py        与 tag 1.2.0 一致
config.json                      与 tag 1.2.0 一致
```

真实环境 reset 必须得到：

```text
observation [E,4,32]
action      [E,4,2]
```

此外必须在不依赖 user-site 的情况下通过核心依赖版本、`pip check`、
`TanhNormal` 有限值和 3-step rollout 检查。

### R1 Gate

R1 用法和产物定义见
[`../sigmarl_1_2_0/R1_BASE_ARTIFACTS.md`](../sigmarl_1_2_0/R1_BASE_ARTIFACTS.md)。
实现已完成；下列数据需由用户运行完整或 pilot 训练后填充：

用固定 tiny/pilot 配置记录：

- 每 iteration 采样时间；
- PPO 优化时间；
- 总 wall time；
- reward/collision；
- policy/critic `.pth` 可加载。

这个数据是后续所有性能优化的比较基准。

### 10.1 每个里程碑都必须保持可训练

每一步实现后都必须保持完整训练和测试入口。开发 Session 负责实现代码、配置、保存
和加载闭环，但不替用户启动训练或判断性能；以下分层验证由用户按需手动执行：

```text
Level A：smoke training
  极小固定预算，验证反向传播、优化、保存和加载均可运行

Level B：pilot comparison
  使用 R1 固定的相同 seed、场景、采样帧数、PPO epochs 和评估预算
  输出与 Base 的 reward、collision、速度和 wall-time 差值

Level C：formal comparison
  多 seed 完整预算，只在 M9/M11 执行，用于判断真实性能提升
```

具体 Gate：

| 阶段 | 每步训练方式 | 与 Base 的预期 | 通过条件 |
|---|---|---|---|
| R0 | 原始 Base tiny smoke | 行为不应改变 | shape/有限值/优化一步/保存加载通过 |
| R1 | 原始 Base pilot | 建立参照 | 固化曲线、碰撞、速度和耗时区间 |
| M2 | `use_opinion_marl=false` 训练 | 应与 Base 等价 | 同 seed 动作/log-prob 合同一致，指标落入 R1 区间 |
| M3 | 数学模块测试 + 模块未接线的 Base 训练 | 应与 Base 等价 | 新模块可反传，但关闭后不影响 Base |
| M4 | pair info 开启但不被 Policy 使用 | 应与 Base 近似一致 | 原始 observation/reward/action 不变，开销被记录 |
| M5 | Direct evidence residual pilot | 开始出现协调信号 | 数值稳定、无碰撞灾难性回退，并与同信息预算 Direct 基线比较 |
| M6 | Stateful rollout + 冻结 Opinion pilot | 不要求立即提升 | global-ID/reset 正确，连续运行无状态串扰 |
| M7 | Sequence Buffer + Base/no-op loss pilot | 应接近对应输入模型 | chunk 不跨 done/env，吞吐和内存可接受 |
| M8 | Evidence Sequence PPO pilot | 期望出现正向趋势 | Evidence 有有效梯度，碰撞/回报至少一项改善且另一项不明显恶化 |
| M9 | Base→Evidence→Joint pilot/正式训练 | 期望优于 Base | 三阶段稳定，至少 3 seeds 出现一致方向信号 |
| M10 | 重跑训练回归 + checkpoint 评估 | 不应改变训练性能 | 评估/可视化不更新参数，指标可复现 |
| M11 | 多 seed 正式训练与消融 | 需要统计提升 | 按预注册指标和置信区间报告结论 |

所有 Level B 比较写入独立目录：

```text
outputs/milestone_validation/<milestone>/<run_id>/
```

禁止不同里程碑复用或覆盖同一个输出目录。

### 10.2 如何解释“和 Base 差不多”与“性能提升”

对于 M2、M3 和 M4 这类尚未改变最终动作分布的步骤，正确目标不是提升，而是证明
没有行为回归：关闭/no-op 路径应通过确定性张量合同，训练指标应落入 R1 测得的
随机波动区间。

对于 M5 之后真正改变动作或时间状态的步骤，单次 pilot 只能作为方向性诊断，不能
作为“优于 Base”的论文结论。阶段性判断优先级为：

1. 碰撞率不得灾难性升高；
2. reward、通行速度或完成率至少一项出现改善；
3. 训练数值稳定、梯度有限、wall time 增量可解释；
4. 最终提升必须由 M9/M11 的多 seed 同预算实验确认。

若某个核心阶段没有出现正向趋势，应停在该阶段检查特征、门控、动力学参数和梯度，
而不是继续堆叠后续模块或修改 Base reward 来制造提升。

## 11. 必须保留的测试原则

### 数学测试

- `b` 有界且受 urgency/confidence 门控；
- `z=0` 附近稳定性和临界紧迫度符合理论；
- 证据符号翻转能驱动意见翻转；
- `pair_mask=false` 时不接收 evidence/自强化，已有意见只执行确定性衰减；
- residual 有界且不随 K 无限制增大。

### 状态测试

- global ID 与候选槽位交换解耦；
- done 清空单环境；
- partial reset 清空关联行列；
- 每物理步只积分一次；
- evaluation 不更新参数。

### PPO 测试

- rollout action 在不更新参数时重算 log-prob 一致；
- Actor loss 能训练 EvidenceNet；
- Critic loss 不能训练 EvidenceNet；
- 早期 evidence 梯度能经后续 `z` 回传；
- 固定 dynamics 不在 optimizer；
- 批量 chunk 与逐 chunk 参考实现在小张量上数值一致。

### 回归测试

- Opinion 关闭时 Base observation/action/reward/done 不变；
- Base 快速路径不调用 Opinion 模块；
- checkpoint stage/config 不匹配时在改写参数前失败；
- 所有指标有限；
- 产物目录不可被旧 run 冒充。

## 12. 正式实验边界

至少比较：

```text
SigmaRL 1.2.0 Base-MAPPO
TSC（外部仓库）
Direct evidence residual
EMA evidence
Linear dynamics
GRU memory
Handcrafted evidence + fixed dynamics
Learned evidence + fixed nonlinear dynamics（Full）
```

实验必须覆盖多 seed、训练场景与未见场景、噪声、延迟和短时遮挡。Pilot/smoke 只
证明实现闭环，不能用于宣称新方法优于 Base 或 TSC。

## 13. 当前状态说明

截至 2026-08-24：

- 文档已统一到 SigmaRL 1.2.0；
- 旧 TSC 文档已移入 `docs/archive_tsc/`；
- 本表不继承旧仓库的 M0-M10“已完成”状态；
- 核心源码、配置和资源已逐文件确认与 commit `5fe715b...` 一致；
- `CPM_mixed` 真实 reset 和 3-step rollout 已通过；
- R0 已在现有 `sigmarl-nod` 环境中设置 user-site 隔离并补齐固定依赖；
- R1 已保持原始向量化 MAPPO 主循环，并增加唯一 run、配置快照、逐轮指标/耗时、
  曲线 PDF、Base Actor/Critic 和完整最终 checkpoint；
- `main_testing.py` 已通过 `latest_run.json` 与最近成功 Base run 及训练场景对齐；
- M2 已新增 strict typed Opinion schema、完整/pilot 配置和独立训练/测试入口；
- M2 的 `use_opinion_marl=false` 入口直接复用 R1 Base 路径，开启未实现阶段会明确
  失败，不会静默伪装为 Opinion 训练；
- M3 已实现反对称有界 EvidenceNet、无可训练参数的固定 OpinionDynamics、归一化
  有界速度 Residual 以及标准库参考测试；模块尚未接入环境或 Actor；
- 按用户要求，实际训练、测试和性能判断由用户手动完成；
- 下一实现步骤是 M4：在 `road_traffic.py` 中增加 gated ConflictGraph、10 维车辆对
  特征和 reset 信息，但仍不改变 Base observation/action/reward。
