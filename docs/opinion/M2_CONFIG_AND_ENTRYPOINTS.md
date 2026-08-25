# M2：Opinion 配置与独立训练/测试入口

> 实现状态：已完成；后续状态见 M3 文档  
> 当前可执行阶段：`stage=base`、`use_opinion_marl=false`  
> 当前算法行为：SigmaRL 1.2.0 Base-MAPPO

## 1. M2 的目的

M2 建立新方法自己的配置和命令入口，但不提前实现意见数学模块。它解决三个工程
问题：

1. Base 参数继续由 SigmaRL `Parameters` 管理，不复制或污染原配置；
2. Opinion 参数由严格的 typed loader 单独管理；
3. 从现在开始，新方法固定使用 `main_training_opinion.py` 和
   `main_testing_opinion.py`，后续 M3–M10 逐步接入真实模块。

M2 不修改 observation、action、reward、Actor、Critic、collector 或 PPO。

## 2. 配置结构

完整配置为根目录 `config_opinion.json`，pilot 配置为
`configs/opinion/pilot.json`。顶层结构：

```json
{
    "schema_version": 1,
    "method": "opinion_marl",
    "stage": "base",
    "use_opinion_marl": false,
    "base_config": "config.json",
    "output_root": "outputs/opinion/base/",
    "opinion": {}
}
```

- `base_config`：相对于当前 Opinion 配置文件的位置解析；
- `output_root`：与原项目一致，按项目根目录运行时解析；
- `stage`：最终支持 `base / evidence / joint`；
- `use_opinion_marl`：`base` 必须为 `false`，其余两阶段必须为 `true`；
- `opinion`：只保存新方法参数，不允许未知字段。

Base 配置和 Opinion 配置作为两个 typed 对象并列传递。Opinion 字段不会写入原始
`Parameters`，因此关闭路径不会构造 Opinion 状态或改变 Base PPO。

## 3. 已冻结的第一版合同

### ConflictGraph

```text
emit_pair_info           = false（M4 pilot 单独设为 true）
candidate_count          = 2
pair_feature_dim         = 10
prediction_horizon       = 3.0 s
conflict_distance        = 2.0 m
sensing_distance         = 20.0 m
```

`candidate_count=2` 必须与 Base 的 `n_nearing_agents_observed=2` 一致，第一版不扩大
感知信息预算。

### Evidence 与固定动力学

```text
Evidence hidden sizes    = [128, 128]
b_max                    = 1.0
temperature              = 1.0
eta_z / response_rate    = 0.5
kappa_z / decay_rate     = 1.0
nu_z / reinforcement     = 0.5
alpha_z / sensitivity    = 1.0
```

这些值是第一版的显式初始配置，M3 会实现数学模块并检查有界性和稳定性。动力学参数
保持固定，不进入 optimizer。

### Residual 与 Sequence PPO

```text
z0 / opinion_scale       = 1.0
residual gain            = 0.1
residual absolute limit  = 0.25
action index             = 0（速度）
chunk length             = 16
Evidence learning rate   = 0.1 × Actor learning rate
```

第一版只允许修改速度分量 `action[0]`，不允许意见直接控制转向 `action[1]`。

### M5 Policy Bridge 扩展字段

从 M5 起 strict schema 还要求 `opinion.policy_bridge`。M2–M4 配置固定
`enabled=false`；M5 的 `evidence` 配置固定 `enabled=true`、
`mode=direct_evidence` 和 `freeze_base_actor=true`。完整语义见
[`M5_POLICY_BRIDGE.md`](M5_POLICY_BRIDGE.md)。

从 M6 起 strict schema 还要求 `opinion.stateful`。M2–M5 配置固定为 disabled、
`evidence_output_root=null`；M6 配置固定 `enabled=true`、指定 M5 输出根目录并冻结
EvidenceNet。完整语义见 [`M6_STATEFUL_OPINION.md`](M6_STATEFUL_OPINION.md)。

从 M7 起 `opinion.sequence_ppo.enabled=true`，并要求独立的
`source_output_root` 指向 M6 输出根目录；M2–M6 固定为 `enabled=false`、
`source_output_root=null`。M7 只启用 Sequence Buffer，Evidence 时间梯度仍留给
M8。完整语义见 [`M7_SEQUENCE_BUFFER.md`](M7_SEQUENCE_BUFFER.md)。

M8 增加 `sequence_ppo.train_evidence=true` 并开放 Evidence 时间梯度。M9 strict
schema 进一步要求 `opinion.trainer`：旧阶段固定 `enabled=false`，M9 `joint` stage
支持从零完整 Joint、Evidence-only、Base/Opinion 初始化 Joint 和 Warmup→Joint。完整语义见
[`M8_SEQUENCE_PPO.md`](M8_SEQUENCE_PPO.md) 与
[`M9_TRAINER_AND_CHECKPOINT.md`](M9_TRAINER_AND_CHECKPOINT.md)。

## 4. 配置启动前检查

`utilities/opinion/config.py` 会在创建环境或输出目录之前拒绝：

- 缺失字段或未知字段；
- 字符串冒充 bool/numeric 等类型错误；
- 非有限数、非正尺度或非法 residual 边界；
- 非 `road_traffic / CPM_mixed / 4 agents`；
- 候选数与 Base 最近邻数量不一致；
- TSC/opponent modeling、priority MARL 或 prioritized replay；
- 训练配置中的 load/continue/testing 模式；
- batch、episode、minibatch 或 chunk 无法整除；
- `stage` 和 `use_opinion_marl` 不一致。

历史 TSC 的 topology、leader、priority、Stackelberg 或 action predictor 字段不在
schema 中，因此会作为未知字段直接拒绝。

## 5. M2 pilot 训练与测试

```bash
conda activate sigmarl-nod
python main_training_opinion.py --config configs/opinion/pilot.json
python main_testing_opinion.py --config configs/opinion/pilot.json
```

pilot 输出：

```text
outputs/opinion_pilot/base/runs/opinion-off-base-seed<seed>-<id>/
```

它应产生与 R1 相同的 Base 模型、指标、曲线和 checkpoint，额外保存：

```text
base_config_source.json
opinion_config_resolved.json
```

因为最终动作分布尚未改变，M2 不应声称性能提升；合理预期是相同 seed 和预算下与
R1 Base 落在相同范围。

## 6. 完整训练与测试

```bash
python main_training_opinion.py
python main_testing_opinion.py
```

测试指定历史 run：

```bash
python main_testing_opinion.py \
  --run-dir outputs/opinion/base/runs/<run_id>
```

标准 Base 命令仍然保留且不受影响：

```bash
python main_training.py
python main_testing.py
```

## 7. 防止静默错误

M2 如果配置为：

```json
{
    "stage": "evidence",
    "use_opinion_marl": true
}
```

入口会明确报错，说明 Opinion 执行模块将在后续阶段实现。它不会生成一个名字叫
Opinion、实际却仍为 Base 的训练结果。

## 8. M2 产物与下一阶段

M2 run 仍使用 R1 的产物合同：配置快照、逐轮 metrics、timing、PDF 曲线、Actor、
Critic 和完整最终 checkpoint。`config_resolved.json` 只包含实际送入 Base 的
`Parameters`；`opinion_config_resolved.json` 单独记录尚未启用的 Opinion 合同。

M3 已新增纯数学模块：

- `OpinionEvidenceNet`；
- 固定 `OpinionDynamics`；
- 有界 `OpinionResidual`。

M3 仍以“数学模块已存在但尚未接线”的 no-op 方式保持训练入口可运行，不接管
Stateful Collector。具体见
[`M3_MATH_MODULES.md`](M3_MATH_MODULES.md)。
