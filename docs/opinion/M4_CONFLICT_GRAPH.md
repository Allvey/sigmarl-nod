# M4：ConflictGraph 与环境 pair-info 接口

> 实现状态：已完成；训练和性能由用户手动验证  
> 本阶段是否改变动作：否  
> 下一阶段：M5 Policy Bridge

## 1. M4 做了什么

M4 从每个物理时刻的车辆位置、速度和航向构造有向冲突候选图，并把结果放入 VMAS
环境 `info`。它只提供后续意见模块所需的信息，不把这些数据拼入原始 observation，
也不运行 EvidenceNet、OpinionDynamics 或 residual。

启用后新增的逻辑链为：

```text
SigmaRL 原最近邻选择
        │
        ├── neighbor_ids [E,N,K]
        │
当前车辆状态 ──> 常速 CPA 外推 ──> pair_mask / urgency / confidence
        │
        └── ego 坐标系归一化 ──> pair_features [E,N,K,10]

车辆/环境重置 ──> agent_reset_mask
```

这些张量不进入 M4 的 Actor，因此：

```text
原 observation 不变
原 action space 不变
原 reward 不变
Base Actor/Critic/PPO 不变
```

## 2. 为什么保留 neighbor ID

`neighbor_ids` 不是让自车学习“第几号车”，也不是 Actor 输入。它只是后续 M6 的
状态表索引：同一邻车在连续帧中的距离排序可能由第 1 位变成第 2 位，Opinion 状态
必须跟随真实车辆，而不能跟随候选槽位。

第一版直接复用 `road_traffic.py` 已有的
`observations.nearing_agents_indices`。因此 M4 不额外观察第三辆车，也不扩大 Base 的
感知范围。全局 ID 只在 collector 内用于 gather/scatter。

## 3. 固定张量合同

`ScenarioRoadTraffic.info(agent)` 在 M4 开启时额外返回当前 agent 的切片；VMAS 汇总
后对应：

```text
pair_features     float [E,N,K,10]
neighbor_ids      long  [E,N,K]
pair_mask         bool  [E,N,K]
urgency           float [E,N,K]
confidence        float [E,N,K]
agent_reset_mask  bool  [E,N,1]  # VMAS 给标量 info 增加末尾 info 维
```

在 Scenario 的单-agent `info()` 返回值中，`agent_reset_mask` 是 `[E]`；后续 collector
读取时应接受 VMAS 的 `[E,N,1]` 并 `squeeze(-1)` 为理论合同 `[E,N]`。当前
TorchRL 0.2.1 会先把 VMAS 的所有 info leaf 转成 `float32`，所以 M4 在
`TransformedEnvCustom` 的输出侧仅对 `neighbor_ids` 和两个 mask 恢复 `long/bool`
dtype；这组 transform 只在 `emit_pair_info=true` 时构造。

10 维 pair feature 的顺序固定为：

```text
0: relative_position_x / sensing_distance
1: relative_position_y / sensing_distance
2: relative_velocity_x / (2 * max_speed)
3: relative_velocity_y / (2 * max_speed)
4: ego_speed / max_speed
5: neighbor_speed / max_speed
6: sin(relative_yaw)
7: cos(relative_yaw)
8: t_cpa / prediction_horizon
9: d_cpa / sensing_distance
```

相对位置和相对速度先旋转到 ego 坐标系，所有数值被限制到有限的 `[-1,1]` 或
`[0,1]` 范围。

## 4. 冲突门控

常速最近接近量：

```text
r = p_neighbor - p_ego
u = v_neighbor - v_ego
raw_t_cpa = -dot(r,u) / (dot(u,u) + eps)
t_cpa = clamp(raw_t_cpa, 0, horizon)
d_cpa = norm(r + t_cpa*u)
```

只有同时满足下列条件时 `pair_mask=true`：

1. ID 合法且不是 ego 自己；
2. 当前距离不超过 `sensing_distance_meters`；
3. 原始 CPA 时间位于 `[0, prediction_horizon_seconds]`；
4. CPA 距离不超过 `conflict_distance_meters`。

紧迫度随 `t_cpa` 和 `d_cpa` 单调衰减，并在非冲突边上归零：

```text
urgency = exp(-t_cpa/tau_t - d_cpa/tau_d) * pair_mask
```

`confidence` 第一版只表达当前可见距离置信度，不读取未来轨迹或其他车辆未来动作。

## 5. reset 合同

车辆单独重生时，只将对应 `[env, agent]` 标记为 `true`；整个环境重置时，将该环境
所有 agent 标记为 `true`。mask 以单步脉冲形式通过 `info` 发出，然后清零。

M4 尚不持有跨步 Opinion 状态。M6 必须使用该 mask 清除对应车辆作为 ego 和
neighbor 的状态行/列，并同时使用 episode `done` 处理完整环境边界。

## 6. 配置与门控

完整/默认 Opinion 配置仍保持 M3 no-op：

```json
"emit_pair_info": false
```

M4 提供完整预算和 pilot 两份独立配置：

```text
configs/opinion/m4_pair_info.json
configs/opinion/m4_pair_info_pilot.json
```

它设置：

```json
"stage": "base",
"use_opinion_marl": false,
"emit_pair_info": true
```

这里 `use_opinion_marl=false` 表示尚未让 Opinion 改变 policy；`emit_pair_info=true`
仅打开环境遥测旁路。标准 `python main_training.py` 完全不传这个配置。

## 7. 用户手动训练与测试

在项目根目录执行：

```bash
conda activate sigmarl-nod
python main_training_opinion.py --config configs/opinion/m4_pair_info_pilot.json
python main_testing_opinion.py --config configs/opinion/m4_pair_info_pilot.json
```

训练输出位于：

```text
outputs/opinion_pilot/m4_pair_info/runs/m4-pair-info-seed<seed>-<id>/
```

如需测试指定 run：

```bash
python main_testing_opinion.py \
  --config configs/opinion/m4_pair_info_pilot.json \
  --run-dir outputs/opinion_pilot/m4_pair_info/runs/<run-id>
```

标准 Base 入口仍为：

```bash
python main_training.py
python main_testing.py
```

pilot 闭环确认后，完整预算训练与测试改用：

```bash
python main_training_opinion.py --config configs/opinion/m4_pair_info.json
python main_testing_opinion.py --config configs/opinion/m4_pair_info.json
```

完整预算输出根目录是 `outputs/opinion/m4_pair_info/`。

## 8. 如何判断 M4

M4 没有改变最终动作分布，所以合理预期是 reward/collision 与同 seed、同预算 Base
近似，而不是稳定提升。主要检查：

- 训练和可视化测试能完整结束；
- 原 observation/action/reward shape 不变；
- 开启 M4 后没有 NaN/Inf；
- 输出目录、权重、曲线和配置快照仍完整；
- wall time 增量来自 ConflictGraph 和额外 info 存储，且在可接受范围内。

参考数学测试文件为 `tests/opinion/test_m4_conflict_graph.py`。按当前项目约定，是否
执行测试和正式训练由用户决定。

## 9. M5 交接

M5 首次把 M3 与 M4 接起来：加载已训练 Base Actor，计算当前帧 Evidence/Residual，
只修正速度分布的 location，保持转向和 scale 不变。M5 不应提前实现跨时间
`z_dense`；真实状态记忆仍留给 M6。
