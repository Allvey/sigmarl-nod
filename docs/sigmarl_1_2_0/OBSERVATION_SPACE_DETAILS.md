# SigmaRL 1.2.0 观测空间说明

> 代码基线：SigmaRL tag `1.2.0`  
> 配置基线：该 tag 根目录 `config.json`  
> 实测场景：`CPM_mixed`、4 agents、2 个 VMAS 环境  
> 对齐日期：2026-08-23

## 1. 默认观测合同

原始环境 reset 后的主要张量为：

```text
observation key: ("agents", "observation")
observation:     [E, 4, 32]
action:          [E, 4, 2]
```

其中 `E` 是并行 VMAS 环境数。每辆车的 32 维观测由 10 维自车信息和两个各 11 维
的邻车槽位构成：

```text
32 = 10 + 2 × 11
```

该结论只适用于 1.2.0 默认开关：

```text
is_partial_observation = true
n_nearing_agents_observed = 2
n_points_short_term = 3
is_ego_view = true
is_observe_vertices = true
is_observe_distance_to_agents = true
is_observe_distance_to_boundaries = true
is_observe_distance_to_center_line = true
is_observe_ref_path_other_agents = false
is_using_opponent_modeling = false
```

改变这些开关会改变维度或字段语义，不能继续硬编码 32。

## 2. 自车观测：10 维

默认使用 ego 坐标系，因此自车绝对位置和绝对航向不进入 Actor。自车部分为：

| 字段 | 维度 | 说明 |
|---|---:|---|
| 纵向速度 | 1 | ego 坐标系下只保留有意义的前向分量 |
| 短期参考路径 | 6 | 3 个参考点，每点二维坐标 |
| 到参考路径距离 | 1 | 归一化标量 |
| 到左边界距离 | 1 | 最近边界距离 |
| 到右边界距离 | 1 | 最近边界距离 |
| 合计 | 10 |  |

如果关闭 ego view，自车位置和旋转会重新进入观测；如果不使用边界距离，则左右
边界会改为若干边界点坐标，维度随之变化。

## 3. 每个邻车槽位：11 维

在部分观测模式下，环境依据当前车辆间距离选择最近的两个邻车。每个邻车槽位为：

| 字段 | 维度 | 说明 |
|---|---:|---|
| 四个矩形顶点 | 8 | 4 个顶点在 ego 坐标系中的二维坐标 |
| 邻车速度 | 2 | 旋转到 ego 坐标系的速度向量 |
| 双车距离 | 1 | 归一化的车辆间距离 |
| 合计 | 11 |  |

默认 `is_observe_ref_path_other_agents=false`，因此邻车短期参考路径不进入 Actor。
如果 `is_observe_vertices=false`，8 维顶点会替换为位置、旋转、长度和宽度的组合。

## 4. 邻车索引与槽位语义

`nearing_agents_indices` 通过 `torch.topk(..., largest=False)` 从当前距离矩阵选择。
因此：

- 槽位表示“当前最近邻”，不是稳定的全局车辆身份；
- 邻车距离排序改变时，同一槽位可能对应另一辆车；
- 原始 Actor 是无状态前馈网络，因此不受跨步身份切换直接影响；
- Opinion 状态若跨时间保存，必须使用 global agent ID 建立 `z[i,j]`，不能把槽位
  编号当作身份。

这也是后续 Opinion 环境适配必须输出显式 `neighbor_ids` 的原因。

## 5. mask 与噪声

默认配置：

```text
is_apply_mask = false
is_add_noise = false
```

启用 mask 时，环境会根据距离和 lanelet 关系遮蔽邻车字段，并用预设的零/一 mask
值写入相应特征。启用噪声时，环境会在拼接后的观测上加入随机传感噪声。

Opinion 的 evidence/confidence 不能把 mask 占位值误认为真实物理观测。后续适配应
显式输出布尔 `pair_mask`，并让无效边满足：

```text
confidence = 0
b = 0
不更新对应 z
```

## 6. `info` 不是 Actor 默认输入

SigmaRL 1.2.0 默认 `info()` 输出：

```text
pos                         [E,4,2]
rot                         [E,4,1]
vel                         [E,4,2]
act_vel                     [E,4,1]
act_steer                   [E,4,1]
ref                         [E,4,6]
distance_ref                [E,4,1]
distance_left_b             [E,4,1]
distance_right_b            [E,4,1]
is_collision_with_agents    [E,4,1]
is_collision_with_lanelets  [E,4,1]
```

这些字段用于评估、记录或可选模块；默认 Actor 仍只读取 32 维 observation。后续
Opinion ConflictGraph 可以从当前物理状态构造附加信息，但必须满足：

- 只使用执行期本车可获得的局部物理量；
- 不向 Actor 泄漏全局标签、其他智能体真实动作或 Critic 信息；
- 环境不保存和更新意见状态 `z`；
- Opinion 关闭时，原始 32 维 observation 和原始行为保持不变。

## 7. 对 Opinion-MARL 的稳定接口建议

在不修改原始 observation 的前提下，建议仅在 Opinion 开启时向 `info` 增加：

```text
pair_features     [E,N,K,F]
neighbor_ids      [E,N,K]
pair_mask         [E,N,K]
urgency           [E,N,K]
confidence        [E,N,K]
agent_reset_mask  [E,N]
```

其中 `N=4`。第一版固定 `K=2`，直接对应原始 observation 中的两个最近邻槽位，
但必须额外输出它们的 global agent ID。该接口只表达当前物理关系，不包含 `b`、
`z`、priority、leader 或 topology label。扩展到第三辆未被原始局部观测覆盖的车辆
会改变感知预算，必须作为独立实验，而不能作为默认实现。

## 8. 验证方式

重置到 1.2.0 后应通过真实环境 reset 验证维度，而不是只读文档：

```python
td = env.reset()
assert td["agents", "observation"].shape[-2:] == (4, 32)
assert env.action_spec.shape[-2:] == (4, 2)
```

加入 Opinion 接口后还必须验证：关闭 Opinion 时 observation 数值、shape 和原始
`info` key 集合不变。
