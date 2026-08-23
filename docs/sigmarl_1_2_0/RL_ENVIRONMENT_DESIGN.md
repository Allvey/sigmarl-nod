# SigmaRL 1.2.0 强化学习环境设计

> 事实来源：tag `1.2.0` 的 `config.json`、`scenarios/road_traffic.py`、
> `utilities/helper_scenario.py` 和 `utilities/mappo_cavs.py`  
> 对齐日期：2026-08-23

## 1. 环境定位

SigmaRL 使用 VMAS 构造并行道路交通环境。每辆车是一个智能体，Actor 分散执行，
MAPPO Critic 在训练时使用联合观测。默认场景为 `CPM_mixed`，配置使用 4 辆车。

本项目重建时，以下部分先保持 1.2.0 原样：

- 地图和参考路径；
- 车辆动力学；
- 原始 observation；
- 二维连续动作；
- reward；
- done/reset；
- VMAS 并行采样。

Opinion Dynamics 只能作为旁路协调状态接入，不能悄悄改变这些基线定义。

## 2. 默认运行规模

1.2.0 根配置的关键值为：

| 参数 | 值 | 含义 |
|---|---:|---|
| `n_agents` | 4 | 每个环境的车辆数 |
| `dt` | 0.05 s | 物理步长 |
| `max_steps` | 128 | 单 episode 最大步数 |
| `frames_per_batch` | 4096 | 每轮采样 team frames |
| `num_vmas_envs` | 32 | 由 `4096 / 128` 推导 |
| `n_iters` | 250 | 训练迭代数 |
| `num_epochs` | 60 | 每轮 PPO epochs |
| `minibatch_size` | 512 | 扁平 minibatch 大小 |

因此原始训练每轮有 8 个 minibatch，每轮共 `60 × 8 = 480` 次 PPO 更新。

## 3. 状态与车辆动力学

每辆车的物理状态至少包含：

```text
position [x,y]
rotation/yaw
velocity [vx,vy]
```

环境使用项目内的 kinematic bicycle dynamics。动作不是直接设置位置，而是驱动车辆
沿动力学演化。车辆几何以矩形表示，碰撞计算使用车辆顶点及 lanelet 边界关系。

## 4. 动作空间

默认动作张量：

```text
("agents", "action") : [E,4,2]
```

两个动作分量分别对应速度控制和转向控制。Actor 输出正态分布参数，随后使用
`TanhNormal` 将采样动作限制在 VMAS action spec 的合法范围。

Opinion 接入第一版只允许对速度通道的分布均值加入有界 residual；转向均值、scale
和动作边界保持 Base Actor 定义。这一限制来自 Opinion 技术路线，不是 1.2.0
环境本身的功能。

## 5. 观测空间

默认 observation 为 `[E,4,32]`，详细拆分见
[`OBSERVATION_SPACE_DETAILS.md`](OBSERVATION_SPACE_DETAILS.md)。关键点是：

- ego 坐标系；
- 10 维自车特征；
- 两个最近邻，每个 11 维；
- 默认不启用 mask、噪声和 opponent modeling；
- 最近邻槽位不是稳定车辆身份。

## 6. 奖励结构

单车 reward 由下列部分相加：

### 正向或任务进展项

- 沿短期参考路径方向前进；
- 保持较高的正向速度；
- 到达目标/出口奖励（默认权重可为零）。

### 安全与约束惩罚

- 靠近 lanelet 边界；
- 靠近其他车辆；
- 偏离参考路径；
- 转向变化过快；
- 与其他车辆碰撞；
- 与 lanelet 边界碰撞；
- 与运动方向相关的时间项。

奖励使用归一化尺度，代码中通过当前物理距离、速度、参考路径方向和碰撞布尔量计算。
Opinion 重建不得为了制造性能提升而单独改变 reward；如需新 reward，必须作为单独
实验变量并同时应用于所有比较方法。

## 7. episode 终止和 reset

### 训练模式

环境在以下情况整体结束：

- 达到 `max_steps - 1`；
- 任意车辆与其他车辆碰撞；
- 任意车辆与 lanelet 边界碰撞。

对非 `CPM_entire` 场景，如果单车离开 entry/exit segment 且环境没有整体 done，
环境会只 reset 对应车辆。

### 测试模式

测试模式下，整体环境通常只因最大步数结束；发生碰撞或离开 entry/exit 时可局部
reset 单车，使其他交互继续。

### 对 Opinion 状态的影响

Opinion 状态必须遵守：

- 整体 done：清空该环境全部 `z[i,j]`；
- 单车 reset：清空所有以该车为端点的有向状态；
- 候选边暂时消失：按固定动力学衰减或按明确合同清除；
- reset 事件每个物理步只能消费一次。

环境负责报告 reset，不负责持有 `z`。

## 8. 原始训练数据流

SigmaRL 1.2.0 的主要训练链路为：

```text
32 个 VMAS 环境并行采样
        ↓
4096-frame TensorDict rollout
        ↓
GAE
        ↓
reshape(-1)，展平环境与时间批次
        ↓
ReplayBuffer 无放回采样 512-frame minibatch
        ↓
ClipPPOLoss × 60 epochs
```

这条扁平化路径是 Base-MAPPO 的速度基准。Opinion 的 Base 阶段必须继续使用它，
不得因为代码复用而进入逐 chunk、逐时间步 Sequence PPO。

## 9. Opinion 接入边界

允许的最小环境修改：

1. 对原始两个最近邻，从当前物理状态构造 10 维有向车辆对特征；
2. 输出这两个槽位对应的显式 global neighbor ID 和有效 mask；
3. 输出 urgency/confidence；
4. 输出精确的 agent/environment reset 事件；
5. 测试时渲染 detached Opinion 诊断快照。

环境禁止：

- 保存或积分 `z`；
- 调用 EvidenceNet；
- 产生 priority/leader/total order；
- 使用真实未来轨迹或其他执行期不可用信息；
- 在 Opinion 关闭时改变原始 observation、reward、done 或动作空间。

第一版冲突判定使用当前相对位置/速度的短时常速最近接近计算，不读取未来真值轨迹
或其他车辆真实未来动作。扩展候选数、引入参考路径交点或行为预测器都属于后续
独立实验，不能静默改变 Base 的感知预算。

## 10. 公平实验合同

Base、Opinion 和外部 TSC 比较必须固定：

- 同一地图与场景分布；
- 同一 observation/action/reward；
- 同一 `dt`、episode 长度和采样预算；
- 同一随机种子集合；
- 同一碰撞指标定义；
- 同一 checkpoint 选择规则和评估步数。

TSC 的 topology/action predictor 可以存在于它自己的外部基线仓库，但不得进入新的
Opinion 训练路径。
