# AVOCADO A0-A2 严格基线：实现、验证与边界

> 状态：A0-A2 已实现并通过自动测试  
> 范围：二维全向、圆盘、单积分器、无学习  
> 与 MARL 的关系：这是接入 SigmaRL/MARL 前的独立控制基线，不修改现有 M2-M9

## 1. 阶段定义

本路线将“先验证 AVOCADO 本身有效”落实为三个可独立验收的阶段：

| 阶段 | 完整工作 | 验收证据 |
|---|---|---|
| A0 | 实现 TTC、有限时域 VO、投影估计、注意力、意见动力学、OCA 半空间与二维速度优化 | 解析值和公式级单元测试 |
| A1 | 建立独立 VMAS 全向圆盘环境，动作直接是 `[vx, vy]` | 验证 `p[k+1]=p[k]+dt*v[k]`，无自行车适配 |
| A2 | 运行 Preferred、固定 ORCA 和 AVOCADO 对照实验 | 成功率、碰撞率、超时率、最小净空、轨迹图和硬门槛 |

A0-A2 明确不包含：

- EvidenceNet；
- MAPPO 或任何策略训练；
- SigmaRL 道路地图和参考路径；
- 车辆航向、角速度、转角或运动学自行车模型；
- 只选择两个最近邻的近似。

因此，A2 通过只能证明 AVOCADO 在其原始全向假设下的闭环有效性，不能直接推出它在
SigmaRL 非完整车辆上的安全性。自行车适配应从 A3 开始。

## 2. 论文公式与官方参考实现的差异

项目同时保留“PDF 字面公式”和“作者公开代码的可执行语义”，避免把两者的差异悄悄
混成一个实现。闭环 A1/A2 默认对齐作者公开的 2025 AVOCADO 仓库。

- 理论来源：项目内 `docs/papers/AVOCADO_2025.pdf`；
- 可执行语义来源：<https://github.com/dmartinezbaselga/AVOCADO>，重点对照
  `src/Agent.cpp`、`actors.py` 和 RVO2 线性规划回退。

| 项目 | 本项目内 PDF | 作者公开实现 | 当前处理 |
|---|---|---|---|
| 注意力 | 式(11)写成连续微分式并声明 Euler 离散 | `A+=(1-delta)A+delta*tanh(kappa/tau)` | A0 两者均实现并分别测试；A2 使用官方离散滤波 |
| TTC 二次项 | 式(13)展开后应为 `||dp||^2-R^2`，但式(14)文字给出相反号 | 使用几何正确的正号 | 使用正确展开，并用解析接触时间测试 |
| VO 相对速度 | 公式描述以首选速度为核心 | 官方控制器以当前执行速度构造 VO | A2 使用当前执行速度，首选速度仅作为优化目标 |
| OCA 锚点 | 式(7)写为首选速度加责任修正 | 官方代码使用当前速度加责任修正 | A2 对齐官方代码 |
| `y` 的修正向量 | Algorithm 1 的书写顺序像是使用当前 `u` | 官方代码先用上一拍保存的 `u` 更新意见，再计算本拍 `u` | A2 保存并使用上一拍 `u` |
| VO 时域 | PDF 未给出统一默认数值 | 官方 Python 包装器默认 `2.5 s` | 配置固定为 `2.5 s` |
| 圆盘裕量 | `r_s` 由几何和安全要求共同给定 | 官方仿真使用物理半径的 `1.1` 倍 | 物理半径与避碰半径显式分离，比例为 `1.1` |
| 感知噪声 | 数值实验采用 `0.0001` | 官方 Python 包装器默认 `0.0005` | A2 配置使用论文实验值 `0.0001`，代码默认值保留官方包装器值 |

这一区分很重要：如果按 PDF 中的连续注意力式直接乘 `dt=0.05`，意见在短时迎面冲突
中变化明显慢于官方实现；如果在多约束不可行时直接退回首选速度，圆形对穿会发生碰撞。
当前实现对后者采用与 RVO2 一致的顺序最小违约回退，同时把 `feasible=false` 记入指标，
不会把不可行解错误报告为具有安全保证。

本项目没有 vendoring 或绑定作者的 Python/C++ 软件包；A0-A2 是在本项目内编写的
PyTorch/VMAS 实现。RVO2 式几何和线性规划回退的来源与双许可证边界记录在
`utilities/avocado/NOTICE.md`，后续发布或分发前仍应随项目总许可证一并复核。

## 3. 代码与数据边界

```text
utilities/avocado/
├── core.py          # A0 纯数学函数、二维 OCA 求解器
├── controller.py    # 官方参考语义的有状态 AVOCADO 控制器
├── config.py        # 独立且严格的 A2 JSON 配置边界
├── benchmark.py     # A2 rollout、指标、门槛与轨迹图
└── NOTICE.md        # 官方源码、RVO2 与许可证来源说明

scenarios/avocado_holonomic.py
    # A1 VMAS 圆盘单积分器环境

configs/avocado/a2_strict_benchmark.json
    # 论文参数、官方仿真默认值和验收门槛

main_testing_avocado.py
    # A2 独立测试入口

tests/avocado/
├── test_a0_math.py
├── test_a1_environment.py
└── test_a2_benchmark.py
```

该命名空间不导入 `utilities.opinion`，也不复用 `main_testing_opinion.py`。因此可以单独
执行、回归和删除，不影响当前 Opinion-MARL 训练产物。

## 4. A0 验收内容

A0 自动检查以下数学性质：

1. 迎面接近的首次接触时间等于式(13)的解析解；
2. 远离运动返回 `tau=inf`，已重叠返回 `tau=0`；
3. PDF 字面注意力 Euler 更新与官方离散滤波分别等于直接公式计算；
4. 非合作 `delta_v=0` 时，式(15)输出 `tanh(-epsilon/2)`；
5. 意见更新精确等于式(10)的前向 Euler 步；
6. VO 内外两侧的半空间法向一致，当前安全速度不会被错误排除；
7. 可行 OCA 交集返回离首选速度最近的点；
8. 不可行交集显式标记 `feasible=false`，返回速度仍满足最大速度界。

## 5. A1 验收内容

VMAS 仅负责批量状态容器和渲染。`DirectVelocityDynamics` 将速度动作转换成恰好抵消
当前速度的力，使 VMAS 单步满足：

\[
v_{k+1}=u_k,
\qquad
p_{k+1}=p_k+\Delta t\,u_k.
\]

环境中的实体：

- 形状为 `Sphere`；
- `rotatable=false`；
- 动作为二维速度；
- 物理碰撞响应关闭，碰撞由圆盘距离独立统计；
- 控制器处理感知半径内的全部实体，而不是 SigmaRL 的两个最近邻。

关闭物理碰撞响应是为了让失败轨迹保持原始单积分器运动，不让 VMAS 接触力掩盖控制器
是否真正完成了避碰。

## 6. A2 场景、指标与通过门槛

默认配置包含：

- `head_on_noncooperative`：一个 AVOCADO 机器人和一个不避让的动态体；
- `head_on_cooperative`：两个 AVOCADO 机器人迎面交换位置；
- `circle_cooperative_6`：六个 AVOCADO 机器人同时前往圆周对点。

每个场景运行三种控制器：

- `preferred`：只执行目标速度，用来证明场景确实包含冲突；
- `orca`：固定 0.5 责任；
- `avocado`：在线更新注意力、意见和合作度。

每次运行记录：

- `success_rate`、`collision_rate`、`timeout_rate`；
- 成功回合平均到达时间；
- 平均路径长度和全局最小净空；
- 单机器人平均控制计算时间；
- OCA 不可行率、最大注意力和平均绝对意见。

默认硬门槛要求每个 AVOCADO 场景成功率不低于 `0.875`、碰撞率不高于 `0.125`，并
要求非合作迎面场景的 Preferred 基线暴露冲突，且 AVOCADO 至少降低 `0.75` 的碰撞率。
入口在门槛失败时返回非零退出码。

## 7. 运行方法

新增测试：

```bash
MPLCONFIGDIR=/tmp/matplotlib-avocado \
conda run -n sigmarl-nod \
python -m unittest discover -s tests/avocado -p 'test_*.py' -v
```

完整 A2 验证：

```bash
MPLCONFIGDIR=/tmp/matplotlib-avocado \
conda run -n sigmarl-nod \
python main_testing_avocado.py \
  --config configs/avocado/a2_strict_benchmark.json
```

程序在 `outputs/avocado/a2_strict/<timestamp>/` 下生成：

- `summary.json`：解析配置、全部指标与逐条门槛结果；
- `trajectories_<case>.png`：Preferred、ORCA、AVOCADO 的同场景轨迹对照。

快速检查可加 `--episodes 2 --no-plots`；指定稳定输出目录可使用 `--output-dir PATH`。

## 8. 2026-08-27 正式验收记录

使用提交配置的 8 个并行回合和 1000 步上限运行，硬门槛通过。证据保存在
`outputs/avocado/a2_strict/a0_a2_validation_20260827/summary.json`。

| 场景 | Preferred 碰撞率 | ORCA 成功/碰撞 | AVOCADO 成功/碰撞 | AVOCADO 最小净空 |
|---|---:|---:|---:|---:|
| head-on non-cooperative | 1.000 | 1.000 / 0.000 | 1.000 / 0.000 | 0.0402 m |
| head-on cooperative | 1.000 | 1.000 / 0.000 | 1.000 / 0.000 | 0.0348 m |
| circle cooperative, 6 agents | 1.000 | 1.000 / 0.000 | 1.000 / 0.000 | 0.0378 m |

圆形六机器人场景的 `infeasible_projection_rate=0.0751`。这不表示发生碰撞，而是表示
部分时刻所有 OCA 半空间与最大速度圆没有共同可行点，控制器使用了官方 RVO2 式顺序
最小违约回退。因此 A2 的结论是“在当前三类确定性布局和随机感知噪声下闭环有效”，而
不是“每一个离散时刻都存在严格可行的 OCA 解”。A3 应继续把该指标作为诊断量。

## 9. 进入 A3 的前置条件

只有同时满足以下条件，才建议把工作推进到 SigmaRL 自行车模型：

1. A0 公式测试全部通过；
2. A1 单积分器等式测试全部通过；
3. A2 `Validation gate: PASSED`；
4. 轨迹图确认避碰不是由超时、停滞或 VMAS 接触力造成；
5. `infeasible_projection_rate` 被保存并在后续实验中持续监控。

A3 必须改名为 `AVOCADO-KB` 或等价名称，并重新验证非完整运动学下的跟踪误差和安全
裕量；不得继续宣称其严格继承原始全向 AVOCADO 的闭环保证。

该前置阶段现已完成，正式实现、验收结果和局限见
[`AVOCADO_A3_ROAD_ENVIRONMENT.md`](AVOCADO_A3_ROAD_ENVIRONMENT.md)。
