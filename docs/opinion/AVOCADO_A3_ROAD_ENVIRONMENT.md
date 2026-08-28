# AVOCADO A3：直接接入 SigmaRL 道路强化学习环境

> 状态：A3.2 交叉口安全修复已实现并通过正式全场景验收  
> 方法名称：AVOCADO-KB  
> 范围：确定性控制器、无训练、原 `road_traffic` 场景、原 RK4 自行车动力学

## 1. A3 回答的问题

A0-A2 证明原始全向 AVOCADO 在圆盘单积分器环境中可以闭环避碰。A3 进一步验证：

> 不依赖 MARL checkpoint，AVOCADO 能否作为确定性策略，直接调用项目训练所用的
> `ScenarioRoadTraffic` 和 `KinematicBicycle`，并通过原生 `env.step()` 连续运行？

答案是可以，但必须增加非完整运动学和道路可行域适配。因此 A3 名为
`AVOCADO-KB`，不宣称仍是严格原始 AVOCADO。

## 2. 闭环数据流

```text
ScenarioRoadTraffic 真实位置、速度、航向、短期参考路径
                         ↓
          Stanley 型横向反馈路径首选速度
                         ↓
        AVOCADO / 固定 ORCA + ±12°路径速度锥
                  （同一半平面求解）
                         ↓
              逆运动学自行车动作适配器
                         ↓
       可审计 TTC 紧急制动屏障（仅 AVOCADO-KB）
                         ↓
                 [速度指令, 前轮转角]
                         ↓
       原 KinematicBicycle(RK4) + 原奖励/碰撞/重置
```

控制器没有修改 `scenarios/road_traffic.py` 或
`utilities/kinematic_bicycle.py`。A3 仅在基准侧增加状态桥接、动作适配和只读事件快照。

### 2.1 A3.1 路径跟随改进

初版 A3 直接指向第三个短期参考点，缺少横向误差反馈，在窄车道和急弯中容易切弯。
A3.1 用短期路径前两个点构造单位切向量和左法向量：

\[
\mathbf t=\frac{\mathbf p_1-\mathbf p_0}
{\|\mathbf p_1-\mathbf p_0\|},\qquad
\mathbf n=(-t_y,t_x),
\]

并计算有符号横向误差与 Stanley 型回正角：

\[
e_y=(\mathbf p_0-\mathbf p)^{\mathsf T}\mathbf n,
\qquad
\Delta\psi=\operatorname{clip}\left(
\operatorname{atan2}(k_e e_y,v_{ref}+v_s),\pm\psi_{max}
\right).
\]

最终首选速度为：

\[
\mathbf v_{pref}=v_{ref}R(\Delta\psi)\mathbf t.
\]

正式配置采用 `k_e=7`、`v_s=0.2 m/s`、`\psi_{max}=20°`。AVOCADO 接收该速度作为
无冲突首选速度，因此意见动力学和 VO 协调结构保持不变。

## 3. 从二维速度到自行车动作

项目自行车模型的瞬时平移方向为：

\[
\psi_v=\psi+\beta,
\qquad
\beta=\arctan\!\left(\tan\delta\frac{l_r}{l_f+l_r}\right).
\]

A3 对 AVOCADO 输出的期望速度 \(\mathbf v_d\) 计算：

\[
e_\psi=\operatorname{wrap}\left(
\operatorname{atan2}(v_{d,y},v_{d,x})-\psi
\right),
\]

再将期望侧偏角限制到车辆可达范围：

\[
\beta_c=\operatorname{clip}(e_\psi,-\beta_{\max},\beta_{\max}),
\qquad
\delta_c=\arctan\!\left(
\tan\beta_c\frac{l_f+l_r}{l_r}
\right).
\]

无法瞬时达到的剩余航向误差会降低速度，但保留 `minimum_speed_ratio=0.2`，避免车辆
在需要转向时完全停住。单元测试验证了可达方向的侧偏角反解、转向饱和、速度界和零
速度行为。

## 4. 道路约束、互补责任与 TTC 屏障

原始 AVOCADO 可以向任意二维方向侧移，而道路车辆不能。A3.1 曾先求解 OCA，再把输出
速度截断到参考路径 `±12°`。这一串行顺序可能把一个已经位于 VO 外的速度重新推入 VO，
正是 `intersection_2` 中两车临近冲突时仍同时加速的主因。

A3.2 将 `±12°` 写成两个过原点的速度半平面，与全部 OCA 半平面及最大速度圆一次联合
求解。因此返回速度同时满足避碰几何和道路航向约束；不再进行会破坏 VO 可行性的事后
航向截断。该道路速度锥属于 A3.2 扩展，不属于原始 AVOCADO。

全受控车辆还对每个无序车辆对归一化意见给出的原始避让责任：

\[
\rho_{ij}=\frac{1-q_{ij}}{(1-q_{ij})+(1-q_{ji})},\qquad
\rho_{ij}+\rho_{ji}=1.
\]

这避免两车各自独立推断后总责任大于 1、同时做出过强动作。意见状态本身仍按 AVOCADO
方程独立更新；互补归一化只作用于道路 OCA 责任分配，A0-A2 严格基线默认关闭。

最后，A3.2 用自行车适配后的真实可执行速度重新计算成对 TTC。若 TTC 小于配置的
`0.6 s`，由责任较大的一车制动；责任相同则用车辆编号确定性打破平局。该屏障是明确
标记、可关闭、单独计数的工程安全兜底，不应被表述为 AVOCADO 理论的一部分，也不能
用其干预后的结果夸大意见动力学效果。

当短期参考路径到达末端时，多个参考点会重合。若直接追踪最后一点，车辆会在出口前
停止或回头。A3 在此时沿当前车辆航向保持巡航，使车辆真正穿过场景定义的出口线。

道路测试模式会在碰撞、越界或完成路线后立即重置单车。`A3ScenarioRoadTraffic` 只在原
`done()` 重置前快照事件原因，不改变任何状态转移；随后 `reset_agents()` 清除该车对应
的 AVOCADO 入边、出边、上一拍修正和意见状态。

## 5. 指标语义

道路测试是固定时域连续交通流，同一实例中的车辆可以完成多条路线。因此 A3 使用每
1000 个“车-时间步”的事件率，而不是 A2 的单目标成功率：

- 车辆碰撞事件；
- 车道边界碰撞事件；
- 错误入口事件；
- 路线完成事件；
- 无车辆碰撞实例比例；
- 平均奖励、参考线距离、最小车辆净空；
- 重置前参考线误差的均值、RMS、P95 和最大值；
- 重置前最小车道净空，负值表示车辆轮廓已经侵入边界；
- 期望二维速度到实际自行车速度的跟踪误差；
- 转向饱和率；
- OCA 不可行率、注意力、意见和控制耗时。
- TTC 屏障干预率、干预后仍不安全的车辆对事件率。

## 6. 验收结果

### 6.1 A3.2 `intersection_2` 专项验证

截图对应的 `seed=2026`、1 个实例、400 步复现中，旧串行 `±12°` 截断为
`3.333` 次车辆碰撞/千车步；改为联合约束后降为 `0`，且路径误差 P95 为
`0.027 m`。完整 A3.2（互补责任与屏障开启）在同一复现中屏障干预为 `0`，说明该次
改善来自联合求解，而不是紧急制动。

8 个并行随机初态、每个 400 步的压力测试中，车辆碰撞仍为 `0`，路径误差 P95 为
`0.026 m`，屏障干预率为 `0.25%`，干预后不安全车辆对事件率为 `0`。另一个 4×400
步消融中，关闭屏障后无论是否启用互补责任，车辆碰撞均为 `0`。这些是确定性回归证据，
不是对任意初态的形式化无碰撞证明。

正式配置为 4 个并行实例、600 步、`dt=0.05 s`、随机种子 2026。完整记录与三张轨迹
图位于 `outputs/avocado/a3_road/a3_joint_ttc_validation_20260828/`，全部验证门槛通过。

| 场景/控制器 | 车辆碰撞/千车步 | 车道事件/千车步 | 完成路线/千车步 | 路径 P95 | 屏障率 |
|---|---:|---:|---:|---:|---:|
| CPM mixed / path | 2.292 | 7.812 | 8.646 | 0.121 m | — |
| CPM mixed / ORCA-KB | **0.000** | **2.396** | 4.062 | 0.119 m | — |
| CPM mixed / AVOCADO-KB | **0.000** | 3.542 | **4.688** | **0.116 m** | 1.16% |
| intersection 1 / path | 7.917 | 4.097 | **17.986** | 0.034 m | — |
| intersection 1 / ORCA-KB | **0.000** | **2.500** | 16.111 | 0.030 m | — |
| intersection 1 / AVOCADO-KB | **0.000** | 2.986 | **16.458** | **0.029 m** | 0.33% |
| intersection 2 / path | 7.500 | 2.361 | **16.597** | **0.025 m** | — |
| intersection 2 / ORCA-KB | **0.000** | 1.667 | 15.625 | 0.027 m | — |
| intersection 2 / AVOCADO-KB | **0.000** | **1.111** | **15.833** | 0.027 m | 0.19% |

所有 AVOCADO-KB 场景的干预后不安全车辆对事件率均为 0。AVOCADO-KB 与 ORCA-KB
在本次固定预算内都实现了零车辆碰撞，因而本阶段仍不能声称意见动力学优于固定 ORCA；
它证明的是 A3.2 修复有效且可作为后续 MARL 耦合的可靠道路基线。

### 6.2 A3.1 历史正式验收

配置：4 个并行实例、600 步、`dt=0.05 s`、随机种子 2026。完整记录位于
`outputs/avocado/a3_road/a3_stanley_validation_20260827/summary.json`。

| 场景/控制器 | 车辆碰撞/千车步 | 车道事件/千车步 | 完成路线/千车步 | 路径误差 P95 |
|---|---:|---:|---:|---:|
| CPM mixed / path | 2.292 | 7.812 | 8.646 | 0.121 m |
| CPM mixed / ORCA-KB | **0.833** | **5.521** | **9.167** | 0.105 m |
| CPM mixed / AVOCADO-KB | 1.667 | 6.875 | 8.750 | **0.103 m** |
| intersection 1 / path | 7.917 | 4.097 | **17.986** | 0.034 m |
| intersection 1 / ORCA-KB | **0.694** | 2.569 | 16.458 | **0.030 m** |
| intersection 1 / AVOCADO-KB | **0.694** | **2.431** | 16.736 | **0.030 m** |

硬门槛通过。相对旧第三点追踪，AVOCADO-KB 在 CPM mixed 的 P95 路径误差从
`0.1409 m` 降至 `0.1034 m`，车道事件从 `21.979` 降至 `6.875/千车步`；在
intersection 1 中，P95 从 `0.0452 m` 降至 `0.0303 m`，车道事件从 `22.083` 降至
`2.431/千车步`。

针对问题最明显的 CPM entire，4 个并行实例、300 步的同种子专项对照为：

| 跟随器 | 平均误差 | RMS | P95 | 车道事件/千车步 | 车辆事件/千车步 |
|---|---:|---:|---:|---:|---:|
| 旧第三点追踪 | 0.0354 m | 0.0492 m | 0.1005 m | 10.333 | **0.000** |
| A3.1 Stanley | **0.0076 m** | **0.0106 m** | **0.0217 m** | **2.167** | 0.444 |

必须同时保留下列结论：

- 结果不是零碰撞安全保证；路径误差降低不能替代车辆避碰安全证明；
- CPM entire 专项中车辆事件从 `0` 增至 `0.444/千车步`，仍需联合优化车道与车辆安全；
- CPM mixed 中固定 ORCA-KB 的车辆碰撞和车道事件优于 AVOCADO-KB；
- A3 证明的是“成功接入且显著优于无避碰控制”，没有证明 AVOCADO 优于 ORCA；
- A3.1 的串行道路偏航截断会破坏全向 OCA 的几何可行性；A3.2 已改为联合约束，仍需
  持续记录联合可行性与屏障干预率。

## 7. 测试和可视化

A3 自动测试：

```bash
MPLCONFIGDIR=/tmp/matplotlib-avocado \
conda run -n sigmarl-nod \
python -m unittest \
  tests.avocado.test_a3_bicycle \
  tests.avocado.test_a3_safety \
  tests.avocado.test_a3_road_environment -v
```

正式验证：

```bash
MPLCONFIGDIR=/tmp/matplotlib-avocado \
conda run -n sigmarl-nod \
python main_testing_avocado_kb.py \
  --config configs/avocado/a3_road_environment.json
```

原生 VMAS 实时可视化（与 `main_testing.py` 使用同一条渲染链路）：

```bash
conda run -n sigmarl-nod \
python main_testing_avocado_kb.py \
  --render \
  --planner avocado_kb \
  --max-steps 1200
```

默认场景由 `main_testing_avocado_kb.py` 顶部的 `TEST_SCENARIO_TYPE` 选择，与
`main_testing.py` 和 `main_testing_opinion.py` 的使用方式一致。也可以不修改源码，直接
通过命令行覆盖：

```bash
conda run -n sigmarl-nod \
python main_testing_avocado_kb.py \
  --render \
  --scenario intersection_2 \
  --planner avocado_kb \
  --max-steps 1200
```

`--scenario` 会使用 `SCENARIOS` 中该环境的默认车辆数。若要严格复现 A3 JSON 中定义的
场景和车辆数，可改用 `--case cpm_mixed_4` 或 `--case intersection_1_6`；二者不能与
`--scenario` 同时使用。

实时模式固定使用一个环境，每次 `env.step()` 后调用 `env.render(mode="human")`，并由
道路场景已有的 `is_real_time_rendering` 按 `dt=0.05 s` 控制节拍。窗口同时显示时间、
步数、车辆速度/转角、横向误差、车道净空、最大注意力、平均意见幅值、真实活动 VO 数、
联合求解不可行数、临界车辆对 TTC/责任、TTC 屏障干预数和本步重置数。可将 `--case`
改为
`intersection_1_6`，或用 `--planner path_following|orca_kb|avocado_kb` 做现场对照。
该模式需要有图形桌面；服务器或无显示器环境继续使用下面的离线动画模式。

快速离线可视化：

```bash
MPLCONFIGDIR=/tmp/matplotlib-avocado \
conda run -n sigmarl-nod \
python main_testing_avocado_kb.py \
  --episodes 1 \
  --max-steps 300 \
  --video
```

每个场景生成三控制器轨迹对照 PNG 和 AVOCADO-KB 动画。有 FFmpeg 时动画为 MP4，
否则自动使用 Pillow 生成 GIF，因此无图形显示器的环境也可以运行。

## 8. A3 与 MARL 的边界

A3 仍然不训练 MARL，也不使用 EvidenceNet。当前控制器从仿真环境读取所有车辆的真实
位置、速度和航向，而不是严格复用 MARL 的局部归一化观测，因此属于全状态确定性道路
基线。

下一阶段接入 MARL 时，应把 A3 固定为可复现对照组，并比较：

1. Base-MAPPO；
2. 固定 ORCA-KB；
3. AVOCADO-KB；
4. MARL + Opinion Dynamics。

只有第四项在相同观测、随机种子和场景预算下稳定优于前三项，才能说明 MARL 与意见
动力学的耦合提供了额外价值。
