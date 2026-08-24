# M3：Evidence、固定 OpinionDynamics 与有界 Residual

> 实现状态：代码、参考测试和文档已完成；运行由用户手动执行  
> 训练状态：仍为 `stage=base / use_opinion_marl=false`  
> 性能预期：与 R1 Base 等价，不声明提升

## 1. M3 的边界

M3 将技术路线中的三个核心数学映射实现为相互独立的 PyTorch 模块：

```text
pair_features → OpinionEvidenceNet → b
z_prev + b    → OpinionDynamics    → z_next
z_next         → OpinionResidual    → bounded speed-loc residual
```

本阶段不修改 `road_traffic.py`，不创建 ConflictGraph，不维护跨步 `z_dense`，也不把
residual 接入 Actor。真实环境特征由 M4 提供，Actor bridge 在 M5 实现，跨时间状态
由 M6 Collector 管理。

这种拆分保证 M3 后仍可完整执行 Base 训练和测试，不会用随机或虚构 pair feature
改变车辆动作。

## 2. OpinionEvidenceNet

实现文件：`utilities/opinion/evidence_net.py`。

输入：

```text
pair_features [E,N,K,10]
urgency       [E,N,K]
confidence    [E,N,K]
pair_mask     [E,N,K] bool
```

固定 10 维布局：

```text
r_x, r_y, u_x, u_y,
ego_speed, neighbor_speed,
sin(relative_yaw), cos(relative_yaw),
t_cpa, d_cpa
```

模块使用一套共享 MLP 计算相对评分：

```text
ell_ij = G(chi_ij) - G(swap(chi_ij))
raw_b  = b_max * tanh(ell_ij / temperature)
b      = raw_b * urgency * confidence * pair_mask
```

`swap_pair_features()` 不需要额外未来信息：它只用当前相对航向，把相对位置和速度
旋转到邻车坐标系，交换双方速度并翻转相对航向。连续交换两次应恢复原特征，因此
`raw_b_ji = -raw_b_ij`。

最后一层以很小的 Xavier gain 初始化，使初始证据接近中性，同时保留非零梯度路径。
EvidenceNet 的输入中没有 `z`，不会形成内部正反馈。

输出：

```text
antisymmetric_logit [E,N,K]
raw_b                [E,N,K]
b                    [E,N,K]
```

## 3. 固定 OpinionDynamics

实现文件：`utilities/opinion/dynamics.py`。

每次调用只积分一个物理步：

```text
z_next = z_prev + dt * eta_z * (
    -kappa_z * z_prev
    + urgency * nu_z * tanh(alpha_z * z_prev)
    + b
)
```

实现约定：

- `eta_z / kappa_z / nu_z / alpha_z` 注册为 buffer，不是 Parameter；
- `list(dynamics.parameters())` 必须为空；
- `pair_mask=false` 时屏蔽 evidence 和自强化，但保留遗忘项，使旧意见衰减至零；
- 显式 Euler 遗忘步要求 `dt * eta_z * kappa_z <= 1`；
- 模块不保存历史状态，因此同一步不会被内部重复积分；
- `theoretical_bound(b_max)` 返回连续时间最终界
  `(nu_z + b_max) / kappa_z`，供后续诊断使用。

`z_dense` 的 global-ID 映射、reset 和每步一次调用由 M6 实现，不放在数学模块中。

## 4. OpinionResidual

实现文件：`utilities/opinion/residual.py`。

```text
q_ij       = tanh(z_ij / z0)
weight_ij  = urgency_ij * pair_mask_ij
weight_bar = weight / sum(weight)
aggregate  = sum(weight_bar * q)
residual   = clamp(gain * aggregate, -max_abs, max_abs)
```

归一化权重使 residual 不会随候选车辆数量累加放大。没有有效车辆对时，权重、聚合和
residual 都为零。

`apply_to_loc()` 只返回：

```text
final_loc[...,0] = base_loc[...,0] + residual  # 速度
final_loc[...,1] = base_loc[...,1]             # 转向不变
```

它不修改 `scale`。M5 构造最终 `TanhNormal` 时才会使用该接口。

## 5. 梯度边界

M3 形成的未来 Actor 梯度路径是：

```text
EvidenceNet parameters
  → raw_b / b
  → fixed differentiable Dynamics
  → q / residual
  → final policy loc（M5）
```

Dynamics 和 Residual 当前没有可训练 Parameter。Residual 对 `z` 可微，因此不会切断
EvidenceNet 的策略梯度。Critic 尚未连接这些模块。

## 6. 用户手动数学检查

参考测试位于 `tests/opinion/test_m3_math_modules.py`，仅依赖标准库 `unittest` 和
项目已有的 PyTorch：

```bash
conda activate sigmarl-nod
python -m unittest tests.opinion.test_m3_math_modules
```

测试覆盖：

- pair feature 交换两次恢复原值；
- 正反车辆对证据符号相反；
- `b` 满足幅值和物理门控；
- EvidenceNet 参数能收到有限梯度；
- Dynamics 没有可训练参数；
- 正负证据产生相反意见；
- inactive edge 只执行衰减；
- 恒定有界输入不超过理论最终界；
- residual 有界、归一化且无有效边时为零；
- 只修改速度 loc，不修改转向 loc。

## 7. 完整训练与测试仍可执行

pilot：

```bash
python main_training_opinion.py --config configs/opinion/pilot.json
python main_testing_opinion.py --config configs/opinion/pilot.json
```

完整预算：

```bash
python main_training_opinion.py
python main_testing_opinion.py
```

这些命令在 M3 仍要求：

```json
"stage": "base",
"use_opinion_marl": false
```

所以 M3 不应带来 reward 或 collision 提升，也不应产生额外训练耗时。此处验证的是
“加入数学模块文件没有破坏 Base 训练闭环”。

## 8. 后续衔接

M4 已在 `scenarios/road_traffic.py` 增加显式门控，并在开启时额外构造：

```text
pair_features, neighbor_ids, pair_mask,
urgency, confidence, agent_reset_mask
```

具体合同与命令见 [`M4_CONFLICT_GRAPH.md`](M4_CONFLICT_GRAPH.md)。M4 仍不让这些
信息进入 Actor，因此训练性能仍应接近 Base；真实动作改变从 M5 开始。
