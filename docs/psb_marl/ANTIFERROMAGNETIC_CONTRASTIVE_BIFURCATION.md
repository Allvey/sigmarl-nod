# AF-PSB：无通信反铁磁对比分岔设计草案

> 状态：P5 之后的候选设计，尚未实现，不修改当前 P5 训练合同。  
> 目标：在无车间通信的 CTDE 框架中，让冲突双方在角色空间形成稳定反对齐，在合作责任空间形成互补分配，并在物理通过时间上产生可验证的分离。  
> 暂定名称：**AF-PSB**（Antiferromagnetic Proximal Saturating Bifurcation）。

## 1. 动机

当前 PSB 使用交换评分差构造有界反对称控制，并以单个无序边状态表达双方的相反视角。这是可靠的一致性约束，但“反对称”本身不足以构成主要创新，也不能独立证明两车最终产生了不同的物理行为。

一个更有物理意义的方向，是把冲突双方视为一对反铁磁耦合的角色序参量：同号角色具有较高能量，异号角色自然成为低能稳定态。进一步将网络表示拆成公共场景子空间和反对齐角色子空间，并在动作结果层约束预计通过时间分离，从而建立：

```text
公共冲突事实保持一致
          +
角色表示自发反对齐
          +
合作责任形成互补
          +
实际通过时间产生间隔
```

本方案不使用“车辆 A 应主动、车辆 B 应让行”之类的监督标签。角色方向由长期任务回报、冲突物理量和对称破缺共同决定。

## 2. 无通信执行合同

本文采用 CTDE：训练阶段可以使用同一环境中多个智能体的联合张量计算 Critic 和成对自监督损失；执行阶段每辆车只能访问自己的局部观测、由本车传感器构造的相对物理特征，以及本车保存的循环状态。

### 2.1 执行阶段允许的信息

- 自车位置、速度、航向和道路几何；
- 传感器测得的邻车相对位置、相对速度和相对航向；
- 可由上述信息解析计算的冲突强度、预计到达冲突点时间和制动裕度；
- 共享但固定的网络参数、车辆动力学和地图规则；
- 本车维护的历史角色或意见状态。

### 2.2 执行阶段禁止的信息

- 另一辆车发送的角色状态、隐特征、动作意图或策略输出；
- 中央 Critic、全局训练缓存或联合环境真值；
- 测试时集中式角色分配器；
- 依赖 agent ID 的固定优先级；
- 未在问题定义中公开的公共随机种子。

### 2.3 保证边界

若双方观测严格互易，交换后的成对特征满足

\[
\chi_{ji}=\mathcal S\chi_{ij},
\]

且双方使用相同的确定性网络，则可以在结构上得到精确的交换等变或反等变输出。若存在独立传感噪声、遮挡、异步采样或不同邻居集合，则只能期望近似反对齐。

在完全对称、无通信、无公共随机源且没有任何公共物理约定的场景中，确定性策略无法决定谁先行。这是信息结构的不可辨识性，不应包装为可由损失函数消除的工程问题。

## 3. 公共特征与角色特征分解

不应把双方的完整特征表示全部推远。双方对冲突位置、相对速度和风险强度等公共事实应形成一致表征；只有角色相关分量应反对齐。

对有向局部观测 \(\chi_{ij}\)，定义：

\[
c_{ij}=C_\theta(\chi_{ij}),
\qquad
r_{ij}=R_\theta(\chi_{ij}),
\]

其中：

- \(c_{ij}\in\mathbb R^{d_c}\) 是公共冲突表示；
- \(r_{ij}\in\mathbb R^{d_r}\) 是角色表示；
- \(q_{ij}=w_q^\top r_{ij}\) 是标量角色序参量。

理想交换关系为：

\[
c(\mathcal S\chi)=c(\chi),
\qquad
r(\mathcal S\chi)=-r(\chi),
\qquad
q(\mathcal S\chi)=-q(\chi).
\]

可使用交换群 \(C_2\) 的对称和反对称投影实现该分解：

\[
c(\chi)
=
\frac{F_c(\chi)+F_c(\mathcal S\chi)}{2},
\]

\[
r(\chi)
=
\frac{F_r(\chi)-F_r(\mathcal S\chi)}{2}.
\]

这是一种网络参数化，不需要角色标签。每辆车都可以仅凭本地成对观测同时计算 \(\chi\) 与其交换变换 \(\mathcal S\chi\)，不涉及车间通信。

## 4. 反铁磁角色能量

对冲突边 \(e=\{i,j\}\) 的两端角色序参量，定义：

\[
\boxed{
\mathcal E_{ij}(q_i,q_j;\rho)
=
\frac{\beta}{4}(q_i^4+q_j^4)
-\frac{\mu(\rho)}{2}(q_i^2+q_j^2)
+\frac{\lambda_{\mathrm{af}}}{2}(q_i+q_j)^2
}
\]

其中 \(\beta>0\)、\(\lambda_{\mathrm{af}}>0\)，\(\mu(\rho)\) 随物理冲突强度跨过零点，例如：

\[
\mu(\rho)=k_\rho(\rho-\rho_c).
\]

### 4.1 无冲突阶段

当 \(\rho<\rho_c\) 时，\(\mu(\rho)<0\)，能量倾向唯一中性状态：

\[
q_i=q_j=0.
\]

系统无需提前制造角色差异。

### 4.2 冲突激活阶段

当 \(\rho>\rho_c\) 时，\(\mu(\rho)>0\)。在反对称子空间 \(q_i=-q_j=q\) 上：

\[
\mathcal E_{ij}(q,-q)
=
\frac{\beta}{2}q^4-\mu(\rho)q^2.
\]

稳定极小值为：

\[
q^\star
=
\pm\sqrt{\frac{\mu(\rho)}{\beta}},
\]

对应两种等价的互补联合角色：

\[
(q_i,q_j)=(+q^\star,-q^\star)
\quad\text{或}\quad
(-q^\star,+q^\star).
\]

在同号子空间 \(q_i=q_j=q\) 上：

\[
\mathcal E_{ij}(q,q)
=
\frac{\beta}{2}q^4
+(2\lambda_{\mathrm{af}}-\mu(\rho))q^2.
\]

选择 \(2\lambda_{\mathrm{af}}>\mu_{\max}\) 可抑制同号非零稳定分支，使“双方都主动”或“双方都让行”不再是能量偏好的角色组合。

### 4.3 与当前 PSB 的关系

当前 PSB 的单边双稳态负责角色承诺、迟滞和恢复；反铁磁项负责两端角色之间的排斥。后续实现不应立即替换现有近端层，而应先将反铁磁能量作为独立候选模块和消融项验证。

如果最终仍只维护一个无序边标量 \(z_e\)，并令 \((q_i,q_j)=(z_e,-z_e)\)，反铁磁约束会退化为已有的结构反对称。因此，本方案的新增价值必须来自“无通信双端局部推断在噪声下是否能通过能量训练形成稳定反对齐”，而不能只来自公式改写。

## 5. 合作责任映射

使用互补 sigmoid 映射：

\[
\alpha_i=\sigma(q_i/T_\alpha),
\qquad
\alpha_j=\sigma(q_j/T_\alpha).
\]

当 \(q_j=-q_i\) 时自动得到：

\[
\boxed{\alpha_i+\alpha_j=1.}
\]

这里 \(\alpha\) 不应被直接解释为固定的“加速概率”。它表示冲突责任或主动程度的连续分配，具体动作仍由分支条件 Actor 根据局部物理状态学习。

合作互补误差：

\[
e_\alpha=|\alpha_i+\alpha_j-1|
\]

优先作为诊断指标。如果 \(q\) 已结构反对称，不应再加入数学上冗余的强监督损失。

## 6. 从特征分离到物理行为分离

仅让隐特征远离可能产生退化解：网络编码车辆 ID、观测噪声或无关方向，但双方动作仍然相同。因此必须测量并约束物理结果。

令 \(\widehat\tau_i^{\mathrm{pass}}\) 表示在当前动作分布或短时动力学展开下，车辆 \(i\) 预计通过共享冲突区的时间。对可行且活跃的冲突边定义：

\[
\mathcal L_{\mathrm{pass}}
=
m_{ij}
\left[
m_\tau
-
\left|
\widehat\tau_i^{\mathrm{pass}}
-
\widehat\tau_j^{\mathrm{pass}}
\right|
\right]_+^2.
\]

该损失只要求形成安全时间间隔，不预先指定谁先通过。它必须与碰撞风险、车道约束和任务回报共同使用，不能通过鼓励一方危险加速来单独最小化。

如果短时通过时间不可稳定求导，首版可将其作为诊断而非训练损失，先记录真实 rollout 中的：

\[
\Delta\tau_{ij}^{\mathrm{real}}
=
|\tau_i^{\mathrm{exit}}-\tau_j^{\mathrm{exit}}|.
\]

## 7. 训练目标

建议将第一版总目标写为：

\[
\mathcal L
=
\mathcal L_{\mathrm{PPO}}
+\lambda_V\mathcal L_V
+\lambda_E\mathcal L_{\mathrm{AF}}
+\lambda_c\mathcal L_{\mathrm{common}}
+\lambda_r\mathcal L_{\mathrm{role}}
+\lambda_\tau\mathcal L_{\mathrm{pass}}
+\mathcal L_{\mathrm{existing\ safety/reg}}.
\]

其中：

\[
\mathcal L_{\mathrm{AF}}
=
\mathbb E_{(i,j)\in\mathcal E_t}
\left[\mathcal E_{ij}(q_i,q_j;\rho_{ij})\right],
\]

\[
\mathcal L_{\mathrm{common}}
=
\mathbb E
\|c_{i\to j}-c_{j\to i}\|_2^2,
\]

\[
\mathcal L_{\mathrm{role}}
=
\mathbb E
\|r_{i\to j}+r_{j\to i}\|_2^2.
\]

这里的成对项是 CTDE 训练中的交换一致性正则，不使用人工角色标签。为防止表示坍缩，还需要对 \(c\) 和 \(r\) 使用有限范数、批次方差或协方差约束；具体采用 VICReg 式方差下界还是由双稳态能量提供非零幅值，应通过消融决定，不能重复施加强约束。

所有成对损失必须仅作用于有效冲突边，并使用固定全集或明确的边掩码归一化，避免车辆数量和图密度改变损失尺度。

## 8. 无通信前向路径

推荐的执行路径为：

```text
本车局部观测 o_i
      +
本车传感器生成的 chi_ij
      |
      v
交换等变编码器 -> 公共特征 c_ij + 角色特征 r_ij
      |
      v
本地角色状态/近端动力学 -> q_i
      |
      v
分支条件 Actor -> 本车动作 a_i
```

另一辆车独立执行同样流程。部署图中不得出现从车辆 \(j\) 的网络输出到车辆 \(i\) 的张量边。

训练时可以在 loss 模块中同时读取 \(r_{i\to j}\) 和 \(r_{j\to i}\)，但该读取不得进入 rollout Actor 的前向图。应增加自动化测试，验证删除其他智能体的隐状态键后，单车 Actor 仍能完整推理。

## 9. 与现有代码的预期接入点

以下仅是后续实施边界，不代表当前代码已经支持。

### 9.1 `utilities/psb_marl/p2_network.py`

新增候选模块：

- `ExchangeDecomposedPairEncoder`：输出 \(c_{ij},r_{ij},q_{ij}\)；
- `AntiferromagneticRoleEnergy`：计算有掩码的成对能量和诊断；
- 可选 `PassageTimeHead`：仅在能够给出明确物理标签或可微估计时启用。

现有 `AntisymmetricBifurcationControl` 保留，作为当前 PSB 基线和消融组。

### 9.2 `utilities/psb_marl/p2_policy.py`

扩展 bridge 输出，但不改变 Base Actor 的输入合同：

```text
common_feature
role_feature
role_scalar
cooperation_share
```

首版仍通过现有 `supported_sector_q_gate` 约束动作影响，避免新表示绕过冲突支持门。

### 9.3 `utilities/psb_marl/p2_loss.py`

在序列重算得到当前参数下的角色表示后计算 AF、公共一致性和角色反对齐损失。不得直接使用 rollout 缓存中的旧角色张量作为需要梯度的训练输出。

### 9.4 `utilities/psb_marl/config.py`

使用独立、默认关闭的配置块，不改变现有 P2/P3/P5 配置解析：

```json
{
  "anti_coordination": {
    "enabled": false,
    "role_dim": 0,
    "beta": 0.0,
    "mu_scale": 0.0,
    "coupling": 0.0,
    "energy_coefficient": 0.0,
    "common_coefficient": 0.0,
    "role_coefficient": 0.0,
    "passage_time_coefficient": 0.0
  }
}
```

正式实现时再确定正值范围和互斥条件；当前文档不预设未经实验验证的默认超参数。

### 9.5 训练器与评估器

- 沿用 P5 的联合 PPO、绝对 Critic、差分 Critic 和安全对偶框架；
- Source Base 继续只读并作为回退；
- 新 Candidate 必须通过相同 paired non-inferiority/superiority 协议；
- 新指标不得改变现有奖励或部署选择逻辑。

## 10. 必要诊断

至少记录：

```text
active_pair_common_alignment_error
active_pair_role_anti_alignment_error
active_pair_role_distance
active_pair_same_sign_fraction
active_pair_cooperation_complement_error
active_pair_passage_time_gap
active_pair_role_flip_rate
active_pair_commitment_duration
inactive_pair_role_norm
anti_coordination_energy
```

关键判据：

- 无冲突时角色范数应回到零附近；
- 活跃冲突中同号比例应下降；
- 角色距离增大必须伴随通过时间间隔增大或冲突解除改善；
- 不能以碰撞率、越界率或剧烈控制为代价制造表面上的角色分离；
- 传感噪声下的反对齐误差必须单独报告。

## 11. 必须设置的基线和消融

| 方法 | 角色互补 | 动态记忆 | 反铁磁能量 | 物理时间分离 |
|---|---:|---:|---:|---:|
| Base PPO | 否 | 原策略决定 | 否 | 否 |
| DirectRole | 结构反对称 | 否 | 否 | 否 |
| 当前 PSB | 共享边反对称 | 是 | 否 | 否 |
| AF-Feature | 近似反对齐 | 可选 | 是 | 否 |
| AF-Feature + Pass | 近似反对齐 | 可选 | 是 | 是 |
| AF-PSB Full | 近似反对齐 | PSB 迟滞 | 是 | 是 |

DirectRole 基线应复用相同编码容量和分支 Actor，但跳过近端动力学。若它与完整 AF-PSB 表现相同，则不能声称分岔记忆具有经验必要性。

## 12. 分阶段验证顺序

后续实现建议按以下顺序推进，每一步失败都应停止扩展：

1. **离线代数验证**：交换观测后公共分量不变、角色分量变号；
2. **能量相图验证**：数值验证中性态、反对称双稳态和同号分支抑制条件；
3. **只读 rollout 诊断**：不影响动作，只测量表示反对齐和噪声误差；
4. **冻结 Base 训练**：只训练角色模块和现有受限 Adapter；
5. **与 DirectRole/当前 PSB 比较**：确认新增动力学具有可测收益；
6. **联合训练**：仅在前述检查通过后解冻 Candidate Base；
7. **多种子正式评估**：报告性能、安全、稳定性和表示指标的联合结果。

## 13. 理论与工程风险

1. **无意义特征分离**：角色特征可能编码 ID 或噪声；必须用物理通过时间和任务收益验证。
2. **局部观测不互易**：独立传感噪声会破坏精确反对齐；需要报告误差而非宣称保证。
3. **同号分支未完全消失**：若 \(2\lambda_{\mathrm{af}}\le\mu_{\max}\)，可能保留不期望的同号稳定态。
4. **损失相互竞争**：过强反对齐会压过 PPO 和安全目标，制造僵硬角色。
5. **多车图不一致**：双端反对齐不自动保证复杂场景中的全局通行最优；若需要无环性，应另行引入可观测通行势或图 Hodge 约束。
6. **隐式通信误用**：训练代码若把另一车辆隐状态直接送入 Actor，则违反执行合同。
7. **完全对称不可辨识**：没有公共物理不对称时无法确定具体哪辆车进入正分支，只能保证两个分支统计上无偏。

## 14. 成功标准

只有同时满足以下条件，AF-PSB 才值得替换当前反对称策略：

1. 相比 DirectRole 和当前 PSB，显著减少角色翻转或协调振荡；
2. 角色/合作空间的分离对应更大的安全通过时间间隔，而非仅有隐空间距离；
3. 在相同安全预算下改善回报、冲突解除时间或跨场景泛化；
4. 在独立传感噪声和异步观测下保持可接受的近似反对齐；
5. 执行图中不存在跨车隐状态传递；
6. 新方法通过现有 Base-relative 成对验证和部署门。

若这些标准未满足，应保留更简单的 DirectRole 或当前 PSB，不因理论形式更复杂而默认采用 AF-PSB。

## 15. 一句话定位

> AF-PSB 将无通信多车协调建模为公共冲突表征上的反铁磁角色相变：共享事实在对称子空间对齐，角色序参量在反对称子空间分离，合作责任由互补映射生成，物理通过时间则验证这种隐空间分岔确实转化为可执行的时序协调。
