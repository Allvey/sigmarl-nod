# Proximal Saturating Bifurcation MARL：完整理论路线（证明级复核版）

> 目标：建立一套可以直接写入论文 Method、Theoretical Analysis 和 Appendix 的理论体系。  
> 核心定位：将多智能体协调中的离散模式选择提升为一个由长期任务价值驱动的、低维受控分岔最优控制问题。  
> 推荐简称：**PSB-MARL**（Proximal Saturating Bifurcation MARL）。  
> 本版复核原则：严格区分连续系统与离散实现、静态灵敏度与有限时域可控性、理想最优控制必要条件与PPO实际更新、确定性迟滞与随机鲁棒性。

全文采用下列主张等级：

- **定理**：在明确假设下可由本文公式直接证明；
- **条件性命题**：还需要局部曲率、慢变、公共边状态等附加条件；
- **算法解释**：说明结构为什么合理，但不等价于PPO收敛定理；
- **经验假设**：必须由消融或数值实验验证，不能包装成理论保证。

---

## 0. 理论主张

本文不把非线性意见动力学当作滤波器、平滑器或普通循环记忆，而是把它作为多智能体增广最优控制系统中的低维协调子系统：

\[
\boxed{
\begin{aligned}
\text{物理冲突几何} &\longrightarrow \text{分岔临界性};\\
\text{MARL} &\longrightarrow \text{能量正则化分岔控制};\\
\text{饱和型NOD} &\longrightarrow \text{协调模式的生成、保持与消失};\\
\text{连续Actor} &\longrightarrow \text{分支内物理控制};\\
\text{增广Critic} &\longrightarrow \text{联合长期价值评估}.
\end{aligned}}
\]

其数学本质是：

\[
\boxed{
\text{用连续吸引子选择替代离散模式枚举，}
\text{再由RL求解吸引子选择与分支内控制的联合最优策略。}
}
\]

完整方法由四个不能互相替代的对象构成：

1. 受控饱和势能，规定协调模式的几何结构；
2. 近端离散算子，规定该结构在离散时间中的稳定演化；
3. 分岔控制网络，学习任务相关的能量正则化选支输入；
4. 分支条件策略与增广价值函数，在受限分散策略类中优化长期联合回报。

---

## 1. 多智能体协调的原始最优控制问题

考虑 \(N\) 个智能体组成的随机动态系统：

\[
x^{t+1}\sim P(\cdot\mid x^t,a^t),
\qquad
a^t=(a_1^t,\ldots,a_N^t),
\]

其中 \(x^t\in\mathcal X\) 是全局物理状态，\(a_i^t\in\mathcal A_i\) 是智能体 \(i\) 的连续控制输入。执行阶段智能体只能获得局部观测：

\[
o_i^t=O_i(x^t).
\]

当前潜在冲突关系形成无向图：

\[
\mathcal G_t=(\mathcal V,\mathcal E_t),
\qquad
\mathcal V=\{1,\ldots,N\}.
\]

对每条冲突边 \(e=\{i,j\}\)，至少存在两种局部协调模式：

\[
\mathcal M_e^+: i\succ j,
\qquad
\mathcal M_e^-: j\succ i.
\]

如果直接使用离散变量：

\[
\sigma_e\in\{-1,+1\},
\]

则原问题成为混合离散—连续随机最优控制：

\[
\min_{\pi,\boldsymbol\sigma}
\mathbb E
\left[
\sum_{t=0}^{\infty}\gamma^t
\ell_{\mathrm{task}}
(x^t,a^t,\boldsymbol\sigma^t)
\right].
\]

该形式存在三个结构困难：

1. \(\boldsymbol\sigma\) 不可微，难以直接接受连续策略梯度；
2. 多边同时冲突时，模式组合随 \(|\mathcal E_t|\) 指数增长；
3. 缺少动态承诺机制时，最优模式可能因观测噪声在相邻时刻频繁翻转。

因此本文不直接优化 \(\sigma_e\)，而对模式变量进行分岔提升。

---

## 2. 从离散模式到连续分岔状态

### 2.1 分岔提升

为每条无序冲突边 \(e=\{i,j\}\) 指定一个仅用于数学记号和状态存储的方向，并维护标量：

\[
z_e\in\mathbb R.
\]

其语义为：

\[
z_e=0
\Longleftrightarrow
\text{协调未决},
\]

\[
z_e>0
\Longleftrightarrow
\mathcal M_e^+,
\qquad
z_e<0
\Longleftrightarrow
\mathcal M_e^-.
\]

离散模式可以在远离零点时由：

\[
\sigma_e=\operatorname{sign}(z_e)
\]

恢复，但训练和控制始终在连续变量 \(z_e\) 上进行。

### 2.2 增广状态

令：

\[
\mathbf z=(z_e)_{e\in\mathcal E},
\]

则增广状态为：

\[
\widetilde x=(x,\mathbf z).
\]

物理状态描述车辆或机器人的位置、速度等；分岔状态描述当前已经形成的协调承诺。相同的物理状态 \(x\) 与不同的 \(\mathbf z\) 对应不同的未来动作分布，因此 \(\mathbf z\) 不是可视化变量，而是系统状态的一部分。

---

## 3. 受控饱和型意见势能

### 3.1 完整势能

对边 \(e\)，定义：

\[
\boxed{
\mathcal U_e(z_e;\rho_e,b_e)
=
\frac{\kappa}{2}z_e^2
-
\frac{\rho_e\nu}{\alpha}
\log\cosh(\alpha z_e)
-b_ez_e.
}
\]

其中：

- \(\kappa>0\)：意见遗忘或恢复强度；
- \(\nu>0\)：非线性自强化强度；
- \(\alpha>0\)：非线性敏感度；
- \(\rho_e\in[0,\rho_{\max}]\)：由物理交互确定的冲突强度；
- \(b_e\in[-b_{\max},b_{\max}]\)：由MARL学习的分岔控制。

对当前冲突图，核心模型采用可分离总势能：

\[
\mathcal U_{\mathcal G}(\mathbf z;\boldsymbol\rho,\mathbf b)
=
\sum_{e\in\mathcal E}
\mathcal U_e(z_e;\rho_e,b_e).
\]

边之间并非完全独立：它们通过共享物理状态、联合Actor、联合回报和中央Critic耦合；可分离性只表示每条边的局部分岔几何具有清晰的一维结构。

### 3.2 连续动力学

令意见沿势能梯度下降：

\[
\boxed{
\tau_z\dot z_e
=
-\frac{\partial\mathcal U_e}{\partial z_e}
=
-\kappa z_e
+\rho_e\nu\tanh(\alpha z_e)
+b_e,
}
\]

其中 \(\tau_z>0\) 是意见时间尺度。

各项含义为：

\[
-\kappa z
=
\text{冲突解除后的中性恢复},
\]

\[
\rho\nu\tanh(\alpha z)
=
\text{冲突激活的饱和自强化},
\]

\[
b
=
\text{选择协调分支的外部控制}.
\]

把 \(b\) 放在 \(\tanh\) 外部非常重要，因为它对应势能中的精确线性倾斜 \(-bz\)，从而具有明确的控制方向、幅值和能量含义。

### 3.3 基本假设

后续理论采用以下假设。

**A1 参数正性与有界输入。**

\[
\kappa,\nu,\alpha,\tau_z>0,
\quad
\rho_e\in[0,\rho_{\max}],
\quad
|b_e|\le b_{\max}.
\]

**A2 物理临界性。** \(\rho_e=R(\chi_e)\) 是成对物理特征 \(\chi_e\) 的外生风险映射。它必须满足：

\[
R(\mathcal S\chi_{ij})=R(\chi_{ij}),
\qquad
0\le R(\chi)\le\rho_{\max},
\]

即交换两智能体不改变冲突强度；它只能使用物理几何、运动学和有效掩码，不得读取 \(z\)、\(b\)、智能体ID或人工优先标签。固定交互状态分析中把 \(\rho\) 视为常数，动态分析中要求其分段绝对连续；边创建/删除时允许有限次跳变，并按混杂系统单独处理。

一个合格的实现例子是先由相对位置、相对速度计算软化的最近接近时间 \(t_{\mathrm{ca}}\) 和最近距离 \(d_{\mathrm{ca}}\)，再令：

\[
\rho_{ij}
=m_{ij}\rho_{\max}
\,\operatorname{sigmoid}\!\left(\frac{r_{\mathrm{safe}}-d_{\mathrm{ca}}}{s_d}\right)
\operatorname{sigmoid}\!\left(\frac{T_h-t_{\mathrm{ca}}}{s_t}\right),
\]

其中 \(m_{ij}=m_{ji}\) 是候选边掩码。该例子不是理论唯一选择；真正必要的是有界性、交换不变性、外生性和可校准的阈值穿越。

**A3 边状态一致性。** 理论主模型对每条无序边维护唯一状态。两端读取相反符号：

\[
z_{ji}=-z_{ij}.
\]

**A4 边生命周期完备性。** 动态冲突图的边创建、保留、失效和重置规则是当前物理状态、当前边注册表与当前随机输入的函数。边注册表、有效掩码以及尚未消费的对称破缺随机变量都被视为增广状态的一部分。

**A5 增广马尔可夫性。** 环境的下一物理状态只依赖当前物理状态和联合动作；意见下一状态只依赖当前意见、当前物理临界性和当前分岔控制。为避免变维状态，理论上可在固定全集 \(\mathcal E_\star=\{\{i,j\}:i<j\}\) 上维护所有边，非活跃边取 \(m_e=0,\rho_e=b_e=0\) 并继续衰减。若使用动态边表或Actor还包含GRU，其注册表、随机种子和额外记忆也必须并入增广状态。

**A6 有界回报。** \(|r_t|\le r_{\max}\)，且 \(0<\gamma<1\)。

**A7 因果时序与状态反馈。** 每一步严格采用

\[
(x^t,z^t)\to(\rho^t,b^t)\to z^{t+1}\to a^t\to x^{t+1},
\]

其中 \(b^t=b_\phi(o^t,z^t,m^t)\)。让分岔控制读取当前意见状态不是装饰：如果静态物理特征相同而 \(b_\phi\) 不读 \(z\)，它无法区分“尚未选支”和“已经承诺”，因而一般不能在选支后主动撤去控制能量。

同时，本文采用严格的采样-保持语义：\(b^t\) 只用旧状态 \(z^t\) 计算一次，在求解整个近端子问题时视为常数；Newton/二分内部不得用候选 \(z\) 重新计算 \(b_\phi\)。因此第6–9节是冻结输入相图，第14节处理跨步反馈造成的参数注入。若在隐式求根内使用 \(b_\phi(z)\)，则闭环曲率变为 \(\mathcal U''-\partial b_\phi/\partial z\)，本文的强凸、折叠阈值和Jacobian公式都不再原样成立。

本文公式中的Actor读取更新后的 \(z^{t+1}\)。若实现改为读取 \(z^t\)，则 \(b^t\) 只能影响未来动作，策略梯度路径与Bellman索引必须整体后移一拍，不能混用两种时序。

**A8 理论层与算法层分离。** Pontryagin与Bellman分析针对“\(a,b\) 可作为理想双控制量联合优化”的增广控制问题；PPO实现则把 \(b_\phi\) 限制为局部观测驱动的确定性内部控制。前者给出必要条件与结构解释，后者只求受限策略类中的局部驻点，二者不被宣称为严格等价。

**A9 精确算子与数值求解器分离。** 定理首先针对精确近端映射 \(P_{h_z}\)。有限次Newton迭代只产生近似解 \(\widehat P_{h_z}\)，必须以残差容差给出误差界；不能把精确能量不等式和精确隐式Jacobian无条件套到未收敛的数值迭代上。

---

## 4. 无量纲化与关键控制参数

定义无量纲状态、时间、临界性和控制：

\[
y=\alpha z,
\qquad
s=\frac{\kappa}{\tau_z}t,
\]

\[
\lambda
=
\frac{\rho\nu\alpha}{\kappa},
\qquad
\beta
=
\frac{\alpha b}{\kappa}.
\]

则动力学化为：

\[
\boxed{
\frac{dy}{ds}
=
-y+\lambda\tanh y+\beta.
}
\]

这说明所有分岔结构主要由两个无量纲量决定：

- \(\lambda\)：冲突驱动的相变参数；
- \(\beta\)：MARL施加的无量纲势能倾斜。

临界冲突强度为：

\[
\boxed{
\rho_c
=
\frac{\kappa}{\nu\alpha},
}
\]

即：

\[
\lambda=1
\Longleftrightarrow
\rho=\rho_c.
\]

若 \(\rho\in[0,1]\)，应选择：

\[
0<\rho_c<1
\quad\Longleftrightarrow\quad
\nu\alpha>\kappa.
\]

---

## 5. 定理一：势能强制性与意见有界性

### 定理1：势能强制性

在A1下，对任意固定 \((\rho,b)\)，有：

\[
\lim_{|z|\to\infty}
\mathcal U(z;\rho,b)
=+\infty.
\]

因此势能至少存在一个全局极小点。

### 证明

因为：

\[
\log\cosh(\alpha z)
\le
\alpha|z|,
\]

所以：

\[
\mathcal U(z;\rho,b)
\ge
\frac\kappa2z^2
-\rho\nu|z|
-|b||z|.
\]

右侧是正二次项减去线性项，因此随 \(|z|\to\infty\) 趋于正无穷。证毕。

### 推论1：连续意见状态最终有界

由：

\[
\tau_z\dot z
=
-\kappa z+\rho\nu\tanh(\alpha z)+b
\]

和 \(|\tanh|\le1\)，可得：

\[
\tau_z\frac{d|z|}{dt}
\le
-\kappa|z|
+\rho_{\max}\nu+b_{\max}.
\]

因此：

\[
\boxed{
\limsup_{t\to\infty}|z(t)|
\le
\frac{\rho_{\max}\nu+b_{\max}}{\kappa}.
}
\]

该结果是全局的，不依赖临界点附近的局部展开。

### 推论2：冻结输入下的连续能量耗散

当 \((\rho,b)\) 固定时：

\[
\frac{d\mathcal U}{dt}
=
\frac{\partial\mathcal U}{\partial z}\dot z
=
-\frac1{\tau_z}
\left|
\frac{\partial\mathcal U}{\partial z}
\right|^2
\le0.
\]

因此 \(\mathcal U\) 是连续意见子系统的Lyapunov函数。结合势能强制性和一维梯度系统性质，所有轨迹均保持有界，并趋向平衡点集合；除初始状态恰处于不稳定平衡点或其稳定流形等退化情况外，轨迹趋向某个稳定极小点。

---

## 6. 定理二：无偏系统的超临界分岔

令 \(b=0\)，考虑无量纲动力学：

\[
\dot y=-y+\lambda\tanh y.
\]

平衡点满足：

\[
y=\lambda\tanh y.
\]

### 定理2：单稳态—双稳态相变

1. 当 \(0\le\lambda\le1\) 时，唯一平衡点为 \(y^\star=0\)；当 \(\lambda<1\) 时其为双曲全局渐近稳定点，在 \(\lambda=1\) 时为非双曲渐近稳定点。
2. 当 \(\lambda>1\) 时，恰有三个平衡点：

\[
y_0^\star=0,
\qquad
y_-^\star<0<y_+^\star,
\qquad
y_-^\star=-y_+^\star.
\]

其中 \(y_0^\star\) 不稳定，\(y_\pm^\star\) 局部渐近稳定。

### 证明要点

定义：

\[
g(y)=\lambda\tanh y-y.
\]

有：

\[
g(0)=0,
\qquad
g'(y)=\lambda\operatorname{sech}^2y-1.
\]

当 \(\lambda\le1\) 时，对所有 \(y\ne0\)：

\[
g'(y)<0
\]

或在 \(\lambda=1,y=0\) 处取零，因此只有零根。又因为 \(y>0\) 时 \(\tanh y<y\)，故向量场指向零点。

当 \(\lambda>1\) 时：

\[
g'(0)=\lambda-1>0,
\]

但：

\[
\lim_{y\to+\infty}g(y)=-\infty.
\]

并且对 \(y>0\)：

\[
g''(y)
=
-2\lambda\operatorname{sech}^2y\tanh y<0,
\]

所以正半轴上恰有一个非零正根，负半轴由奇对称性得到一个负根。在线性化中：

\[
f'(y^\star)
=
-1+\lambda\operatorname{sech}^2y^\star.
\]

零点处 \(f'(0)=\lambda-1>0\)，故失稳；两个外侧根位于 \(g'\) 变为负值之后，故稳定。证毕。

### 物理解释

\[
\boxed{
\begin{cases}
\rho<\rho_c:& \text{未决中性模式};\\
\rho>\rho_c:& \text{两个稳定通行次序模式}.
\end{cases}}
\]

因此冲突强度并不是意见输入，而是控制协调模式是否存在的相参数。

---

## 7. 与pitchfork normal form的严格局部联系

### 7.1 势能展开

在 \(z=0\) 附近：

\[
\log\cosh(\alpha z)
=
\frac{\alpha^2z^2}{2}
-\frac{\alpha^4z^4}{12}
+O(z^6).
\]

代入完整势能：

\[
\mathcal U(z;\rho,b)
=
\frac{\kappa-\rho\nu\alpha}{2}z^2
+\frac{\rho\nu\alpha^3}{12}z^4
-bz
+O(z^6).
\]

对应动力学为：

\[
\tau_z\dot z
=
(\rho\nu\alpha-\kappa)z
-\frac{\rho\nu\alpha^3}{3}z^3
+b
+O(z^5).
\]

定义：

\[
\mu=\rho\nu\alpha-\kappa,
\qquad
c=\frac{\rho\nu\alpha^3}{3}>0,
\]

则：

\[
\tau_z\dot z
=
\mu z-cz^3+b+O(z^5).
\]

在冻结 \(\rho>0\) 并忽略五阶余项时，取

\[
u=\sqrt{c}\,z,
\qquad
\theta=\frac{t}{\tau_z},
\qquad
\widetilde b=\sqrt c\,b,
\]

得到未完全归一化但系数明确的规范形：

\[
\boxed{
\frac{du}{d\theta}
=
\mu u-u^3+\widetilde b
+O(u^5/c^2).
}
\]

进一步缩放时间和状态即可得到标准的受控pitchfork。这里的局部等价仅在 \(|\alpha z|\ll1\) 且参数位于临界邻域时成立；它不能替代完整 \(\tanh\) 模型的全局有界性分析。故pitchfork不是另一个人为模型，而是完整饱和NOD在临界点附近的中心流形/规范形描述。

### 7.2 临界分支幅值

令 \(\lambda=1+\epsilon\)，\(0<\epsilon\ll1\)。由：

\[
y=\lambda\left(y-\frac{y^3}{3}+O(y^5)\right)
\]

得到非零分支：

\[
y_\pm^\star
=
\pm
\sqrt{\frac{3(\lambda-1)}{\lambda}}
+O((\lambda-1)^{3/2}).
\]

即：

\[
z_\pm^\star
=
\pm\frac1\alpha
\sqrt{\frac{3(\lambda-1)}{\lambda}}
+O((\lambda-1)^{3/2}).
\]

这给出了典型的平方根临界标度，是超临界pitchfork的主要可验证特征。

### 7.3 临界减速：高灵敏度不等于快速决策

无量纲向量场在平衡点的局部恢复率为：

\[
r(y^\star)
=
1-\lambda\operatorname{sech}^2y^\star.
\]

稳定点满足 \(r>0\)，其无量纲线性恢复时间约为 \(T_{\mathrm{rel}}=1/r\)。当 \(\beta=0\) 时：

\[
T_{\mathrm{rel}}
\sim
\begin{cases}
(1-\lambda)^{-1}, & \lambda\uparrow1,\\[2mm]
[2(\lambda-1)]^{-1}, & \lambda\downarrow1\text{ from above}.
\end{cases}
\]

在临界点 \(\lambda=1\)、\(\beta=0\) 上，线性恢复率为零，局部方程为：

\[
\frac{dy}{ds}=-\frac{y^3}{3}+O(y^5),
\]

因此收敛是代数型而非指数型。若 \(\lambda=1\) 且施加小常值 \(\beta\ne0\)，则：

\[
y^\star\sim(3\beta)^{1/3},
\qquad
T_{\mathrm{rel}}\sim(3|\beta|)^{-2/3}.
\]

所以临界点同时带来两件相反的事：静态增益变大，但响应变慢。本文只能严格声称“临界附近静态选支灵敏度高”；“在固定决策时限内控制能量最小”必须把到达时间、噪声和目标置信度写入有限时域优化后再证明或实验验证。

### 7.4 与AVOCADO动力学的关系和区别

AVOCADO原文Eq. (10)的成对适应律为：

\[
\dot x_i
=
-d_ix_i
+d_iA_i\tanh(a_ix_i+c_iy_i)
+b_i.
\]

其中 \(A_i\) 由碰撞时间驱动的独立动态方程更新，\(y_i\) 由速度变化投影估计，原文并明确使用前向Euler离散。它与本文共享“线性遗忘、注意力激活、饱和非线性和分岔”这一NOD母结构，但变量和优化角色不同：

- AVOCADO中的 \(x_i\) 估计对方合作程度，本文中的 \(z_{ij}\) 表示双方通行次序；
- AVOCADO中的 \(y_i\) 是几何投影估计，本文不构造人工通行证据标签；
- AVOCADO中的 \(b_i\) 是先验偏置，本文中的 \(b_{\phi,ij}\) 是由长期联合回报学习的控制；
- AVOCADO把意见映射到Velocity-Obstacle约束，本文把意见作为分支条件输入连续Actor；
- AVOCADO采用显式Euler更新，本文采用可微近端算子并分析单步适定性。

因此本文借用的是饱和非线性意见系统的动力学类别，而不是复制AVOCADO的状态语义、几何估计器或优化程序。特别地，AVOCADO的 \(y_i\) 位于 \(\tanh\) 内，而本文把学习控制放在 \(\tanh\) 外，从而保留精确的线性势能倾斜 \(-bz\)。故本文的分岔阈值、势能和近端定理都是针对新模型重新推导，不是从AVOCADO定理直接继承。

---

## 8. 定理三：受控非完美分岔与迟滞区

当 \(b\ne0\) 时，对称pitchfork变成非完美分岔。无量纲平衡方程为：

\[
F(y;\lambda,\beta)
=
y-\lambda\tanh y-\beta
=0.
\]

### 8.1 折叠边界

平衡点发生鞍结折叠时，同时满足：

\[
F(y;\lambda,\beta)=0,
\qquad
\frac{\partial F}{\partial y}=0.
\]

第二个条件给出：

\[
1-\lambda\operatorname{sech}^2y=0.
\]

当 \(\lambda>1\) 时：

\[
y_f^\pm
=
\pm\operatorname{arcosh}\sqrt\lambda.
\]

代回平衡方程：

\[
\beta_f
=
y_f-\lambda\tanh y_f.
\]

定义正的切换阈值：

\[
\boxed{
\beta_{\mathrm{sw}}(\lambda)
=
\sqrt{\lambda(\lambda-1)}
-\operatorname{arcosh}\sqrt\lambda,
\qquad\lambda>1.
}
\]

两条折叠曲线的符号对应为：

\[
\beta_f(y_f^-)=+\beta_{\mathrm{sw}},
\qquad
\beta_f(y_f^+)=-\beta_{\mathrm{sw}}.
\]

因此正控制消除负井，负控制消除正井。令 \(\epsilon=\lambda-1\downarrow0\)，精确阈值的临界展开为：

\[
\boxed{
\beta_{\mathrm{sw}}(1+\epsilon)
=
\frac23\epsilon^{3/2}
-\frac15\epsilon^{5/2}
+O(\epsilon^{7/2}).
}
\]

这与局部三次规范形的 \(3/2\) 次阈值标度一致。

展开可由恒等式 \(\operatorname{arcosh}\sqrt{1+\epsilon}=\operatorname{arsinh}\sqrt\epsilon\) 直接得到，因此不是对数值曲线的经验拟合。
对应有量纲阈值为：

\[
\boxed{
b_{\mathrm{sw}}(\rho)
=
\frac\kappa\alpha
\beta_{\mathrm{sw}}
\left(
\frac{\rho\nu\alpha}{\kappa}
\right).
}
\]

### 定理3：平衡点数量

完整相图需要先补充单稳态侧：当 \(0\le\lambda\le1\) 时，对任意 \(\beta\in\mathbb R\)，\(F_y=1-\lambda\operatorname{sech}^2y\ge0\)，且 \(F(y)\to\pm\infty\) 当 \(y\to\pm\infty\)，因而恰有一个平衡点。除 \((\lambda,\beta,y)=(1,0,0)\) 外该根为双曲稳定点；例外情形为非双曲渐近稳定点。

当 \(\lambda>1\) 时：

\[
|\beta|<\beta_{\mathrm{sw}}(\lambda)
\]

对应三个平衡点，其中两个稳定、一个不稳定；

\[
|\beta|=\beta_{\mathrm{sw}}(\lambda)
\]

对应一个简单根和一个二重根；

\[
|\beta|>\beta_{\mathrm{sw}}(\lambda)
\]

对应唯一稳定平衡点。

### 证明

有：

\[
\frac{\partial F}{\partial y}
=
1-\lambda\operatorname{sech}^2y.
\]

当 \(\lambda>1\) 时，该导数仅在 \(y_f^\pm\) 处为零；由于：

\[
\frac{\partial^2F}{\partial y^2}
=
2\lambda\operatorname{sech}^2y\tanh y,
\]

\(y_f^-<0\) 为局部极大点，\(y_f^+>0\) 为局部极小点。又有：

\[
\lim_{y\to-\infty}F(y)=-\infty,
\qquad
\lim_{y\to+\infty}F(y)=+\infty.
\]

两个驻点处的函数值相差一个由 \(\beta\) 控制的平移。当 \(|\beta|<\beta_{\mathrm{sw}}\) 时，局部极大值为正且局部极小值为负，因此按介值定理有三个简单根；等号时一个驻点与零轴相切，产生二重根；超过阈值后两个驻点位于零轴同侧，只剩一个简单根。稳定性由向量场 \(-F\) 的导数 \(-F_y\) 判定，外侧根稳定、中间根不稳定。证毕。

### 8.2 势能倾斜方向

对任意 \(z>0\)：

\[
\mathcal U(z;\rho,b)
-
\mathcal U(-z;\rho,b)
=
-2bz.
\]

因此：

\[
b>0
\Longrightarrow
\text{正分支势能更低},
\]

\[
b<0
\Longrightarrow
\text{负分支势能更低}.
\]

这给出分岔控制符号的严格语义。

### 8.3 迟滞解释

在 \(|b|<b_{\mathrm{sw}}\) 时，较高势能的一侧仍可能保持为局部稳定的亚稳态。系统不会因为微小的瞬时反向输入立即翻转；准静态切换通常需要：

1. 反向控制推动状态穿过不稳定分界点；或
2. 反向控制达到折叠阈值，使原稳定分支消失；或
3. 冲突强度降低到临界点以下，使双稳态整体消失。

这就是分岔动力学相对于线性滤波器的结构性抗抖动来源。

### 8.4 势垒与“抗噪”主张的边界

无量纲势能为：

\[
V(y;\lambda,\beta)
=
\frac12y^2-\lambda\log\cosh y-\beta y.
\]

当 \(\beta=0,\lambda=1+\epsilon\) 且 \(0<\epsilon\ll1\) 时，从稳定井底到中间马鞍点的势垒为：

\[
\boxed{
\Delta V
=
\frac{3}{4}\frac{\epsilon^2}{\lambda}
+O(\epsilon^3).
}
\]

证明只需使用：

\[
V(y;1+\epsilon,0)
=
-\frac\epsilon2y^2
+\frac\lambda{12}y^4
+O(y^6),
\qquad
(y_+^\star)^2
=
\frac{3\epsilon}{\lambda}
+O(\epsilon^2),
\]

并计算 \(V(0)-V(y_+^\star)\)。

所以刚越过分岔点时，虽然已经出现双稳态，势垒仍然很低。确定性系统中的折叠阈值只能严格支持“对有界准静态反向偏置存在滞后”；若意见动力学中还有持续噪声，不能仅凭 \(b_{\mathrm{sw}}\) 宣称不会误切换。

一个严格的局部充分条件是：在某稳定分支 \(z^\star\) 附近半径 \(r\) 内有 \(\mathcal U''(z)\ge m_s>0\)，并加性扰动满足 \(|d(t)|\le\delta_d<m_sr\)。则只要初值在该局部吸引域内，有局部输入-状态稳定界：

\[
\limsup_{t\to\infty}|z(t)-z^\star|
\le
\frac{\delta_d}{m_s}.
\]

若采用白噪声SDE，则需要用首达时间/Kramers分析估计跨势垒概率；这属于随机扩展，不在基础确定性定理之内。

---

## 9. 临界灵敏度、有限时域选支与能量正则化控制

稳定平衡点满足：

\[
H(z;\rho,b)
=
\kappa z
-\rho\nu\tanh(\alpha z)
-b
=0.
\]

在双曲稳定平衡点，隐函数定理给出：

\[
\boxed{
\frac{\partial z^\star}{\partial b}
=
\frac{1}
{\kappa-\rho\nu\alpha
\operatorname{sech}^2(\alpha z^\star)}.
}
\]

稳定平衡点满足分母为正。

在临界点以下的中性分支 \(z^\star=0\) 上：

\[
\frac{\partial z^\star}{\partial b}
=
\frac1{\kappa-\rho\nu\alpha}.
\]

当 \(\rho\uparrow\rho_c\) 时：

\[
\frac{\partial z^\star}{\partial b}
\to+\infty.
\]

因此临界点附近存在高**静态**控制灵敏度：很小的常值 \(b\) 就能显著移动平衡点。结合第7.3节可知，这个结论不能自动推出有限时间内的快速、鲁棒选支。

### 9.1 有限时域选支问题

若要求在截止时间 \(T_d\) 前形成置信度至少为 \(z_{\mathrm{tar}}>0\) 的正分支，可定义最小能量可达问题：

\[
\begin{aligned}
E_+^\star(T_d,z_{\mathrm{tar}})
=\min_{b(\cdot)}\;&
\int_0^{T_d}b(t)^2dt\\
\text{s.t. }\;&
\tau_z\dot z
=-kappa z+\rho\nu\tanh(\alpha z)+b,\\
&z(0)=z_0,\quad z(T_d)\ge z_{\mathrm{tar}},\\
&|b(t)|\le b_{\max}.
\end{aligned}
\]

负分支同理。只有这个问题或其随机可达版本，才精确定义“给定时限和置信度下的最小选支能量”。静态 susceptibility、折叠阈值和势垒分别描述平衡移动、保证井消失和抗扰性，三者不能互相代替。

### 9.2 分岔控制能量

定义联合目标：

\[
\boxed{
J(\pi)
=
\mathbb E_\pi
\left[
\sum_{t=0}^{\infty}\gamma^t
\left(
r_{\mathrm{task}}^t
-\lambda_b
\frac{
\sum_{e\in\mathcal E_\star}b_e^{t\,2}
}{
M_\star
}
\right)
\right].
}
\]

其中 \(M_\star=\max(1,|\mathcal E_\star|)\) 是固定归一化常数，非活跃边由结构门强制 \(b_e=0\)。使用固定常数而不是随时间变化的 \(|\mathcal E_t|\) 可避免图规模变化时同一条边的控制价格突然改变，也使PMP/Bellman中的系数保持一致。若应用必须按活跃边平均，应把 \(M_t\) 明确并入混杂状态，并只在固定图段上使用后续微分条件。

惩罚 \(b^2\) 而不是 \(z^2\) 的原因是：

- \(b\) 是外部干预能量；
- 非零 \(z\) 是期望保留的稳定协调承诺；
- 在双稳态区，选支完成后可以令 \(b\to0\)，而 \(z\) 继续保持。

### 9.3 可选的临界窗口结构

令：

\[
\lambda_e
=
\frac{\rho_e\nu\alpha}{\kappa}.
\]

首先定义满足远离冲突时严格关闭的激活函数：

\[
w_0(\rho)
=
1-\exp(-\rho/\rho_0),
\qquad
w_0(0)=0.
\]

可以再定义软临界门：

\[
g_{\mathrm{crit}}(\lambda)
=
g_{\min}
+(1-g_{\min})
\exp\left[
-\frac{(\lambda-1)^2}{2\sigma_\lambda^2}
\right],
\]

以及可选门控：

\[
\boxed{
g_{\mathrm{sel}}(\rho,\lambda)
=
w_0(\rho)g_{\mathrm{crit}}(\lambda).
}
\]

其中 \(0\le g_{\min}\le1\)。因 \(w_0(0)=0\)，无冲突时 \(b\) 严格归零；\(g_{\min}>0\) 则保留深度双稳态区域中的有限反转能力。必须强调：高斯临界门是优化归纳偏置，不是分岔理论的必要组成。核心理论只要求支持门 \(w_0\)；是否加入 \(g_{\mathrm{crit}}\) 应由“临界门消融”决定，因为过强的临界门会削弱错误分支的纠正能力。

### 9.4 选支能力与强制反转能力

“临界点附近能够选支”和“进入错误分支后能够强制反转”是两个不同要求。若只需在分岔发生时选择分支，小幅 \(b\) 即可利用临界灵敏度；若要求在固定 \(\lambda>1\) 下通过消除原稳定井保证准静态反转，则最大可用控制必须满足：

\[
\boxed{
b_{\max}
m_ec_e
w_0(\rho_e)\widetilde g_e
>
b_{\mathrm{sw}}(\rho_e).
}
\]

若该不等式不成立，系统仍可能通过跨越不稳定分界面完成动态切换，但不能保证原分支发生鞍结消失。论文应明确区分“稳定承诺”与“任意时刻可强制翻转”，二者存在设计权衡。

---

## 10. MARL分岔控制的结构约束

### 10.1 反对称控制

对有向表示 \((i,j)\)，使用共享评分器：

\[
s_{ij}=G_\phi(\chi_{ij},z_{ij}),
\qquad
s_{ji}=G_\phi(\mathcal S\chi_{ij},-z_{ij}),
\]

其中 \(\mathcal S\) 是交换两个智能体的特征变换。定义：

\[
\ell_{ij}=s_{ij}-s_{ji}.
\]

于是：

\[
\ell_{ji}=-\ell_{ij}.
\]

最终控制为：

\[
\boxed{
b_{ij}
=
b_{\max}
m_{ij}c_{ij}
w_0(\rho_{ij})\widetilde g_{ij}
\tanh\left(\frac{\ell_{ij}}{T_b}\right).
}
\]

其中 \(m_{ij}\in\{0,1\}\) 是当前边有效掩码，\(c_{ij}\in[0,1]\) 是无向共享的感知置信度；若不需要置信度门控，取 \(c_{ij}=1\)。\(\widetilde g_{ij}=1\) 表示不使用临界高斯门；若启用第9.3节的可选归纳偏置，则取 \(\widetilde g_{ij}=g_{\mathrm{crit}}(\lambda_{ij})\)。\(m,c,w_0,\widetilde g\) 都必须在交换下不变，且不得携带智能体ID优先级。

若 \(m,c,\rho,\lambda\) 采用无向边共享值，且交换算子满足 \(\mathcal S^2=I\)，则：

\[
b_{ji}=-b_{ij}.
\]

读取 \(z_{ij}\) 使控制器能够表达真正的反馈律：在 \(|z|\) 小时短暂施加选支偏置，在正确分支形成后令 \(b\to0\)，在当前承诺价值下降时施加反向控制。不读取 \(z\) 的纯前馈 \(b_\phi(\chi)\) 可作为消融，但一般无法在静态物理观测下同时实现“先选支、后撤去控制”，不应作为完整理论的默认控制类。

但闭环长期平衡点必须同时满足：

\[
\kappa z^\star
-\rho\nu\tanh(\alpha z^\star)
-b_\phi(\chi,z^\star)=0.
\]

其局部连续时间稳定性由：

\[
-\kappa
+\rho\nu\alpha\operatorname{sech}^2(\alpha z^\star)
+\frac{\partial b_\phi}{\partial z}(\chi,z^\star)
\]

的符号决定：负值对应局部渐近稳定。所以 \(b_{\mathrm{sw}}(\rho)\) 是冻结常值控制的精确折叠阈值，不是任意学习反馈律的闭环全局阈值。最简单的闭环稳定性措施是对 \(|\partial b_\phi/\partial z|\) 做梯度正则，并在稳定分支上监测上述Jacobian。

对本文真正采用的离散采样-保持反馈，在固定 \(\chi,\rho\) 时复合映射的导数为：

\[
\boxed{
\frac{d}{dz^t}
P_{h_z}\!\left(z^t;\rho,b_\phi(\chi,z^t)\right)
=
\frac{h_z^{-1}+\partial b_\phi/\partial z}
{h_z^{-1}+\mathcal U''(z^{t+1};\rho,b^t)}.
}
\]

因此近端子问题唯一只保证分母为正，不保证学习反馈后的复合映射一定收缩。离散局部稳定的精确线性条件是上式绝对值小于1。

上式是单边标量情形。若 \(b_\phi\) 通过图网络耦合多条边，则复合Jacobian为：

\[
J_{\mathrm{cl}}
=
P_z+P_bJ_{b,z},
\]

局部离散稳定条件是谱半径 \(\rho_{\mathrm{spec}}(J_{\mathrm{cl}})<1\)。这一条件与冲突强度 \(\rho_e\) 不同，因此谱半径使用下标 \(\mathrm{spec}\) 以避免符号混淆。

### 10.2 为什么不直接学习 \(\rho\) 和动力学参数

第一版固定：

\[
\kappa,\nu,\alpha,\tau_z,
\quad
\rho=R(\chi).
\]

只学习 \(b_\phi\)。否则网络可能通过移动 \(\rho_c\)、消除双稳态或改变时间尺度来绕过分岔控制语义。

若后续学习这些参数，应采用正参数化和相图约束，例如：

\[
\kappa=\operatorname{softplus}(\bar\kappa),
\quad
\nu=\operatorname{softplus}(\bar\nu),
\quad
\rho_c\in[\rho_c^{\min},\rho_c^{\max}],
\]

并单独报告临界点漂移。

---

## 11. 近端离散化

### 11.1 近端意见更新

给定步长 \(h_z>0\)，定义：

\[
\boxed{
z^{t+1}
=
\arg\min_z
\left[
\frac{(z-z^t)^2}{2h_z}
+\mathcal U(z;\rho^t,b^t)
\right].
}
\]

一阶最优条件为：

\[
F_t(z)
=
\frac{z-z^t}{h_z}
+\kappa z
-\rho^t\nu\tanh(\alpha z)
-b^t
=0.
\]

该更新是势能梯度流的隐式Euler离散，也可以理解为一个一维可微优化层。若物理离散时间为 \(\Delta t\)，则这里的近端步长应取

\[
h_z=\frac{\Delta t}{\tau_z}.
\]

所以 \(h_z\) 已吸收意见时间常数，不能再在一阶条件中重复除以 \(\tau_z\)。

为了数值求解，令 \(A_h=h_z^{-1}+\kappa\)。根方程等价于：

\[
A_hz^{t+1}
=
h_z^{-1}z^t
+\rho^t\nu\tanh(\alpha z^{t+1})
+b^t.
\]

由 \(|\tanh|\le1\)，唯一根必位于显式区间：

\[
\boxed{
\frac{h_z^{-1}z^t+b^t-\rho^t\nu}{A_h}
\le z^{t+1}\le
\frac{h_z^{-1}z^t+b^t+\rho^t\nu}{A_h}.
}
\]

该区间为Newton—二分混合求根提供了全局保险括号，避免仅凭“4–8次Newton”假设收敛。

---

## 12. 定理四：近端子问题适定性

近端目标的二阶导数为：

\[
\frac1{h_z}
+\kappa
-\rho\nu\alpha
\operatorname{sech}^2(\alpha z).
\]

由于 \(\operatorname{sech}^2\le1\)，若：

\[
\boxed{
\frac1{h_z}+\kappa
>
\rho_{\max}\nu\alpha,
}
\]

则近端目标对所有合法 \((\rho,b)\) 严格凸。

### 定理4：单步唯一性

在上述步长条件下，每一个 \((z^t,\rho^t,b^t)\) 都对应唯一的 \(z^{t+1}\)。因此近端分岔层定义了单值、连续且光滑的状态转移映射：

\[
z^{t+1}
=
P_{h_z}(z^t;\rho^t,b^t).
\]

注意，单步近端目标严格凸不等于长期系统只有一个平衡点。该映射的固定点仍满足：

\[
\frac{\partial\mathcal U}{\partial z}=0,
\]

所以保留原连续系统的单稳态—双稳态结构。这里“保留”仅指固定点集合及其局部稳定/不稳定分类；它不表示离散轨迹与连续轨迹逐点相同，也不表示时变输入下无条件耗散。

### 推论：离散意见的全局最终有界性

由上述根方程有：

\[
|z^{t+1}|
\le
q_0|z^t|+\frac{\rho_{\max}\nu+b_{\max}}{h_z^{-1}+\kappa},
\qquad
q_0=\frac{1}{1+h_z\kappa}<1.
\]

因此：

\[
\boxed{
\limsup_{t\to\infty}|z^t|
\le
\frac{\rho_{\max}\nu+b_{\max}}{\kappa}.
}
\]

该上界与连续系统一致，是对时变有界 \((\rho_t,b_t)\) 仍成立的全局结论。

---

## 13. 定理五：离散能量耗散与固定点稳定性

### 定理5：冻结参数下的能量耗散

当 \((\rho,b)\) 在一个离散步内及连续多个分析步中保持不变时：

\[
\boxed{
\mathcal U(z^{t+1};\rho,b)
+
\frac{|z^{t+1}-z^t|^2}{2h_z}
\le
\mathcal U(z^t;\rho,b).
}
\]

### 证明

将近端目标记为：

\[
Q_t(z)
=
\frac{|z-z^t|^2}{2h_z}
+\mathcal U(z;\rho,b).
\]

由 \(z^{t+1}\) 的最优性：

\[
Q_t(z^{t+1})\le Q_t(z^t)=\mathcal U(z^t;\rho,b).
\]

整理即得。证毕。

### 近似求解的残差保证

设精确解为 \(z^{t+1}=P_{h_z}(z^t;\rho^t,b^t)\)，数值求解器返回 \(\widehat z^{t+1}\)，并满足：

\[
|F_t(\widehat z^{t+1})|\le\varepsilon_F.
\]

在强凸常数

\[
m_P=h_z^{-1}+\kappa-\rho_{\max}\nu\alpha>0
\]

下，\(F_t\) 强单调，因而：

\[
\boxed{
|\widehat z^{t+1}-z^{t+1}|
\le
\frac{\varepsilon_F}{m_P}.
}
\]

且对近端目标 \(Q_t\)) 有：

\[
0\le
Q_t(\widehat z^{t+1})-Q_t(z^{t+1})
\le
\frac{\varepsilon_F^2}{2m_P}.
\]

因此近似实现只能声称带 \(O(\varepsilon_F^2/m_P)\) 误差的能量关系。论文实验必须报告最大残差，而不能仅报告Newton迭代次数。

### 固定点的离散稳定性

对固定点 \(z^\star\)，近端映射的导数为：

\[
\boxed{
P_{h_z}'(z^\star)
=
\frac1
{1+h_z\mathcal U''(z^\star)}.
}
\]

若 \(z^\star\) 是势能局部极小点，则 \(\mathcal U''(z^\star)>0\)，从而：

\[
0<P_{h_z}'(z^\star)<1,
\]

固定点离散稳定。

若 \(z^\star\) 是势能局部极大点，且近端唯一性条件成立，则：

\[
0<1+h_z\mathcal U''(z^\star)<1,
\]

从而：

\[
P_{h_z}'(z^\star)>1,
\]

固定点离散不稳定。因此近端离散保持连续系统的稳定性分类。

在分岔点 \(\lambda=1,b=0,z^\star=0\) 处，\(\mathcal U''(0)=0\)，因而 \(P_{h_z}'(0)=1\)。线性化在此处不能判定渐近稳定性；必须使用四次势能/三次向量场项，才能得到与连续系统一致的非双曲渐近稳定和临界减速。所以“稳定性分类保持”对双曲固定点可由Jacobian直接证明，对临界非双曲点则需单独的高阶分析。

---

## 14. 时变交互下的能量关系

实际系统中 \(\rho_t,b_t\) 会变化。定义：

\[
\mathcal U_t(z)=\mathcal U(z;\rho_t,b_t).
\]

近端最优性仍给出：

\[
\mathcal U_t(z^{t+1})
-\mathcal U_t(z^t)
\le
-\frac{|z^{t+1}-z^t|^2}{2h_z}.
\]

对下一时刻势能：

\[
\begin{aligned}
\mathcal U_{t+1}(z^{t+1})
-\mathcal U_t(z^t)
\le{}&
-\frac{|z^{t+1}-z^t|^2}{2h_z}\\
&+
\left[
\mathcal U_{t+1}(z^{t+1})
-\mathcal U_t(z^{t+1})
\right].
\end{aligned}
\]

参数变化项满足：

\[
\left|
\mathcal U_{t+1}(z)-\mathcal U_t(z)
\right|
\le
\frac\nu\alpha
|\rho_{t+1}-\rho_t|
|\log\cosh(\alpha z)|
+|b_{t+1}-b_t||z|.
\]

该关系本身只给出“耗散减去参数注入”的能量账本，尚不足以无条件推出分支跟踪；还需要远离折叠点的统一曲率和分支漂移界。

### 命题：慢变稳定分支跟踪

设 \(\zeta_t\) 是 \(\mathcal U_t\) 的同一条稳定平衡分支，并假设在包含迭代轨迹和该分支的邻域内：

\[
\mathcal U_t''(z)\ge m_s>0,
\]

且分支每步漂移满足：

\[
|\zeta_{t+1}-\zeta_t|
\le\delta_\zeta.
\]

这个漂移界不需要完全凭经验设定。若连接 \((\rho_t,b_t)\) 与 \((\rho_{t+1},b_{t+1})\) 的参数路径上始终有 \(\mathcal U''(\zeta;\rho,b)\ge m_s\)，由隐函数定理可得保守界：

\[
\boxed{
\delta_\zeta
\le
\frac{\nu|\rho_{t+1}-\rho_t|+|b_{t+1}-b_t|}{m_s}.
}
\]

当路径接近折叠点时 \(m_s\to0\)，该界发散，正好表明慢变跟踪理论在动态切换区失效。

若该邻域对近端映射保持正向不变，则近端映射在该邻域内关于前一状态的Lipschitz常数满足：

\[
q
\le
\frac1{1+h_zm_s}
<1.
\]

因此跟踪误差满足：

\[
\boxed{
|z^{t+1}-\zeta_{t+1}|
\le
q|z^t-\zeta_t|
+\delta_\zeta,
}
\]

从而：

\[
\boxed{
\limsup_{t\to\infty}
|z^t-\zeta_t|
\le
\frac{\delta_\zeta}{1-q}.
}
\]

该结论只在轨迹不穿过分岔点、折叠点或吸引域边界时成立。分支切换阶段应使用非自治分岔分析或直接数值验证，不能套用上述局部跟踪界。

---

## 15. 命题一：近端层的可微性与梯度界

令：

\[
D_t
=
\frac1{h_z}
+\kappa
-\rho_t\nu\alpha
\operatorname{sech}^2(\alpha z^{t+1}).
\]

在近端唯一性条件下 \(D_t>0\)。由隐函数定理：

\[
\boxed{
\frac{\partial z^{t+1}}{\partial z^t}
=
\frac{1/h_z}{D_t},
}
\]

\[
\boxed{
\frac{\partial z^{t+1}}{\partial b^t}
=
\frac1{D_t},
}
\]

\[
\boxed{
\frac{\partial z^{t+1}}{\partial\rho^t}
=
\frac{\nu\tanh(\alpha z^{t+1})}{D_t}.
}
\]

令：

\[
m_P
=
\frac1{h_z}+\kappa-\rho_{\max}\nu\alpha>0,
\]

则有单步上界：

\[
\left|
\frac{\partial z^{t+1}}{\partial b^t}
\right|
\le\frac1{m_P},
\]

\[
\left|
\frac{\partial z^{t+1}}{\partial\rho^t}
\right|
\le\frac\nu{m_P}.
\]

同时：

\[
\left|
\frac{\partial z^{t+1}}{\partial z^t}
\right|
\le
\frac{h_z^{-1}}{m_P}.
\]

该全局上界可能大于1，因此强凸单步求解并不自动保证跨时间梯度收缩。只有在稳定分支邻域 \(\mathcal U''\ge m_s>0\) 中，才有更强的局部界：

\[
0<
\frac{\partial z^{t+1}}{\partial z^t}
\le
\frac1{1+h_zm_s}<1.
\]

该结果说明近端层既保留临界灵敏度，又避免精确单步映射的奇异Jacobian。接近不稳定平衡点时，跨时间导数可以大于1，这是分岔放大效应，而不是数值求解失败；训练时需要序列截断和梯度裁剪。实现上应对收敛根使用上述隐式Jacobian，或明确声称使用固定次数迭代的展开梯度；两者不完全等价。

---

## 16. 传统最优控制解释

### 16.1 双控制增广系统

将物理动作 \(a\) 与分岔控制 \(b\) 同时视为控制量：

\[
\dot x=f(x,a),
\]

\[
\tau_z\dot{\mathbf z}
=
-\nabla_{\mathbf z}
\mathcal U
(\mathbf z;\boldsymbol\rho(x),\mathbf b).
\]

为了严格使用经典Pontryagin原理，本节先考察确定性有限时域理想问题。连续时间代价为：

\[
\mathcal J
=
\int_0^T
\left(
\ell_{\mathrm{task}}(x,a)
+\bar\lambda_b\|\mathbf b\|^2
\right)dt
+\ell_T(x(T),\mathbf z(T))
.
\]

与第9.2节的固定归一化一致，取 \(\bar\lambda_b=\lambda_b/M_\star\)。若环境转移随机，则需使用随机最大值原理并增加驯项；本节公式只作确定性结构解释，离散随机问题由下一节Bellman条件承接。

这不是“RL加一个隐藏层”，而是一个标准的增广最优控制问题：

- \(a\) 控制物理状态；
- \(b\) 控制协调吸引子；
- \(\mathbf z\) 把离散模式历史转化为连续动态状态。

### 16.2 Pontryagin协态条件

令物理协态为 \(p_x\)，意见协态为 \(p_z\)。Hamiltonian为：

\[
\begin{aligned}
\mathscr H
={}&
\ell_{\mathrm{task}}(x,a)
+\bar\lambda_b\|b\|^2
+p_x^\top f(x,a)\\
&+
\frac1{\tau_z}
p_z^\top
\left[
-\kappa z
+\rho(x)\nu\tanh(\alpha z)
+b
\right].
\end{aligned}
\]

若暂不考虑控制边界，关于 \(b\) 的一阶必要条件为：

\[
\frac{\partial\mathscr H}{\partial b}
=
2\bar\lambda_b b
+\frac{p_z}{\tau_z}
=0.
\]

因此：

\[
\boxed{
b^\star
=
-\frac{p_z}{2\bar\lambda_b\tau_z}.
}
\]

考虑 \(|b|\le b_{\max}\) 后：

\[
\boxed{
b^\star
=
\Pi_{[-b_{\max},b_{\max}]}
\left(
-\frac{p_z}{2\bar\lambda_b\tau_z}
\right).
}
\]

对多边情形，该投影按无序边坐标逐分量作用。若还强制Hodge梯度空间、门控幅值或其他耦合可行集 \(\mathcal B(x,z)\)，则必要条件应写为：

\[
\boxed{
0\in
2\bar\lambda_b b
+\frac{p_z}{\tau_z}
+N_{\mathcal B(x,z)}(b),
}
\]

其中 \(N_{\mathcal B}\) 是法锥。简单逐分量裁剪只在 \(\mathcal B\) 是box约束时成立。

这一结果赋予分岔控制非常明确的最优控制含义：

\[
\boxed{
b^\star
\text{由未来代价对协调状态的协态决定，}
\bar\lambda_b
\text{控制干预保守程度。}
}
\]

协态方程同样不应省略。在运行代价不显式依赖 \(z\)、物理动力学不直接依赖 \(z\) 的理想化问题中：

\[
\dot p_z
=
-\frac1{\tau_z}
\operatorname{diag}
\left[
-\kappa+\rho(x)\nu\alpha
\operatorname{sech}^2(\alpha z)
\right]p_z,
\]

\[
\dot p_x
=
-\nabla_x\ell_{\mathrm{task}}
-(\nabla_xf)^\top p_x
-\frac\nu{\tau_z}
J_\rho(x)^\top
\left[p_z\odot\tanh(\alpha z)\right].
\]

终端条件为 \(p_x(T)=\nabla_x\ell_T\)、\(p_z(T)=\nabla_z\ell_T\)。这表明协调临界性 \(\rho(x)\) 通过 \(J_\rho\) 把意见协态反传回物理状态；理想问题并不是两个完全独立的控制器。若 \(\ell\) 或 \(f\) 显式依赖 \(z\)，应在 \(\dot p_z\) 中加上对应偏导项。

在未知环境中无法直接求解协态，所以使用MARL从长期回报中学习 \(b_\phi(\chi,z)\)。

这里的 \(p_z\) 是成本最小化问题中的协态。在满足Hamilton–Jacobi–Bellman方程、值函数可微且采用相同成本符号约定时：

\[
p_z=\nabla_z W,
\]

更严格地说，有限时域中应写为 \(p_z(t)=\nabla_zW(t,x(t),z(t))\)，其中 \(W\) 是成本到达函数。若MARL Critic使用与其完全相同时域、折扣和符号约定的累计回报 \(V=-W\)，则对应关系为：

\[
p_z=-\nabla_zV.
\]

这一区分决定后续公式的符号，不能把成本值函数和回报值函数混写。

---

## 17. 离散Bellman最优性与价值梯度

在离散系统中：

\[
z^{t+1}
=P_{h_z}(z^t;\rho^t,b^t).
\]

令 \(W(x,z)\) 表示成本到达函数。本节为简化符号，将固定归一化后的离散系数 \(\lambda_b/M_\star\) 仍记为 \(\lambda_b\)。将物理动作 \(a\) 与分岔控制 \(b\) 视为两个独立决策量，Bellman一步最优性写为：

\[
\min_{a,b}
\left[
\ell_{\mathrm{task}}(x,a)
+\lambda_b b^2
+\gamma
\mathbb E
\left[
W
\left(
X'(x,a),
P_{h_z}(z^t;\rho^t,b)
\right)
\right]
\right].
\]

这里假设物理转移在给定 \(a\) 后不直接依赖 \(b\)，并在理想双控制问题中先把 \(a,b\) 视为可独立选择的控制坐标。若实际Actor强制采用 \(a=\pi(o,P(z,b))\)，则对 \(b\) 求导还会出现动作通道

\[
\frac{\partial P_x}{\partial a}
\frac{\partial a}{\partial z^{t+1}}
\frac{\partial P_{h_z}}{\partial b}.
\]

该效应由实际策略梯度吸收，而不包含在下面这个理想化Bellman必要条件中。对内部局部最优解 \(b^\star\) 应用一阶必要条件，得到隐式方程：

\[
2\lambda_b b^\star
+\gamma
\mathbb E
\left[
\left(
\left.
\frac{\partial P_{h_z}}{\partial b}
\right|_{b^\star}
\right)^\top
\nabla_zW
\left(
X'(x,a^\star),
P_{h_z}(z;\rho,b^\star)
\right)
\right]
=0.
\]

因此：

\[
\boxed{
b^\star
=
-\frac\gamma{2\lambda_b}
\mathbb E
\left[
\left(
\left.
\frac{\partial P_{h_z}}{\partial b}
\right|_{b^\star}
\right)^\top
\nabla_zW
\right].
}
\]

在一般可行集 \(\mathcal B(x,z)\) 上，精确的一阶条件是包含法锥的变分不等式：

\[
\boxed{
0\in
2\lambda_bb^\star
+\gamma\mathbb E
\left[
P_b^\top\nabla_zW(x',z')
\right]
+N_{\mathcal B(x,z)}(b^\star).
}
\]

除非 \(\mathcal B\) 为box、期望梯度在当前求解中视为常量，否则不能把该非线性隐式条件简单等同为一次裁剪。

结合：

\[
\frac{\partial P_{h_z}}{\partial b}
=
\frac1{D_t},
\]

得到：

\[
b^\star
\propto
-\frac{\nabla_zW}{D_t}
=
\frac{\nabla_zV}{D_t},
\]

其中最后一个等号使用回报Critic \(V=-W\)。这建立了传统控制与MARL之间的目标一致性：

\[
\boxed{
\nabla_zW\text{是意见成本协态，}
\quad
-\nabla_zV=\nabla_zW,
\quad
\frac{\partial P}{\partial b}\text{决定价值对分岔控制的灵敏度。}
}
\]

该式是理想最优控制的必要条件，不是充分条件，也不表示标准PPO会把Critic的 \(\nabla_zV\) 直接反向传播给Actor。本文的主算法对价值回归路径停止梯度，使用策略梯度在可表示的分散策略类中寻找驻点。该驻点是否近似满足上述Bellman条件需要事后检验，不是PPO自动保证。只有另行加入“冻结Critic的可微价值改进损失”时，才会显式使用 \(\nabla_zV\)；该扩展不属于基础算法。

由于 \(b_\phi\) 只使用局部信息，其可表示类是 \(\{b_\phi(\chi_i,z_i)\}\)。训练目标是在该受限类中近似集中式 \(b^\star(x,z)\)，但标准PPO通常只能达到局部驻点，不能声称得到“最佳可实现近似”。

---

## 18. 增广Dec-POMDP与Markov性质

定义一步内部更新：

\[
\bar{\mathbf z}^{\,t}
=
P_{h_z}
(\mathbf z^t;
\boldsymbol\rho(x^t),
\mathbf b_\phi(o^t,\mathbf z^t,\mathcal R^t,h^t)).
\]

物理动作策略为：

\[
a_i^t
\sim
\pi_{\theta_a}
\left(
\cdot\mid
o_i^t,
\bar{\mathbf z}_i^{\,t},
h_i^t
\right).
\]

环境转移后：

\[
x^{t+1}\sim P(\cdot\mid x^t,a^t),
\qquad
\mathbf z^{t+1}=\bar{\mathbf z}^{\,t}.
\]

### 定理6：增广Markov性

在A4–A5下，给定固定策略参数，令 \(\mathcal R^t\) 表示边注册表、有效掩码和必要的对称破缺随机状态，并令 \(h^t\) 表示其他策略记忆。完整增广状态为：

\[
\widehat x^t
=
(x^t,\mathbf z^t,\mathcal R^t,h^t).
\]

若采用固定边全集、无额外GRU且对称破缺噪声在生成后立即写入 \(z\)，则 \(\mathcal R^t\) 可简化为掩码 \(m^t\)，\(h^t\) 可删除，从而 \((x^t,\mathbf z^t,m^t)\) 就是足够的固定维状态。

该联合状态满足：

\[
\Pr
(\widehat x^{t+1}\mid
\widehat x^{0:t},a^{0:t})
=
\Pr
(\widehat x^{t+1}\mid
\widehat x^t,a^t).
\]

因此可以对完整增广系统定义合法的价值函数：

\[
V^\pi(\widehat x)
=
\mathbb E_\pi
\left[
\sum_{k=0}^{\infty}
\gamma^k\widetilde r_{t+k}
\mid \widehat x_t=\widehat x
\right].
\]

### 证明

在确定性观测情形，由于 \(o^t=O(x^t)\)、\(\rho^t=R(x^t,\mathcal R^t)\) 且 \(b^t=b_\phi(o^t,\mathbf z^t,\mathcal R^t,h^t)\)，意见和边生命周期转移在给定当前完整增广状态后是确定性的；若包含观测噪声或随机破缺，则它们构成当前转移核中的外生随机变量。意见部分可写为：

\[
\Pr(\mathbf z^{t+1}\mid \text{history})
=
\delta\!\left(
\mathbf z^{t+1}
-P_{h_z}(\mathbf z^t;\boldsymbol\rho(x^t),\mathbf b_\phi(o^t,\mathbf z^t,\mathcal R^t,h^t))
\right).
\]

物理转移满足：

\[
\Pr(x^{t+1}\mid\text{history},a^t)
=
P(x^{t+1}\mid x^t,a^t).
\]

再乘以边注册表和策略记忆的更新核，所得联合转移只依赖当前 \(\widehat x^t\)、当前策略和当前动作，因此完整过程满足一阶Markov性质。证毕。

若Critic忽略 \(\mathbf z\)、掩码或边生命周期状态，则相同物理状态下不同协调承诺可能被错误聚合，Critic实际上只能学习对被省略状态条件边缘化后的价值，通常具有更大的条件方差。不能简单说其“不是单值函数”，更准确的说法是其输入不再是充分统计量。

---

## 19. 策略梯度闭环

令总参数为：

\[
\theta=(\theta_a,\phi),
\]

其中 \(\phi\) 控制 \(b_\phi\)，\(\theta_a\) 控制物理动作分布。

虽然 \(b\) 是确定性内部控制，不单独作为随机动作出现在log-prob中，但它通过确定性策略记忆递推改变后续物理动作分布。令 \(h_t\) 包含边状态、边注册表和其他Actor记忆，则轨迹中与参数有关的随机密度为：

\[
\log p_\theta(\tau)
=
\mathrm{const}
+\sum_t
\log\pi_\theta(a_t\mid o_t,h_t).
\]

因此每一个log-prob都必须对其参数依赖的记忆状态取总导数。例如：

\[
\boxed{
\frac{D}{D\phi}
\log\pi_\theta(a_t\mid o_t,h_t)
=
\frac{\partial\log\pi}{\partial\phi}
+
\frac{\partial\log\pi}{\partial h_t}
\frac{Dh_t}{D\phi}.
}
\]

对意见分量：

\[
\frac{Dz_{t+1}}{D\phi}
=
\frac{\partial P_t}{\partial z_t}
\frac{Dz_t}{D\phi}
+
\frac{\partial P_t}{\partial b_t}
\frac{Db_t}{D\phi},
\]

\[
\frac{Db_t}{D\phi}
=
\frac{\partial b_\phi}{\partial\phi}
+\frac{\partial b_\phi}{\partial z_t}
\frac{Dz_t}{D\phi}
+\frac{\partial b_\phi}{\partial h_t^{\mathrm{other}}}
\frac{Dh_t^{\mathrm{other}}}{D\phi}.
\]

第二式是状态反馈 \(b_\phi(\chi,z)\) 必须补上的链式求导项。若将 \(b\) 误写为纯前馈控制并在此处忽略 \(\partial b/\partial z\)，训练梯度与实际闭环不一致。

所以任务优势可以沿如下路径反向传播：

\[
\boxed{
\widehat A_t
\rightarrow
\log\pi(a_t\mid o_t,h_t)
\rightarrow
h_t
\rightarrow
P_{h_z}
\rightarrow
b_\phi.
}
\]

这使MARL能够学习哪些早期小输入会在未来形成更高价值的协调分支。

严格地说，参数依赖的确定性记忆使完整轨迹似然比等于各时刻条件概率比的乘积；PPO使用的逐时刻裁剪比率不是该完整乘积，而是局部、截断的代理目标。因此本文只主张“存在合法的序列策略梯度路径”，不主张PPO裁剪在该循环系统上保持TRPO式单调改进保证。分岔控制信赖域、短序列重算与经验稳定性检验正是为这一近似服务。

此外，控制能量是显式依赖参数的正则项，其梯度不能只依赖似然比估计，而应直接反向传播：

\[
\nabla_\phi
\left[
\frac{\lambda_b}{M_\star}\|b_t\|^2
\right]
=
\frac{2\lambda_b}{M_\star}
\left(
\frac{Db_t}{D\phi}
\right)^\top b_t.
\]

这里必须是总导数 \(Db_t/D\phi\)，因为当 \(b_t\) 读取 \(z_t\) 时，早期参数会经由意见递推影响后续控制能量。

因此基础算法包含两类梯度：物理动作log-prob产生的策略梯度，以及分岔控制能量的路径梯度。它不包含价值回归损失对 \(b_\phi\) 的梯度。

还有一个容易被忽略的策略梯度一致性问题。第9.2节的真正单步回报是：

\[
\widetilde r_t
=
r_{\mathrm{task},t}
-\frac{\lambda_b}{M_\star}\|b_t\|^2.
\]

因此若要对该目标给出无偏的on-policy score-function梯度，GAE/回报必须由 \(\widetilde r_t\) 计算，同时保留上述显式路径梯度。前者处理动作对未来能量代价的状态分布效应，后者处理在固定轨迹上代价对参数的显式依赖。若 \(\widehat A_t\) 只由任务回报计算，而 \(b^2\) 只作为采样状态上的附加损失，那是常用但有偏的工程正则化，不能称为第9.2节 \(J\) 的精确策略梯度。

### 序列训练必要性

若训练重算时在每一步将 \(z_t\) detach，则只保留：

\[
b_t\rightarrow z_{t+1}\rightarrow a_t
\]

的瞬时梯度，而丢失：

\[
b_{t-k}\rightarrow z_{t-k+1}\rightarrow\cdots\rightarrow z_t\rightarrow a_t
\]

的长期选支梯度。因此必须采用recurrent PPO式序列重算和截断BPTT。rollout缓存中的状态当然可以脱离旧计算图；“不得detach”仅指更新阶段在同一训练chunk内部重建的递推状态。

---

## 20. PPO的约束优化解释

对同一条rollout历史，新旧策略必须分别从同一chunk初始记忆顺序重建各自的内部状态。逐时刻比率定义为：

\[
\boxed{
r_t(\theta)
=
\frac{
\pi_\theta(a_t\mid o_t,h_t^\theta)
}{
\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t,h_t^{\mathrm{old}})
}.
}
\]

分母应使用rollout记录的旧log-prob，分子使用新策略重算状态；不能把旧 \(z_t\) 直接喂给新策略后声称完成了循环策略比率校正。

下文 \(\widehat A_t\) 默认由增广回报 \(\widetilde r_t=r_{\mathrm{task},t}-\lambda_b\|b_t\|^2/M_\star\) 计算。若使用task-only advantage，必须在方法中标明这是偏置的工程变体。

理想更新可以写成：

\[
\max_\theta
\quad
\mathbb E
\left[
r_t(\theta)\widehat A_t
-\frac{\lambda_b}{M_\star}\|b_\phi^t\|^2
\right]
\]

满足：

\[
\mathbb E
\left[
D_{\mathrm{KL}}
(\pi_{\theta_{\mathrm{old}}}
\|\pi_\theta)
\right]
\le\delta_\pi.
\]

物理动作KL并不能直接限制确定性内部控制 \(b_\phi\) 的变化。尤其在临界区，较小的 \(b\) 变化也可能被分岔动力学放大。因此建议至少在训练早期加入独立的分岔控制信赖域：

\[
\mathbb E
\left[
\|b_\phi-b_{\phi_{\mathrm{old}}}\|^2
\right]
\le\delta_b.
\]

更直接的诊断量是同一chunk上新旧内部轨迹的偏差：

\[
\Delta_z
=
\frac1L\sum_{t=1}^L
\|z_t^\theta-z_t^{\mathrm{old}}\|^2.
\]

它可用于早停、自适应学习率或额外罚项，但不必强行作为基础方法的理论约束。

实际采用裁剪代理目标：

\[
\mathcal L_{\mathrm{clip}}
=
-\mathbb E
\left[
\min
\left(
r_t(\theta)\widehat A_t,
\operatorname{clip}
(r_t,1-\epsilon,1+\epsilon)
\widehat A_t
\right)
\right].
\]

总损失为：

\[
\boxed{
\begin{aligned}
\mathcal L
={}&
\mathcal L_{\mathrm{clip}}
+c_V\mathcal L_V
-\beta_H\mathcal H\\
&+\frac{\lambda_b}{M_\star}
\mathbb E\|b_\phi\|^2
+\lambda_{\Delta b}
\mathbb E
\|b_\phi-b_{\phi_{\mathrm{old}}}\|^2.
\end{aligned}}
\]

最后一项可在训练稳定后减小，但只有在监测到 \(\Delta b\)、分支切换率和内部状态重算误差均稳定时才建议移除。PPO ratio中的log-prob必须由rollout时的旧策略记录；新策略则从每个chunk保存的初始完整记忆状态顺序重算 \(b,z,h\)。

---

## 21. Critic的理论角色

中央Critic采用：

\[
V_\psi
\left(
x_t,
\operatorname{sg}[\mathbf z_t],
\mathcal R_t,
h_t^{\mathrm{other}}
\right).
\]

在固定边全集且无额外记忆的简化实现中，上式简化为 \(V_\psi(x_t,\operatorname{sg}[\mathbf z_t],m_t)\)。

这里 \(\operatorname{sg}\) 表示价值回归损失不得通过Critic输入修改分岔状态。原因是：

Critic的回归目标应与Actor目标一致，即默认使用增广回报 \(\widetilde r_t\)。若Critic只估计任务回报而Actor另外惩罚能量，优势就不再是完整目标的优势。

1. Critic需要读取 \(z\)，因为 \((x,z^+)\) 与 \((x,z^-)\) 的未来价值不同；
2. Critic损失的目标是拟合价值，而不是控制系统；
3. 若让价值回归梯度直接更新 \(b,z\)，Actor可能通过改变状态表示降低回归误差，而非提高回报；
4. Actor梯度仍可通过策略分布中的 \(z\) 正常传播到 \(b_\phi\)。

因此Critic的严格算法作用是在完整增广状态上拟合值函数并降低优势估计方差。它与协态的关系必须更精确地表述：

- 若 \(V\) 光滑，\(-\nabla_zV\) 在理想成本符号下可解释为意见协态；
- 标准PPO只使用Critic产生的标量优势，不直接使用 \(\nabla_zV\)，因此不能声称Critic在训练中显式恢复了协态。

不需要额外构造 \(Q^+-Q^-\) 教师，除非作为后续辅助诊断。

---

## 22. 交换对称性与互补行为

### 22.1 反对称边状态

为无序边 \(e=\{i,j\}\) 只维护一个 \(z_e\)。若存储方向为 \(i\to j\)，则：

\[
z_{ij}=z_e,
\qquad
z_{ji}=-z_e.
\]

同理：

\[
b_{ij}=b_e,
\qquad
b_{ji}=-b_e.
\]

近端映射本身也保持该对称性。由于：

\[
\mathcal U(-z;\rho,-b)=\mathcal U(z;\rho,b),
\]

并且强凸条件下近端解唯一，可得：

\[
\boxed{
P_{h_z}(-z^t;\rho,-b)
=
-P_{h_z}(z^t;\rho,b).
}
\]

因此只要初始边状态和控制严格反对称，精确近端更新不会破坏成对一致性。对数值近似解，两端分别求根会产生容差级不一致；实现上应只求一个无序边状态，另一端直接取负号。

### 命题：联合交换等变性

若：

1. 智能体共享策略参数；
2. 成对特征在交换智能体时按 \(\mathcal S\) 变换；
3. \(b\) 和 \(z\) 具有反对称性；
4. Actor对邻居集合采用置换不变聚合；

则交换智能体标签会交换其动作分布，而不会改变联合策略的物理含义。

该结构允许参数共享策略在对称物理场景中表达互补联合动作，而无需人为给某个固定ID更高优先级。

### 22.2 精确对称破缺

若：

\[
z=0,
\quad b=0,
\quad\lambda>1,
\]

数学上零点不稳定，但确定性有限精度实现可能永久停留在零点。需要在边首次进入临界区域时加入：

\[
\epsilon_e\sim\mathcal D,
\qquad
\mathbb E[\epsilon_e]=0,
\]

并保证双方读取相反符号。对称分布保证两分支在交换对称任务中没有系统性偏置。

若加在连续控制通道，可写为 \(b_e\leftarrow b_e+\epsilon_e\)；若只在边创建时初始化，则写为 \(z_e\leftarrow\epsilon_e\)。两种方案的概率模型不同，不应在理论与代码中混用。严格无通信且无公共随机性时，双方无法保证独立抽样恰好反对称，这一边界与第24节一致。

---

## 23. 多车循环冲突的图一致性扩展

成对反对称只能保证：

\[
z_{ij}=-z_{ji},
\]

不能自动排除：

\[
i\succ j,
\qquad
j\succ k,
\qquad
k\succ i.
\]

若任务中三车以上同步冲突频繁，可以引入图势函数 \(p_i\)：

\[
b_{ij}
=
g_{ij}
\tanh
\left(
\frac{p_i-p_j}{T_p}
\right),
\qquad
g_{ij}\ge0.
\]

只要 \(p_i\) 互异，则瞬时控制偏好按节点标量排序，因此不含有向偏好环。

另一种形式是对原始边控制 \(\widetilde{\mathbf b}\) 做图Hodge梯度投影：

\[
\mathbf b
=
B^\top(BB^\top)^\dagger B
\widetilde{\mathbf b},
\]

其中 \(B\) 是有向关联矩阵。投影后的 \(\mathbf b\) 属于图梯度空间，环流分量被移除。

该扩展应作为多车全局一致性的第二层贡献，而不是二车核心分岔理论的必要组成。由于迟滞可能保留历史分支，即使瞬时 \(b\) 无环，也应额外测量 \(z\) 的循环率。

---

## 24. 严格分散执行的边界

理论主模型的一条无序边只有一个状态，这要求至少满足以下之一：

1. 双方基于相同的公共成对观测确定性更新；
2. 允许交换一个标量边状态或随机种子；
3. 状态由公共基础设施维护并分别发送有向副本。

若严格无通信且双方观测不一致，则只能分别维护：

\[
z_{ij}^{(i)},
\qquad
z_{ji}^{(j)},
\]

精确反对称性一般无法保证。此时论文应改为分析一致性误差：

\[
e_{ij}
=
z_{ij}^{(i)}+z_{ji}^{(j)},
\]

并通过共享参数、观测对齐、一致性正则或有限通信证明/验证 \(e_{ij}\) 有界。不能在非对称感知下无条件声称精确共同意见。

---

## 25. 冲突解除与中性恢复

冲突解除时：

\[
\rho_e\to0,
\qquad
b_e\to0.
\]

动力学变为：

\[
\tau_z\dot z_e
=
-\kappa z_e.
\]

从而：

\[
\boxed{
z_e(t)
=
z_e(t_0)
\exp
\left[
-\frac\kappa{\tau_z}(t-t_0)
\right].
}
\]

恢复时间常数为：

\[
T_{\mathrm{reset}}
=
\frac{\tau_z}{\kappa}.
\]

对应的精确近端离散更新为：

\[
\boxed{
z^{t+1}
=
\frac{z^t}{1+h_z\kappa}.
}
\]

因此离散实现也以几何速率回归中性。为保留Markov性与可复现性，边失效后必须在以下两种语义中二选一：

1. **自然衰减**：保留边槽，取 \(\rho=b=0\)，直到 \(|z|<\varepsilon_{\mathrm{reset}}\)；
2. **混杂重置**：边删除事件发生时立即令 \(z^+=0\)。

前者保留短暂遮挡下的承诺，后者消除跨交互污染。论文必须选定一种并在增广转移中明示，不能只在代码中隐式处理。

因此系统不会永久保留已经失效的优先关系。

---

## 26. 与线性滤波和RNN的理论区别

### 26.1 线性滤波器

线性系统：

\[
\dot z=-\kappa z+b
\]

只有唯一平衡点：

\[
z^\star=b/\kappa.
\]

当 \(b=0\) 时必然回到零，因此必须持续输入才能维持意见；不存在相变、多稳态、势垒和迟滞。

### 26.2 GRU/LSTM

循环网络理论上可以逼近分岔动力学，但通常不显式保证：

- 临界点位置；
- 单稳态—双稳态相图；
- 势能耗散；
- 分支切换阈值；
- 中性恢复时间；
- 控制输入的反对称符号；
- 干预能量的最优控制含义。

因此本方法的贡献不是更强的函数逼近能力，而是把协调模式的动态结构限制在一个可分析、可审计的函数族中。

---

## 27. 方法能够保证与不能保证的内容

### 可以理论支持

1. 意见状态的全局有界性；
2. 冲突临界点处的超临界pitchfork；
3. 冻结常值控制下的精确折叠边界和确定性迟滞区；
4. 分岔控制的明确势能倾斜方向；
5. 近端子问题的唯一性；
6. 冻结输入下的离散能量耗散；
7. 近端层的可微性和单步梯度界；
8. 冲突解除后的指数中性恢复；
9. 增广状态的Markov性；
10. 理想双控制问题中分岔控制与协态/价值梯度之间的一阶必要条件；
11. 在明确观测和状态共享假设下的交换等变与成对一致性。
12. 精确近端更新下的离散全局最终有界性；
13. 数值求根残差对状态误差和近端目标误差的显式上界。

### 不能单独保证

1. 物理碰撞永不发生；
2. PPO收敛到全局最优策略；
3. 任意数量智能体下优先图始终无环；
4. 严格无通信且非对称感知下双方意见精确相反；
5. 所有动态场景都满足慢变分支跟踪条件；
6. 分岔选支一定早于不可避免碰撞时间。
7. 临界灵敏度高自动意味有限时间决策更快或能量更小；
8. 确定性折叠阈值能无条件阻止白噪声或感知跳变诱发的跨势垒切换；
9. 固定次数、未收敛的Newton近似解精确满足能量耗散和隐式Jacobian公式；
10. 标准PPO显式恢复 \(-\nabla_zV\) 协态或严格满足Bellman一阶条件。
11. 冻结常值控制的 \(b_{\mathrm{sw}}(\rho)\) 是任意状态反馈 \(b_\phi(\chi,z)\) 下的闭环切换阈值，或近端子问题唯一能自动保证复合反馈映射收缩。

若需要硬安全保证，应在所有比较方法上统一增加CBF、GCBF、ORCA或MPC安全层，并将安全可行性与协调最优性分开陈述。

---

## 28. 建议的论文理论结构

正文理论部分建议按以下顺序组织。

### 28.0 证明依赖与主张等级

为避免把不同强度的结论混在一起，论文应明确采用以下依赖链：

\[
\text{A1--A2}
\Rightarrow
\text{强制性与连续有界性}
\Rightarrow
\text{无偏分岔与受控折叠},
\]

\[
\text{A1}+\left(h_z^{-1}+\kappa>\rho_{\max}\nu\alpha\right)
\Rightarrow
\text{近端唯一性、光滑性、冻结输入固定点分类与耗散},
\]

\[
\text{A7}+
\rho_{\mathrm{spec}}(P_z+P_bJ_{b,z})<1
\Rightarrow
\text{学习反馈固定点的局部离散稳定性},
\]

\[
\text{A4--A9}
\Rightarrow
\text{增广Markov建模与受限策略梯度解释}.
\]

据此，全文结论分为三档：

1. **严格全局结论**：势能强制性、输入有界下的意见最终有界、近端单步唯一性、冻结输入能量下降；
2. **严格局部或条件性结论**：稳定分支跟踪、临界灵敏度、固定点稳定性保持、交换等变性；
3. **算法解释而非收敛定理**：PMP/Bellman给出的理想控制必要条与PPO能量正则化选支目标之间的结构对应。

其中，近端步长条件是整个离散理论的关键设计约束；一旦违反，就不能继续声称单步解唯一、隐式Jacobian处处有界或冻结 \((\rho,b)\) 下的离散稳定性分类保持。即便该条件成立，学习反馈复合后仍需检查 \(P_z+P_bJ_{b,z}\)。PPO部分则不声称全局收敛、单调策略改进或恢复集中式最优解。

### Definition 1：Conflict-induced controlled opinion potential

定义 \(\mathcal U_e(z_e;\rho_e,b_e)\) 及其物理语义。

### Proposition 1：Coercivity and boundedness

证明势能强制性与意见最终有界。

### Theorem 1：Conflict-induced supercritical bifurcation

证明 \(\rho_c=\kappa/(\nu\alpha)\) 处由中性单稳态进入双稳态。

### Corollary 1：Pitchfork normal-form equivalence

通过Taylor展开给出三阶规范形及临界平方根标度。

### Theorem 2：Controlled imperfect bifurcation and hysteresis margin

给出 \(b_{\mathrm{sw}}(\rho)\) 与平衡点数量。

### Proposition 2：Critical susceptibility and critical slowing down

给出 \(\partial z^\star/\partial b\) 和局部恢复时间标度，说明静态高灵敏度与有限时间快速选支的区别。

### Theorem 3：Well-posed proximal realization

给出近端唯一性、能量耗散和稳定性保持。

### Proposition 3：Differentiability and residual-certified realization of Prox-NOD

给出对 \(z_t,b_t,\rho_t\) 的解析Jacobian，以及数值残差到状态/目标误差的上界。

### Theorem 4：Augmented Markov property

证明 \((x,z,\mathcal R,h^{\mathrm{other}})\) 是一般完整状态；在固定边、无额外记忆时简化为 \((x,z,m)\)，Critic应在对应充分统计量上估值。

### Proposition 4：Costate interpretation of ideal bifurcation control

给出连续协态条件和离散价值梯度必要条件，连接理想传统最优控制与MARL目标，但不宣称PPO显式实现该条件。

### Proposition 5：Exchange-equivariant complementary policy

在共享参数、反对称边状态和置换不变聚合下证明联合策略的交换等变性。

附录可以给出完整证明、时变参数扰动界和图一致性扩展。

---

## 29. 简要网络与实现路线

实现只需要六个模块。

### 29.1 冲突适配器

输入局部物理观测，输出：

```text
pair_features      [B,N,K,Fp]
edge_ids           [B,N,K]
candidate_mask     [B,N,K]
conflict_rho       [B,N,K]
confidence         [B,N,K]
```

### 29.2 分岔控制网络

共享PairScorer读取 \((\chi_{ij},z_{ij})\) 与 \((\mathcal S\chi_{ij},-z_{ij})\)，计算交换差分，输出有界反对称 \(b_{ij}\)。

### 29.3 Proximal Saturating NOD层

求解：

\[
\frac{z-z_t}{h_z}
+\kappa z
-\rho\nu\tanh(\alpha z)
-b
=0.
\]

使用显式根括号上的Newton—二分混合求根，直到 \(|F_t(z)|\le\varepsilon_F\)；前向记录残差，反向使用收敛根的隐式Jacobian。固定4–8次迭代只能作为有残差监测的工程上限，不是理论收敛条件。

### 29.4 分支条件Actor

将：

\[
q_{ij}=\tanh(z_{ij}/z_0)
\]

编码为边上下文，经masked attention或图消息传递聚合后生成连续动作分布。不要使用固定的“正意见加速、负意见减速”动作残差。

为排除Actor忽略或绕过分岔状态的退化解，可将一般分支条件策略收紧为Base锚定的
因果门控子类。令：

\[
e_{ij}=E_\theta(\chi_{ij},|q_{ij}|,\rho_{ij}),
\qquad
c_i=\sum_j\alpha_{ij}q_{ij}e_{ij},
\]

并对分布参数 \(\eta=(\mu,\log\sigma)\) 使用：

\[
r_i=N_\theta(o_i,c_i,\eta_i^{\mathrm{base}}),
\qquad
g_i=\tanh(W_gc_i),
\]

\[
\eta_i
=
\eta_i^{\mathrm{base}}
+B\tanh(r_i\odot g_i),
\]

其中 \(W_g\) 不含偏置。于是得到严格的零分支恢复性质：

\[
\boxed{
\{q_{ij}=0\}_{j\in\mathcal N_i}
\Longrightarrow
\pi_{\mathrm{PSB}}(\cdot\mid o_i,\mathbf q_i)
=
\pi_{\mathrm{Base}}(\cdot\mid o_i)
}.
\]

该约束只缩小Actor函数类，不改变近端分岔动力学、增广Markov状态或序列策略梯度。
由于 \(r_i\) 和 \(W_g\) 均由MARL学习，\(q\) 的符号没有被预先绑定为加速、减速、
转向或任何固定物理动作语义。

当任务中的Base探索尺度已经充分校准时，可进一步采用尺度冻结的mean-only子类：

\[
\mu_i
=
\mu_i^{\mathrm{base}}
+B_\mu\tanh(r_{\mu,i}\odot g_{\mu,i}),
\qquad
\log\sigma_i=\log\sigma_i^{\mathrm{base}}.
\]

此时MARL仍通过连续动作均值表达分支条件控制，但不能通过扩大冲突状态下的采样方差
绕过Base探索先验。该约束同样只收紧Actor函数类；近端根、隐式梯度、反对称控制和
CTDE训练结构均保持不变。

若Base策略已经提供可靠的路径跟踪控制，还可以把分支修正投影到冲突协调子空间。对
SigmaRL的原生控制 (a=(v,\delta))，取纵向投影：

\[
P_{\parallel}=\operatorname{diag}(1,0),
\qquad
\mu_i
=
\mu_i^{\mathrm{base}}
+P_{\parallel}B_\mu\tanh(r_{\mu,i}\odot g_{\mu,i}).
\]

于是Base Actor负责空间路径稳定和转向，分岔MARL负责冲突到达时序和纵向协调。这是
对动作修正空间的控制投影，不预设 (q>0) 对应加速或减速；速度修正的符号和大小仍由
MARL学习。投影只删除与协调目标无关的控制自由度，不改变无通信执行、近端唯一性或
隐式梯度理论。

仅有零分支恢复仍允许一种可缩放退化：能量正则把 \(b,z,q\) 压小，而下游网络增大
参数增益，使动作修正保持非零。为使“分岔幅值”与“策略影响”具有不可绕过的量化联系，
定义局部分岔活动度：

\[
a_i
=
\sum_{j\in\mathcal N_i}\alpha_{ij}|q_{ij}|,
\qquad
0\le a_i\le1,
\]

并采用增益有界的扇区门控子类：

\[
\mu_i
=
\mu_i^{\mathrm{base}}
+P_{\parallel}B_\mu a_i
\tanh(r_{\mu,i}\odot g_{\mu,i}),
\qquad
\log\sigma_i=\log\sigma_i^{\mathrm{base}}.
\]

于是得到与网络参数无关的硬约束：

\[
\boxed{
|\Delta\mu_i|
\le
P_{\parallel}B_\mu a_i
}.
\]

特别地，\(a_i\to0\Rightarrow\Delta\mu_i\to0\)，任何 adapter 增益都不能补偿消失的
分岔活动。该约束仍不规定动作修正的符号；分支内采取加速还是减速继续由 MARL 依据
长期回报学习。它只收紧 Actor 函数类，不修改近端势能、单步适定性、隐式 Jacobian、
反对称控制或 CTDE 假设。

若希望意见保留协调记忆，但避免已失去物理冲突支持的残留意见继续干预车辆，则进一步
定义冲突支持系数：

\[
s_{ij}=\frac{\rho_{ij}}{\rho_{\max}}\in[0,1],
\qquad
a_i^\rho
=
\sum_{j\in\mathcal N_i}\alpha_{ij}s_{ij}|q_{ij}|.
\]

将上式中的 \(a_i\) 替换为 \(a_i^\rho\)，得到 urgency-supported sector gate：

\[
|\Delta\mu_i|
\le
P_{\parallel}B_\mu a_i^\rho,
\qquad
\{\rho_{ij}=0\}_{j\in\mathcal N_i}
\Longrightarrow
\Delta\mu_i=0.
\]

这并不在冲突解除时清除 \(z\)：近端意见仍可连续衰减并保留迟滞记忆，只是当前动作
影响被解析冲突图暂时关闭。由此把“协调记忆”和“动作干预资格”分开，保持
物理冲突 \(\rightarrow\) 非线性分岔 \(\rightarrow\) MARL 连续控制的因果链条。

### 29.5 增广中央Critic

训练阶段使用：

\[
V_\psi(x,\operatorname{sg}[\mathbf z],m,\mathcal R,h^{\mathrm{other}}).
\]

不使用动态边表和额外记忆时，可简化为 \(V_\psi(x,\operatorname{sg}[\mathbf z],m)\)。

执行阶段删除Critic。

### 29.6 序列PPO

rollout保存chunk初始边状态、边关联、物理观测、动作和旧log-prob；更新时从chunk初始状态顺序重算 \(b,z\)，在chunk内执行截断BPTT。

---

## 30. 单步算法

```text
Input: physical observation o_t, pair features chi_t,
       previous edge state z_t

1. rho_t = analytic_conflict_map(chi_t)
2. lambda_t = rho_t * nu * alpha / kappa
3. score_ij = PairScorer(chi_ij, z_ij)
4. score_ji = PairScorer(swap(chi_ij), -z_ij)
5. b_t = bounded_antisymmetric_control(score_ij-score_ji, rho_t, lambda_t)
6. z_next = ProxSaturatingNOD(z_t, rho_t, hold_within_solve(b_t), residual_tol=eps_F)
7. branch_context = GraphAggregate(o_t, chi_t, tanh(z_next/z0))
8. activity = Sum(attention * normalized_rho * abs(tanh(z_next/z0)))
9. action_dist = SectorBoundedActor(o_t, branch_context, activity)
10. a_t ~ action_dist
11. x_next, reward = environment.step(a_t)
12. 检查根残差、反对称误差和有限性；rollout时提交z_next
13. 训练时从chunk初态顺序重算，chunk内部不detach
```

`hold_within_solve` 表示Newton/二分迭代中不重新调用PairScorer，不表示对 \(b_t\) 停止梯度；隐式反向仍然通过 \(\partial P/\partial b_t\) 更新 \(\phi\)。

---

## 31. 理论验证实验

理论结果不能只通过最终回报间接证明。至少需要以下验证。

### 31.1 相图验证

扫描 \((\rho,b)\)，绘制数值平衡点和稳定性，与理论边界：

\[
\rho_c=\frac\kappa{\nu\alpha},
\qquad
b_{\mathrm{sw}}(\rho)
\]

比较。

### 31.2 规范形误差

在临界点附近比较完整NOD与三阶normal form轨迹，验证误差随 \(|z|^5\) 缩放。

### 31.3 近端能量

冻结 \((\rho,b)\)，逐步检查：

\[
\mathcal U(z^{t+1})
+\frac{|z^{t+1}-z^t|^2}{2h_z}
\le\mathcal U(z^t).
\]

### 31.4 Jacobian校验

将隐函数解析Jacobian与自动微分、有限差分对比。

### 31.5 迟滞环

准静态增加再减小 \(b\) 或 \(\rho\)，测量实际切换点与理论折叠边界。

### 31.6 临界灵敏度—减速权衡

在多个截止时间 \(T_d\)、目标置信度 \(z_{\mathrm{tar}}\) 和噪声强度下，比较不同 \(\rho\) 的成功概率、到达时间与 \(\sum_tb_t^2\)。目标是验证“静态灵敏度增大”与“临界减速”同时存在，而不预先宣称能量最优点必在 \(\lambda=1\)。

### 31.7 增广Critic

固定近似相同物理状态，比较不同 \(z\) 下的经验回报分布，证明Critic忽略 \(z\) 会产生多峰目标和更大估值误差。

### 31.8 数值求解证书

报告 \(\max|F_t(\widehat z^{t+1})|\)、\(|\widehat z-P|\) 的有限差分估计、能量不等式的最大违反量，并验证违反随 \(\varepsilon_F^2/m_P\) 下降。

### 31.9 有界扰动与势垒

在不同 \(\lambda\) 下注入有界扰动，比较局部偏移界 \(\delta_d/m_s\)、误切换率与势垒 \(\Delta V\)。若注入白噪声，则将结果标为随机经验扩展，不冒充确定性定理。

### 31.10 闭环反馈稳定性

在学习到的常见平衡分支上，统计 \(\partial b_\phi/\partial z\) 与复合近端映射的完整Jacobian，检查其谱半径是否小于1。这项实验用来区分“近端求解适定”与“学习反馈闭环稳定”。

---

## 32. 必要消融

| 消融 | 回答的问题 |
|---|---|
| Base MARL | 基础性能 |
| GRU/LSTM MARL | 改善是否只来自循环记忆 |
| 线性意见滤波 | 分岔是否必要 |
| 直接 \(b\to\) Actor | 吸引子状态是否必要 |
| 多项式pitchfork实现 | 饱和全局动力学是否改善鲁棒性 |
| 显式Euler饱和NOD | 近端离散是否必要 |
| 无 \(b^2\) 代价 | 能量正则化是否抑制持续干预 |
| \(b_\phi\) 不读 \(z\) | 是否无法“先选支、后撤去控制” |
| Critic不读 \(z\) | 增广价值是否必要 |
| 每步detach \(z\) | 长期分岔控制是否需要BPTT |
| 独立双端意见 | 反对称边状态是否必要 |
| 无临界门 | 控制是否集中于高灵敏度区域 |
| 完整PSB-MARL | 完整方法 |

---

## 33. 论文贡献凝练

### 贡献一：协调模式的分岔提升

将多智能体冲突中的离散通行模式表示为受物理冲突临界性控制的饱和非线性意见吸引子，实现“未决—模式形成—承诺保持—中性恢复”的可分析连续动力学。

### 贡献二：近端受控意见算子

提出可微的近端饱和NOD层，在保留原分岔固定点和稳定性分类的同时，保证单步状态转移唯一及冻结输入下的离散能量耗散。

### 贡献三：价值驱动的能量正则化分岔控制

将MARL输出定义为有界、反对称、带控制能量代价的状态反馈分岔控制，而非动作残差或意见标签；通过Pontryagin协态条件和离散Bellman必要条件建立理想增广控制问题与MARL目标的结构联系，不越界声称PPO恢复了精确协态。

### 贡献四：增广多智能体策略

构建分支条件Actor和增广Critic，使意见状态成为CTDE系统的真实Markov状态，并通过序列PPO联合优化分支选择和分支内连续控制。

---

## 34. 最终核心公式

完整理论可以压缩为以下五式。

### 物理临界性

\[
\lambda_e
=
\frac{\rho_e\nu\alpha}{\kappa},
\qquad
\lambda_e=1
\text{为分岔点}.
\]

### 受控饱和势能

\[
\mathcal U_e
=
\frac\kappa2z_e^2
-\frac{\rho_e\nu}{\alpha}
\log\cosh(\alpha z_e)
-b_ez_e.
\]

### 近端意见更新

\[
z_e^{t+1}
=
\arg\min_z
\left[
\frac{(z-z_e^t)^2}{2h_z}
+\mathcal U_e(z;\rho_e^t,b_e^t)
\right].
\]

### 能量正则化分岔控制目标

\[
J
=
\mathbb E
\sum_t\gamma^t
\left[
r_{\mathrm{task}}^t
-\frac{\lambda_b}{M_\star}\|\mathbf b^t\|^2
\right].
\]

其中 \(\mathbf b^t=\mathbf b_\phi(o^t,\mathbf z^t,\mathbf m^t)\)。

### 分支条件策略

\[
a_i^t
\sim
\pi_{\theta_a}
\left(
\cdot\mid
o_i^t,
\{\tanh(z_{ij}^{t+1}/z_0)\}_{j\in\mathcal N_i}
\right).
\]

最终闭环为：

\[
\boxed{
\text{物理冲突}
\rightarrow
\text{临界相变}
\rightarrow
\text{价值驱动的能量正则化选支控制}
\rightarrow
\text{稳定协调吸引子}
\rightarrow
\text{分支内连续动作}
\rightarrow
\text{联合长期回报}.
}
\]

---

## 35. 一句话定位

> 本文将多智能体协调建模为一个增广的随机最优控制问题：物理冲突通过饱和非线性意见势能诱发协调分岔，MARL依据长期联合回报学习带能量代价的分岔状态反馈，近端意见算子将该控制转化为稳定且可恢复的协调承诺，分支条件策略则在受限分散策略类中学习各协调模态内的连续控制。

英文可表述为：

> We formulate multi-agent coordination as an augmented stochastic control problem in which physical conflicts induce bifurcations in a saturating opinion potential, MARL learns an energy-regularized feedback that biases attractor selection, a differentiable proximal opinion operator realizes stable and recoverable commitments, and a branch-conditioned decentralized policy learns continuous control within each coordination mode.
