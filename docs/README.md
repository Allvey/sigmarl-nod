# 项目文档导航

本文档目录以 **SigmaRL 1.2.0 为唯一代码底座**，并将新的 Opinion Dynamics +
MARL 方法与旧 TSC 实现明确分开。

## 新 Session 的阅读顺序

1. [`opinion/opinion_dynamics_marl_technical_route.md`](opinion/opinion_dynamics_marl_technical_route.md)
2. [`opinion/OPINION_MARL_IMPLEMENTATION_GUIDE.md`](opinion/OPINION_MARL_IMPLEMENTATION_GUIDE.md)
3. [`opinion/OPINION_MARL_NETWORK_ARCHITECTURE.md`](opinion/OPINION_MARL_NETWORK_ARCHITECTURE.md)
4. [`opinion/M2_CONFIG_AND_ENTRYPOINTS.md`](opinion/M2_CONFIG_AND_ENTRYPOINTS.md)
5. [`opinion/M3_MATH_MODULES.md`](opinion/M3_MATH_MODULES.md)
6. [`opinion/M4_CONFLICT_GRAPH.md`](opinion/M4_CONFLICT_GRAPH.md)
7. [`opinion/M5_POLICY_BRIDGE.md`](opinion/M5_POLICY_BRIDGE.md)
8. [`opinion/M6_STATEFUL_OPINION.md`](opinion/M6_STATEFUL_OPINION.md)
9. [`opinion/M7_SEQUENCE_BUFFER.md`](opinion/M7_SEQUENCE_BUFFER.md)
10. [`opinion/M8_SEQUENCE_PPO.md`](opinion/M8_SEQUENCE_PPO.md)
11. [`opinion/M9_TRAINER_AND_CHECKPOINT.md`](opinion/M9_TRAINER_AND_CHECKPOINT.md)
12. [`sigmarl_1_2_0/CODEBASE_AUDIT.md`](sigmarl_1_2_0/CODEBASE_AUDIT.md)
11. [`sigmarl_1_2_0/R0_USAGE.md`](sigmarl_1_2_0/R0_USAGE.md)
12. [`sigmarl_1_2_0/R1_BASE_ARTIFACTS.md`](sigmarl_1_2_0/R1_BASE_ARTIFACTS.md)
13. [`sigmarl_1_2_0/RL_ENVIRONMENT_DESIGN.md`](sigmarl_1_2_0/RL_ENVIRONMENT_DESIGN.md)
14. [`sigmarl_1_2_0/OBSERVATION_SPACE_DETAILS.md`](sigmarl_1_2_0/OBSERVATION_SPACE_DETAILS.md)
15. [`sigmarl_1_2_0/NETWORK_STRUCTURE_DETAILS.md`](sigmarl_1_2_0/NETWORK_STRUCTURE_DETAILS.md)

## 目录职责

```text
docs/
├── README.md
├── opinion/
│   ├── opinion_dynamics_marl_technical_route.md
│   ├── OPINION_MARL_IMPLEMENTATION_GUIDE.md
│   ├── OPINION_MARL_NETWORK_ARCHITECTURE.md
│   ├── M2_CONFIG_AND_ENTRYPOINTS.md
│   ├── M3_MATH_MODULES.md
│   ├── M4_CONFLICT_GRAPH.md
│   ├── M5_POLICY_BRIDGE.md
│   ├── M6_STATEFUL_OPINION.md
│   ├── M7_SEQUENCE_BUFFER.md
│   ├── M8_SEQUENCE_PPO.md
│   └── M9_TRAINER_AND_CHECKPOINT.md
├── sigmarl_1_2_0/
│   ├── CODEBASE_AUDIT.md
│   ├── R0_USAGE.md
│   ├── R1_BASE_ARTIFACTS.md
│   ├── RL_ENVIRONMENT_DESIGN.md
│   ├── OBSERVATION_SPACE_DETAILS.md
│   └── NETWORK_STRUCTURE_DETAILS.md
├── papers/
│   ├── AVOCADO_2025.pdf
│   └── TASE-TSC.pdf
└── archive_tsc/
    └── 旧 TSC 文档，仅用于追溯，不作为当前实现依据
```

### `opinion/`

当前研究方法的规范性文档。

- 技术路线定义理论、变量语义和不可破坏的因果结构；
- 实施指南定义从 SigmaRL 1.2.0 开始的文件边界、阶段、测试和交接规则。

若两者冲突，以技术路线为准；工程实现不得反向修改理论语义来迁就旧 TSC 代码。

### `sigmarl_1_2_0/`

只描述 tag `1.2.0` 的原始环境、观测、动作、Actor、Critic 和 PPO 数据流。
其中的数值以该 tag 的 `config.json` 为基准，而不是以后新增的 Opinion 配置。

### `archive_tsc/`

这里的文件包含 topology、priority、leader、action predictor、opponent modeling 或
Stackelberg 等旧设计。它们可以帮助理解历史决策和构造外部 TSC 基线，但：

- 不是当前方法规范；
- 不用于决定 Opinion 模块接口；
- 不应被新 Session 当作待恢复代码清单；
- 旧 PDF 也不代表 SigmaRL 1.2.0 的观测空间。

### `papers/`

保存外部方法资料。TSC 在本项目中是外部比较对象，不是 Opinion-MARL 的理论底座。
其中 AVOCADO 是意见动力学的启发来源，TASE-TSC 仅用于理解外部 TSC
基线。两者都不直接定义当前代码接口。

## 当前基线状态

当前根目录以 SigmaRL 1.2.0 原始源码为底座；恢复时的核心代码、配置和资源已与
基线 commit 逐文件核对，且真实 reset 和 3-step rollout 已通过。此后只叠加了本
指南记录的 R0/R1/M2/M3/M4/M5/M6/M7/M8/M9 修改，没有引入 TSC；M6 已按 global agent ID
维护连续意见状态，M7 建立不跨环境或 episode 的连续 chunk Buffer，M8 已沿 chunk
展开固定 Dynamics，并用 PPO Actor loss 训练 EvidenceNet；M9 已支持 Evidence 独立训练、
Base/Evidence/Critic 从零完整联合训练、历史权重微调和 warmup 自动切换。正式主训练使用
`configs/opinion/m9_joint_from_scratch.json`，不依赖任何 Base/M5–M8 checkpoint，且只
消耗一次与 SigmaRL Base 相同的 250-iteration 环境预算。

现有 `sigmarl-nod` Conda 环境已经实施 user-site 隔离和依赖补全。R0、R1 代码实现
已完成；实际训练和测试由用户按照
[`sigmarl_1_2_0/R1_BASE_ARTIFACTS.md`](sigmarl_1_2_0/R1_BASE_ARTIFACTS.md)
手动执行。

## 基线声明

当前重建基线固定为：

```text
repository: https://github.com/bassamlab/SigmaRL
tag:        1.2.0
commit:     5fe715bdfba4ff3e33d901d69dfa220f1222c060
```

下一实现步骤为 M10 评估、诊断和可视化。旧 TSC 文件不得批量复制回根目录。
