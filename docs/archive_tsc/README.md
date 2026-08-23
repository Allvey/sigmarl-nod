# 旧 TSC 文档归档

本目录保存 2026-08-23 以前基于 TSC 工作树形成的说明和早期 Opinion 设计。

这些文件没有删除，是为了保留历史分析、图表和设计取舍；但它们不再描述当前目标
代码底座，也不构成实现要求。

归档内容包含下列旧概念：

- TopologyLearner / topology labels；
- TopologyActionPredictor；
- opponent modeling；
- priority policy/critic；
- leader/follower 与 Stackelberg；
- 在 TSC 上接入 Opinion 的旧里程碑。

根目录整理时保留的主要文件：

- `MODEL_STRUCTURE_TSC_legacy.md`；
- `NETWORK_STRUCTURE_DETAILS_TSC_legacy.md`；
- `OBSERVATION_SPACE_DETAILS_TSC_legacy.md` 及其旧 PDF；
- `RL_ENVIRONMENT_DESIGN_TSC_legacy.md`；
- `OPINION_MARL_IMPLEMENTATION_GUIDE_PRE_RESET_legacy.md`；
- `opinion_dynamics_marl_technical_route_pre_alignment_legacy.md`；
- `Opinion_Dynamics_MARL_Design_TSC_based_legacy.md`。

当前规范请阅读：

- [`../opinion/opinion_dynamics_marl_technical_route.md`](../opinion/opinion_dynamics_marl_technical_route.md)
- [`../opinion/OPINION_MARL_IMPLEMENTATION_GUIDE.md`](../opinion/OPINION_MARL_IMPLEMENTATION_GUIDE.md)
- [`../sigmarl_1_2_0/`](../sigmarl_1_2_0/)

禁止把本目录文件复制回根目录并据此恢复旧代码。TSC 只在正式实验中作为外部基线。
