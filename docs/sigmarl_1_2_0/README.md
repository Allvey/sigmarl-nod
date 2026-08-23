# SigmaRL 1.2.0 事实文档

本目录只描述上游 tag `1.2.0`，不描述旧 TSC 工作树。

基线：

```text
repository: https://github.com/bassamlab/SigmaRL
tag:        1.2.0
commit:     5fe715bdfba4ff3e33d901d69dfa220f1222c060
```

文档：

- [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md)：当前目录的源码差异、环境版本、
  smoke 证据和尚未通过的 R0 Gate；
- [`RL_ENVIRONMENT_DESIGN.md`](RL_ENVIRONMENT_DESIGN.md)：环境、动作、奖励、reset 和
  原始向量化训练数据流；
- [`OBSERVATION_SPACE_DETAILS.md`](OBSERVATION_SPACE_DETAILS.md)：默认 32 维局部
  观测及 Opinion 接口边界；
- [`NETWORK_STRUCTURE_DETAILS.md`](NETWORK_STRUCTURE_DETAILS.md)：原始 Actor、Critic、
  PPO 和目标 Opinion 网络结构。

核对方式包括 tag 源码审计，以及在 `CPM_mixed` 上真实创建 2 个 VMAS 环境、
reset 并执行 3 个随机动作步。
实测结果：

```text
observation key  ("agents", "observation")
observation      [2,4,32]
action           [2,4,2]
3-step rollout   finite
```

上述 smoke 已通过，但 Conda 环境仍依赖 user-site 中的部分包，所以完整 R0
尚未结束。不得只根据 rollout 成功就宣称依赖已可复现。

以后如果基础 tag 改变，必须建立新的版本目录，不得直接覆盖本目录并继续沿用
`sigmarl_1_2_0` 名称。
