# A4：Base-MAPPO 与固定 AVOCADO 的动作级耦合

## 目标

A4只验证动作接口，不训练意见动力学。Base-MAPPO输出名义自行车动作，固定的A3 AVOCADO-KB负责冲突协调，之后再经过自行车动作适配与TTC紧急保护：

\[
o_i \to [v_i^{nom},\delta_i^{nom}]
\to \mathbf v_i^{nom}
\to \mathrm{AVOCADO}(A,y^H,z)
\to [v_i^{exec},\delta_i^{exec}].
\]

本阶段保持 `opinion_bias=0`，不存在EvidenceNet或学习得到的 \(\Delta y^{RL}\)。A4使用独立的 `coupling.velocity_continuity_weight=0`，避免A3为启发式控制器增加的速度平滑项在无冲突时持续削弱MARL动作；A3配置本身不受影响。

## 实时可视化

在 `main_testing_avocado_marl.py` 中修改 `TEST_SCENARIO_TYPE`，然后运行：

```bash
conda run -n sigmarl-nod python main_testing_avocado_marl.py --render
```

也可以从命令行选择场景、策略checkpoint和步数：

```bash
conda run -n sigmarl-nod python main_testing_avocado_marl.py \
  --render --scenario intersection_2 --max-steps 600
```

界面同时显示MARL名义动作、最终执行动作、动作是否被修改、活跃VO、注意力、意见状态和TTC shield介入情况。

## 可复现实验

运行同种子下的原始Base-MAPPO与A4混合控制对照：

```bash
conda run -n sigmarl-nod python main_testing_avocado_marl.py \
  --episodes 4 --max-steps 600
```

结果保存在 `outputs/avocado_marl/a4_action_coupling/<timestamp>/summary.json`。重点检查：

- 原始Base动作透传率为100%；
- A4动作全部有限；
- 名义/执行动作相关性、无冲突透传率与冲突介入率；
- 碰撞、路线完成、路径误差、平均速度和TTC shield介入率。

配置入口为 `configs/avocado_marl/a4_base_avocado.json`。`run_directory` 与 `checkpoint` 为空时自动选择 `outputs/base` 下最新的可测试Base运行，也可通过命令行显式指定。
