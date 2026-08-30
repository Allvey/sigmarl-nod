# A6：单步PPO学习有界合作估计修正

## 目标与边界

A6固定 Base-MAPPO Actor、AVOCADO非线性参数、OCA/道路求解器、自行车适配器和TTC
屏障，只训练 `YCorrectionNet`。Central Critic独立更新以计算优势，价值损失不得进入
Base Actor或修正网络。

历史 `utilities/opinion/evidence_net.py` 不参与A6。A6学习的是启发式合作估计的修正：

\[
y^F=\operatorname{clip}(y^H+\Delta y^{RL},-1,1),
\qquad |\Delta y^{RL}|\leq0.1.
\]

## 单步梯度链

A6不使用序列PPO。Rollout保存更新前的 \(z_t\)，PPO重算时将其视为停止梯度的数据：

\[
\phi_y\rightarrow\Delta y_t^{RL}\rightarrow y_t^F
\rightarrow z_{t+1}\rightarrow\Delta\mu_{op,t}
\rightarrow\log\pi_t.
\]

意见残差只修改Base Actor的纵向均值，转向均值保持不变。环境实际执行经AVOCADO、
道路约束、自行车适配和TTC屏障处理后的动作；PPO记录和重算的是安全层之前的MARL
名义动作及其log-prob。

## 入口

正式配置：

```bash
conda run --no-capture-output -n sigmarl-nod python main_training_avocado_marl_a6.py \
  --config configs/avocado_marl/a6_y_correction.json
```

快速冒烟配置：

```bash
conda run --no-capture-output -n sigmarl-nod python main_training_avocado_marl_a6.py \
  --config configs/avocado_marl/a6_y_correction_pilot.json
```

确定性评估：

```bash
conda run --no-capture-output -n sigmarl-nod python main_testing_avocado_marl_a6.py \
  --config configs/avocado_marl/a6_y_correction.json \
  --checkpoint <a6-run>/final_checkpoint.pt
```

实时可视化会自动切换为单环境，并在画面中显示名义/执行动作、\(\Delta y\)、\(y^F\)、
\(z\)、活动VO、TTC屏障和车辆重置：

```bash
conda run --no-capture-output -n sigmarl-nod python main_testing_avocado_marl_a6.py \
  --config configs/avocado_marl/a6_y_correction.json \
  --render --scenario intersection_2 --max-steps 600
```

不指定 `--scenario` 时使用A6训练配置中的场景。批量数值评估保持配置中的并行环境数；
`--render` 模式固定为一个环境。

恢复训练时，`--iterations` 表示恢复后的总目标迭代数：

```bash
conda run --no-capture-output -n sigmarl-nod python main_training_avocado_marl_a6.py \
  --config configs/avocado_marl/a6_y_correction.json \
  --resume <a6-run>/latest_checkpoint.pt --iterations 50
```

## 审计项

- A6配置必须显式绑定同一Base运行目录内的Actor和Critic checkpoint；
- Base Actor保存源哈希，训练结束再次校验；
- 无效车辆对的修正严格为0；
- checkpoint保存修正网络、Critic、optimizer、RNG、控制器随机数状态和迭代数；
- 训练指标包含修正幅值、饱和率、符号切换率、梯度范数、碰撞和屏障干预率；
- A6的性能结论必须与A5在同预算、多随机种子下比较，短pilot只证明训练链可运行。

## A5/A6单checkpoint配对评估

统一比较入口会让A5和A6使用相同的Base Actor checkpoint、场景、seed、并行环境数和
物理步数。默认配置评估 `CPM_mixed` 与 `intersection_2`，使用5个配对环境seed、每个
seed 4个并行环境和600步：

```bash
conda run --no-capture-output -n sigmarl-nod \
  python main_comparing_avocado_marl_a6.py \
  --config configs/avocado_marl/a6_comparison.json \
  --checkpoint <a6-run>/final_checkpoint.pt
```

在正式实验前可用单seed短预算检查完整入口；该结果不能用于性能结论：

```bash
conda run --no-capture-output -n sigmarl-nod \
  python main_comparing_avocado_marl_a6.py \
  --scenarios CPM_mixed --seeds 0 --parallel-envs 1 --max-steps 20
```

每次比较生成逐stage/场景/seed的JSON、`summary.json`和便于阅读的`report.md`。所有配对
差值定义为 `A6 - A5`：奖励、完成率和速度越高越好；碰撞、路径误差、屏障率、动作
变化、冲突窗口转向反向率和停车率越低越好。只有至少两个环境seed时才给方向性结论；
默认5个环境seed用于判断当前checkpoint是否具有稳定正向趋势，并要求配对95%置信区间
不跨0。安全指标不能因奖励改善而被忽略，若碰撞或车道事件恶化，应先判定当前A6
checkpoint未通过。TTC分箱结果用于检查修正是否集中在真实冲突阶段，不作为单独的性能
通过条件。

这里的多seed是环境初始状态seed，不是独立训练seed。它适合筛查当前checkpoint，但不能
替代最终的多训练seed实验。只有当前checkpoint先表现出正向趋势，才值得使用相同总采样
预算分别训练多个A6 seed，并把每个训练seed对应的checkpoint作为独立统计样本。
