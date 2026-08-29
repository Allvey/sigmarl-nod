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
