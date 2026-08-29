# A5：零修正网络等价性

## 目标

A5在A4的启发式合作估计与非线性意见更新之间加入正式融合接口：

\[
y^F=\operatorname{clip}(y^H+\Delta y^{RL},-1,1).
\]

本阶段实例化共享 `YCorrectionNet`，但最后一层严格零初始化、所有参数冻结，因此
\(\Delta y^{RL}\equiv0\)。A3与A4不传修正张量，原有行为保持不变。

网络输入是每个有向车辆对的14维局部特征：相对位置、相对速度、双方速度、相对航向、
TTC、CPA距离、注意力、启发式 \(y^H\)、上一拍VO修正模长和有效mask。网络不读取
意见状态 \(z\)、未来信息、全局ID或Critic状态。第一版只允许两个最近的有效邻车产生
修正。

## 验证

快速逐步等价测试：

```bash
conda run -n sigmarl-nod python main_testing_avocado_marl_a5.py \
  --episodes 1 --max-steps 100
```

程序分别从相同随机种子运行A4和A5，要求下列序列最大差值严格为0：

- MARL名义动作和最终执行动作；
- 注意力 \(A\) 与有效车辆对mask；
- 启发式估计 \(y^H\) 与融合估计 \(y^F\)；
- 非线性意见状态 \(z\)。

输出目录包含 `summary.json` 和每个场景的 `trace_<case>.pt`。trace保存完整的
\(A,y^H,\Delta y,y^F,z\) 以及名义/执行动作序列。

实时可视化：

```bash
conda run -n sigmarl-nod python main_testing_avocado_marl_a5.py \
  --render --scenario intersection_2 --max-steps 600
```

配置入口为 `configs/avocado_marl/a5_zero_correction.json`。A5只验证接口等价性，不训练
网络，也不用于宣称性能提升。
