# R1 Base-MAPPO 训练、测试与产物说明

> 实现状态：代码与文档已完成，实际训练和性能判断由用户手动执行  
> Conda 环境：`sigmarl-nod`  
> 算法路径：SigmaRL 1.2.0 原始向量化 MAPPO

## 1. R1 修改了什么

R1 不增加 Opinion Dynamics，也不改变环境、observation、action、reward、Actor、
Critic、GAE、PPO minibatch 或优化顺序。它只在原始 Base-MAPPO 外围增加：

- 显式随机种子；
- 每次训练独立且不可复用的 run 目录；
- 源配置和实际生效配置快照；
- 每轮 reward、碰撞率、PPO loss、学习率和耗时；
- 训练曲线 PDF、最终模型、完整 checkpoint 和文件哈希清单；
- `latest_run.json`，供标准测试入口定位最近一次成功训练。

因此，R1 的性能目标是建立后续 M2–M11 的 Base 参照，不声明性能提升。

## 2. 环境

```bash
conda activate sigmarl-nod
```

激活后统一使用 `python`，不使用 `.venv/bin/python`。

## 3. 完整 Base 训练

根目录 `config.json` 是完整训练配置，默认输出根目录为 `outputs/base/`：

```bash
python main_training.py
```

每次执行都会新建：

```text
outputs/base/runs/base-seed<seed>-<UTC时间>-<随机后缀>/
```

旧 run 不会被覆盖。只有训练正常完成后，`outputs/base/latest_run.json` 才会指向
这个新 run。

## 4. 小预算 pilot 训练

若想先确认端到端训练、保存和加载闭环，可使用 R1 提供的小预算配置：

```bash
python main_training.py --config configs/base/pilot.json
```

它只用于功能检查，不用于评价模型性能。输出位于 `outputs/base_pilot/` 下的独立
run 目录。

## 5. 测试模型

测试完整训练最近一次成功 run：

```bash
python main_testing.py
```

测试 pilot 最近一次成功 run：

```bash
python main_testing.py --config configs/base/pilot.json
```

测试某个固定的历史 run（便于复现实验，不受 latest 指针变化影响）：

```bash
python main_testing.py --run-dir outputs/base/runs/<run_id>
```

测试入口先读取所选配置的 `where_to_save`，再读取其中的 `latest_run.json`，最后从
该 run 的 `config_resolved.json` 恢复网络和场景参数，并加载 `final_policy.pth`。
测试不会自动选择正在运行或失败的 run。显式使用 `--run-dir` 时，责任主体是用户
指定的这个目录，入口仍要求其中存在可解析配置和最终权重。

默认测试使用实时可视化、单环境、1200 个仿真步，并沿用训练场景。跨场景统一评估
将在 M10 实现。

## 6. 单个成功 run 的产物

```text
<output_root>/
├── latest_run.json
└── runs/
    └── <run_id>/
        ├── config_source.json
        ├── config_resolved.json
        ├── validation_protocol.json
        ├── training_status.json
        ├── metrics.json
        ├── timing.json
        ├── training_curves.pdf
        ├── comparison_to_base.json
        ├── final_policy.pth
        ├── final_base_actor.pth
        ├── final_critic.pth
        ├── final_checkpoint.pt
        ├── artifacts_manifest.json
        └── reward*_data.json / reward*_training_process.pdf
```

说明：

- `final_policy.pth`：保留 SigmaRL 1.2.0 的测试兼容名称；
- `final_base_actor.pth`：同一 Actor 的稳定阶段桥接名称，供后续 Opinion 阶段使用；
- `final_checkpoint.pt`：包含 Actor、Critic、optimizer、iteration、配置和 PyTorch RNG；
- `metrics.json`：每轮 reward、碰撞率、loss、学习率和 wall time；
- `training_curves.pdf`：统一的四面板训练曲线；
- `reward*` 文件：原始 SigmaRL 中间最优模型机制产生的兼容产物；
- `comparison_to_base.json`：R1 将自身登记为 Base reference，后续里程碑再写真实差值。

`training_status.json` 会在训练中持续记录最后完成的 iteration；异常退出时状态为
`failed`。R1 尚未提供 exact-resume 命令，因此 `final_checkpoint.pt` 是完整训练结束
快照，不应被解释为可从任意中断位置精确续训。

## 7. 用户手动检查清单

训练时确认：

1. 进度条持续更新，没有 NaN/Inf 断言；
2. 当前 run 中 `metrics.json` 和 `training_status.json` 持续更新；
3. 训练结束后状态为 `completed`，且 `latest_run.json` 已生成；
4. `training_curves.pdf` 可正常打开；
5. `final_policy.pth`、`final_base_actor.pth`、`final_critic.pth` 和
   `final_checkpoint.pt` 均存在；
6. 随后执行对应的 `python main_testing.py ...` 能加载并显示车辆运行。

性能判断应记录 seed、总 frames、reward、agent/lane/total collision 和 wall time。
后续涉及核心动作逻辑的里程碑必须使用相同预算与本 R1 run 比较。

## 8. R1 边界

- 不包含 Opinion 配置和网络；
- 不包含 TSC topology、priority、leader 或 action predictor；
- 不实现跨场景正式评估；
- 不实现训练中断后的 exact resume；
- 不自动判断 reward/collision 是否优于 Base；
- 用户负责启动训练、测试和作出性能判断。
