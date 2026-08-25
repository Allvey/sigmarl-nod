# M9：统一独立/联合 Trainer 与 Checkpoint

> 状态：实现完成；性能由用户手动验证  
> 前置能力：M8 可微分 Sequence PPO  
> 默认主训练：`joint_from_scratch`

## 1. M9 的定位

M9 不新增决策网络，而是统一控制三个可训练网络：

```text
Base Actor + EvidenceNet + Central Critic
```

固定模块仍然是 ConflictGraph、OpinionDynamics、OpinionResidual、StateTracker 和
Sequence Buffer。M9 支持独立训练和完整联合训练，不把分阶段训练写成理论必需条件。

## 2. 五个正式配置

| 配置 | 初始化 | Base Actor | EvidenceNet | Critic |
|---|---|---:|---:|---:|
| `m9_joint_from_scratch.json` | 全部随机 | 训练 | 训练 | 训练 |
| `m9_evidence_only.json` | Base | 冻结 | 训练 | 训练 |
| `m9_joint_from_base.json` | Base | 训练 | 训练 | 训练 |
| `m9_joint_from_m8.json` | M8/M9 Opinion | 训练 | 训练 | 训练 |
| `m9_warmup_then_joint.json` | Base | 先冻结后训练 | 训练 | 训练 |

Base 初始化时，EvidenceNet 使用近中性的随机初始化，不依赖 M5–M8 长训练。Opinion
初始化严格要求来源是 M8/M9 Stateful Sequence-PPO，并核对 Evidence、Dynamics、
Residual 配置。

`joint_from_scratch` 不读取 `outputs/base`、M5、M8 或任何历史 checkpoint。M3–M8
实现的 ConflictGraph、EvidenceNet、OpinionDynamics、StateTracker、Sequence Buffer
和可微分时序 PPO 仍完整参与计算，但它们不再代表必须依次训练的前置阶段。

## 3. 推荐训练方式

正式主训练从零开始，只消耗一次与 SigmaRL Base 相同的 250-iteration 预算：

```bash
conda activate sigmarl-nod

python main_training_opinion.py \
  --config configs/opinion/m9_joint_from_scratch.json
```

快速检查完整训练闭环：

```bash
python main_training_opinion.py \
  --config configs/opinion/m9_joint_from_scratch_pilot.json
```

正式配置直接复用根目录 `config.json`：`n_iters=250`、`frames_per_batch=4096`、
`num_epochs=60`、`minibatch_size=512`、`lr=2e-4`、`lr_min=1e-5`。Actor、
EvidenceNet 和 Critic 从第 1 轮同时更新，三组学习率比例均为 `1.0`，不设 warmup，
不使用 Base anchor。

从 Base warmup 后联合训练仍作为对照模式保留：

```bash
python main_training_opinion.py \
  --config configs/opinion/m9_warmup_then_joint.json
```

直接从 Base 联合训练：

```bash
python main_training_opinion.py \
  --config configs/opinion/m9_joint_from_base.json
```

从最近 M8 继续联合微调：

```bash
python main_training_opinion.py \
  --config configs/opinion/m9_joint_from_m8.json
```

只训练 EvidenceNet + Critic：

```bash
python main_training_opinion.py \
  --config configs/opinion/m9_evidence_only.json
```

## 4. 参数组

从零主训练的学习率相对 Base JSON 中的 `lr`：

```text
Base Actor     1.00 × lr
EvidenceNet    1.00 × lr
Central Critic 1.00 × lr
```

其他 M9 初始化模式仍保留各自配置中的 Evidence 学习率比例，当前为 `0.10 × lr`。

Evidence warmup 期间 Base 参数 `requires_grad=false` 且 Base 参数组学习率严格为 0；切换
Joint 后在原 optimizer 中激活，不重置 Evidence/Critic 的 Adam 状态。

联合 Actor loss 同时更新 Base Actor 与 EvidenceNet，Evidence 正则只约束 EvidenceNet，
Critic loss 只更新 Critic。可选 `base_anchor_coefficient` 限制 Base loc 过快偏离初始化
策略；Evidence-only 配置中该系数为 0。

## 5. Checkpoint 与恢复

每轮原子更新：

```text
latest_checkpoint.pt
```

每 `checkpoint_interval` 轮额外保存：

```text
checkpoint_iteration_000050.pt
checkpoint_iteration_000100.pt
...
```

最终保存 `final_checkpoint.pt`。其中包含：

- 完整 Policy、Critic 和 optimizer；
- iteration、training mode、当前 phase；
- 历史 metrics；
- Python、NumPy、Torch 随机状态；
- Opinion runtime 配置、Base anchor 和终止意见状态。

恢复示例：

```bash
python main_training_opinion.py \
  --config configs/opinion/m9_joint_from_scratch.json \
  --resume outputs/opinion/m9_joint_from_scratch/runs/<run-id>/latest_checkpoint.pt
```

恢复必须使用同一输出根目录和同一训练模式，`n_iters` 必须大于 checkpoint iteration。
恢复会在原 run 目录追加指标并只运行剩余 iteration。

恢复边界：模型、optimizer、阶段、指标和随机状态会恢复；VMAS 世界不会恢复到中断的
物理帧，而是从新的 rollout 边界继续。因此这是训练状态精确恢复，不是交通仿真世界的
逐比特帧恢复。

## 6. 训练产物和诊断

除已有 Policy/Critic/EvidenceNet/PDF 外，M9 指标新增：

```text
training_mode
training_phase
base_actor_trainable
base_actor_learning_rate
evidence_learning_rate
critic_learning_rate
base_actor_gradient_norm
evidence_gradient_norm
loss_base_anchor
```

Evidence warmup 中预期：

```text
base_actor_trainable=false
base_actor_learning_rate=0
base_actor_gradient_norm=0
evidence_gradient_norm>0
```

Joint 中预期 Base/Evidence 梯度均为有限非零值。梯度范数是在统一 gradient clipping
之前记录，用于诊断，不代表实际更新幅度。

## 7. 测试

```bash
python main_testing_opinion.py \
  --config configs/opinion/m9_joint_from_scratch.json
```

测试精确 run：

```bash
python main_testing_opinion.py \
  --config configs/opinion/m9_joint_from_scratch.json \
  --run-dir outputs/opinion/m9_joint_from_scratch/runs/<run-id>
```

## 8. 工程验证与性能边界

实现时已经完成：从零 Joint 两轮、直接 Joint 两轮、warmup→Joint 两轮、同 run
checkpoint 恢复、从在训 M8 中间 Policy/Critic 初始化、final policy 加载。从零 pilot
确认没有 Base 来源字段，Actor/Evidence/Critic 从第 1 轮同时获得有限梯度，并生成完整
Policy/Critic/Evidence/checkpoint/PDF 产物。自动环境无显示器时仍会在 pyglet 渲染初始化
停止，本地桌面可视化不受影响。

这些结果只证明训练闭环。正式比较以 `joint_from_scratch` 和 Base 各自一次 250-iteration
预算为主；`evidence_only`、`joint_from_base`、`joint_from_m8` 和
`warmup_then_joint` 作为初始化与训练调度消融。
