# M5：Base checkpoint、Direct Evidence 与 Policy Bridge

> 实现状态：已完成；训练和性能由用户手动验证  
> 本阶段首次改变动作分布  
> 后续状态：M6 Stateful Collector 已完成，见 `M6_STATEFUL_OPINION.md`

## 1. M5 的定位

M5 将 M3 的 `OpinionEvidenceNet/OpinionResidual`、M4 的真实车辆对信息和已经训练好的
SigmaRL Base Actor 接起来。第一版仍不保存跨时间状态，而是显式采用：

```text
z_direct = b
```

因此 M5 是可训练的 Direct-Evidence 中间模型，不是最终的连续意见动力学模型。

## 2. 前向结构

```text
observation ─> frozen Base Actor ─> base_loc, base_scale

pair_features, urgency, confidence, pair_mask
        └─> OpinionEvidenceNet ─> raw_b, b
                                      │
                                      └─> z_direct=b
                                              │
                                              └─> OpinionResidual ─> delta_mu

final_loc.speed = base_loc.speed + delta_mu
final_loc.steer = base_loc.steer
final_scale     = base_scale

final_loc, final_scale ─> TanhNormal ─> action, sample_log_prob
```

实现位于 [`../../utilities/opinion/policy.py`](../../utilities/opinion/policy.py)。

## 3. Base checkpoint 合同

M5 配置中的 `policy_bridge.base_output_root` 指向 R1 Base 输出根目录。训练入口通过
`latest_run.json` 优先固定最近一次成功 run，并要求成对的 Actor/Critic：

```text
final_policy.pth（run 中同时保留 final_base_actor.pth 作为跨阶段语义副本）
final_critic.pth
```

Base Actor 使用 `strict=True` 加载并完全冻结。Critic 结构和输入不变，从 Base Critic
初始化后继续训练。解析出的精确 run 和 checkpoint 路径会写入
`opinion_config_resolved.json`，不会只记录易漂移的 latest 指针。

如果尚无 completed Base，入口会回退到最新一个可测试的 running/failed run，选择
reward 数值最高且名称严格匹配的一对：

```text
reward<value>_policy.pth
reward<value>_critic.pth
```

只存在 Policy、缺少同 reward 的 Critic 时会拒绝启动。快照会额外记录
`resolved_base_run_status` 和 `resolved_base_checkpoint_kind`，因此不会把中间 Base
误记成最终 Base。该回退用于开发闭环和较早观察 M5 行为；正式性能比较仍应先完整
训练 Base，否则 M5 的表现会混入“Base 尚未收敛”的影响。

M5 不支持中途 resume；但支持测试在训模型。测试入口优先加载 M5 run 的
`final_policy.pth`；最终文件尚未生成时，会加载 reward 最优的中间
`reward<value>_policy.pth`。也可以通过 `--run-dir` 或 `--checkpoint` 精确选择。
测试加载与上述训练回退相互独立：测试只需 Policy；启动 M5 训练则必须具有配置兼容、
命名匹配且来自同一个 Base run 的 Actor/Critic 对。

## 4. 优化边界

M5 继续使用 SigmaRL 原始扁平 PPO，不使用 Sequence Buffer。可训练参数只有：

```text
EvidenceNet       可训练，lr = Base lr * evidence_learning_rate_scale
Central Critic    可训练，lr = Base lr
Base Actor        冻结
OpinionResidual   无参数
OpinionDynamics   不运行
```

PPO 的新旧 log-prob 均由最终 residual 修正后的 TanhNormal 计算，不能使用 Base
分布的 log-prob。

## 5. 配置

完整预算：

```text
configs/opinion/m5_direct_evidence.json
```

pilot：

```text
configs/opinion/m5_direct_evidence_pilot.json
```

关键字段：

```json
"stage": "evidence",
"use_opinion_marl": true,
"conflict_graph": {"emit_pair_info": true},
"policy_bridge": {
    "enabled": true,
    "mode": "direct_evidence",
    "base_output_root": "outputs/base_pilot/",
    "freeze_base_actor": true,
    "visualize_agent_id": 0
},
"stateful": {
    "enabled": false,
    "evidence_output_root": null,
    "freeze_evidence": false,
    "zero_threshold": 1e-6
}
```

完整配置使用 `outputs/base/`，pilot 使用 `outputs/base_pilot/`，避免误把不同训练预算
的 Base 权重混用。

## 6. 手动训练顺序

先保证对应 Base 已经成功训练。pilot 顺序：

```bash
conda activate sigmarl-nod

python main_training.py --config configs/base/pilot.json
python main_training_opinion.py \
  --config configs/opinion/m5_direct_evidence_pilot.json
python main_testing_opinion.py \
  --config configs/opinion/m5_direct_evidence_pilot.json
```

完整预算：

```bash
python main_training.py
python main_training_opinion.py \
  --config configs/opinion/m5_direct_evidence.json
python main_testing_opinion.py \
  --config configs/opinion/m5_direct_evidence.json
```

如果 Base 已经训练完成，不需要为了重复 M5 实验重新训练 Base；M5 会从指定输出根
目录的最近成功 Base run 开始。

## 7. 产物

M5 run 在原 R1 产物基础上增加：

```text
source_base_actor.pth            # stage 启动即保存，训练中断也保留
reward<value>_evidence_net.pth  # 每次刷新最优中间策略时保存
final_evidence_net.pth          # EvidenceNet 独立 state_dict
final_opinion_policy.pth
final_checkpoint.pt:
  stage = evidence_direct
  base_actor_state
  evidence_state
  opinion_policy_state
  critic_state
  optimizer_state
  opinion_runtime_config
```

`reward<value>_policy.pth` 和 `final_policy.pth` 都是完整 Opinion Policy state_dict，
其中已经包含冻结 Base Actor 与 EvidenceNet；独立的 `*_evidence_net.pth` 用于参数分析、
消融和后续阶段初始化。`final_policy.pth` 保留为测试兼容入口；M5 run 中的
`final_base_actor.pth` 是冻结 Base Actor 的原始状态，而不是 Opinion Policy 的别名。
`source_base_actor.pth` 在进入训练循环前立即写入，防止来源是 reward 中间 checkpoint
且随后被 Base 的更优 checkpoint 清理，保证 M6 仍能复用完全相同的冻结 Actor。

M3 的 `OpinionDynamics` 与 `OpinionResidual` 当前没有可训练参数，因此不会生成对应
权重文件；M4 的 ConflictGraph 是确定性物理计算，也没有网络权重。M5 当前新增且真正
被优化的 Actor 侧网络只有 EvidenceNet。

`metrics.json` 额外记录：

```text
raw_b_abs_mean
gated_b_abs_mean
speed_residual_abs_mean
active_pair_fraction
evidence_learning_rate
critic_learning_rate
```

## 8. 可视化面板

`main_testing_opinion.py` 会将 `policy_bridge.visualize_agent_id` 指定车辆的诊断量显示
在 VMAS 界面。默认固定车辆 0：

```text
Opinion diagnostics | ego=0 | speed residual=-0.0312
j=2 | tCPA=1.42s dCPA=0.81m mask=1 rho=0.56 conf=0.73
  raw_b=-0.48 b=-0.20 z_direct=-0.20 z_stateful=N/A (M6)
```

车辆 ID 只用于显示和后续状态关联，不进入 Base Actor 或 EvidenceNet。

## 9. M5 的性能解释

M5 首次改变动作，因此可以与同 seed、同预算 Base 比较 reward、collision 和 wall
time。但单个 pilot 只用于发现方向和数值问题，不能形成论文结论。

首先检查：

- inactive pair 时 residual 必须严格为 0；
- 转向 loc 和 scale 与 Base 一致；
- residual 始终满足配置边界；
- Base Actor 不产生梯度；
- EvidenceNet 有有限梯度；
- `raw_b/b/residual` 不是长期全零或饱和；
- 碰撞率没有灾难性回退。

参考测试为 `tests/opinion/test_m5_policy_bridge.py`。按项目约定，本 Session 不运行
测试、训练或性能评估。

## 10. M6 交接（已完成）

M6 已用 `neighbor_ids` 和 `agent_reset_mask` 建立 `[E,N,N]` 的 `z_dense`，把
`z_direct=b` 替换成固定 OpinionDynamics 的逐步更新，同时保留 M5 配置和 checkpoint
加载路径作为 Direct Evidence 消融。详见 [`M6_STATEFUL_OPINION.md`](M6_STATEFUL_OPINION.md)。
