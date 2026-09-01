# PSB-MARL P2.1-U：Urgency-Supported Sector Gate

P2.1-U 在 P2.1-G 固定增益界上增加一个无参数的物理冲突支持门。意见状态可以在冲突
解除后继续保留记忆，但只有当前解析冲突强度非零时，意见才允许修正动作：

\[
q_{ij}=\tanh(z_{ij}/z_0),
\qquad
s_{ij}=\rho_{ij}/\rho_{\max}\in[0,1],
\]

\[
a_i^\rho
=
\sum_j\alpha_{ij}s_{ij}|q_{ij}|,
\qquad
\mu_i
=
\mu_i^{\mathrm{Base}}
+P_{\parallel}B_\mu a_i^\rho
\tanh(r_{\mu,i}\odot g_{\mu,i}).
\]

这给出两条硬性质：

\[
|\Delta\mu_i|\le P_{\parallel}B_\mu a_i^\rho,
\qquad
\{\rho_{ij}=0\}_j\Longrightarrow\Delta\mu_i=0.
\]

因此，“意见记忆是否存在”和“意见当前是否有权影响车辆”被明确分离。前者仍由近端
饱和分岔动力学决定，后者由当前物理冲突提供支持。该门没有新增可学习参数，也没有把
正负意见预先绑定到加速或减速。

## 1. P2.1-G 正式结果

P2.1-G run：

```text
outputs/psb_marl/p2_1_g_sector_projected_mean_only/runs/
psb-p2-seed0-20260831T160701321843Z-c4cabb50
```

5-seed、每 seed 4 episodes 的 paired Base 测试结果：

| Candidate - Base | 均值差 | 置信边界 | 结果 |
|---|---:|---:|---|
| 奖励 | `+0.000175` | 下界 `-0.000774` | 通过 |
| 车辆碰撞 | `-0.000501` | — | 改善 |
| 车道碰撞 | `+0.001252` | — | 恶化 |
| 总碰撞 | `+0.000723` | 上界 `+0.002099` | 未通过 |

总碰撞上界只比 `+0.002` 门限高约 `9.9e-5`，且所有结构门均通过，但当前协议仍必须
判为失败，不能 promotion。恶化完全来自车道碰撞；这与残留意见在当前冲突已弱或消失
时仍可改变速度、进而间接改变后续路径跟踪状态的机制相符。P2.1-U 用当前解析冲突支持
收紧动作作用域，而不修改近端意见记忆。

## 2. 实现合同

```text
configs/psb_marl/p2_1_u_urgency_supported_sector.json
outputs/psb_marl/p2_1_u_urgency_supported_sector/
```

关键字段：

```json
"conditioning_mode": "supported_sector_q_gate",
"max_delta_log_scale": 0.0,
"action_projection": "longitudinal_only"
```

P2.1-U 继续要求：

```text
rollout_sector_bound_max_violation <= 1e-7
rollout_delta_steering_abs_max == 0
rollout_delta_log_scale_abs_max == 0
rollout_scale_matches_base_exactly == true
```

P2.1-G 保留为不含物理支持门的消融。两个模式的 runtime contract 和输出目录不同，
P2.1-U 必须从 P1/Base 创建新 run，不能恢复 P2.1-G checkpoint。

## 3. 代码级验证

```bash
conda activate sigmarl-nod
PYTHONNOUSERSITE=1 python -m unittest discover -s tests/psb_marl -v
```

测试额外证明：即使 \(z,q\) 仍非零，只要全部 \(\rho=0\)，动作修正就逐元素严格为零；
在非零冲突下，速度动作梯度仍能穿过支持门、近端根和反对称控制器。

## 4. 下一次手动训练

```bash
python main_training.py \
  --config configs/psb_marl/p2_1_u_urgency_supported_sector.json \
  --iterations 30
```

不要附加 `--resume`。训练结束后先分析 30 轮结构和趋势，再决定是否执行正式 5-seed
测试。

## 5. P2.1-U 验证结果

完成的 run：

```text
outputs/psb_marl/p2_1_u_urgency_supported_sector/runs/
psb-p2-seed0-20260831T164749259850Z-8ffdd6a2
```

最初固定的 5-seed 评估中，奖励门和全部结构门通过；总碰撞上置信界为
`+0.002001988`，仅比 `+0.002` 门限高约 `1.99e-6`，因此该报告仍严格记为失败。随后
预先固定完整 seeds `101–110`，执行不提前停止的 10-seed 确认性评估，并使用独立报告
标签保留原结果：

```bash
python main_testing.py \
  --config configs/psb_marl/p2_1_u_urgency_supported_sector.json \
  --run-dir <P2_1_U_RUN_DIR> \
  --checkpoint <P2_1_U_RUN_DIR>/candidate_policy.pth \
  --scenario intersection_2 \
  --max-steps 600 \
  --episodes 4 \
  --seeds 101 102 103 104 105 106 107 108 109 110 \
  --no-render \
  --compare-base \
  --psb-report-label confirmatory10
```

确认性结果：

| Candidate - Base | 均值差 | 置信边界 | 结果 |
|---|---:|---:|---|
| 奖励 | `+0.001132` | 下界 `+0.000056` | 通过 |
| 车辆碰撞 | `-0.000445` | — | 改善 |
| 车道碰撞 | `+0.000668` | — | 仍有恶化 |
| 总碰撞 | `+0.000223` | 上界 `+0.001214` | 通过 |

所有有限性、反对称、近端残差、控制有界、尺度冻结、steering 投影和扇区界检查均
通过。该版本是当前第一个同时获得正的奖励置信下界并通过总碰撞非劣门的 PSB 候选。
但车道碰撞分项仍弱于 Base，后续论文表述必须区分“总体安全非劣”与“所有安全分项均
改善”。当前评估没有自动 promotion，部署仍保持 Base fallback，需单独作出部署决定。

独立报告为：

```text
p2_manual_validation_confirmatory10.json
comparison_to_base_confirmatory10.json
```

由于 seeds `101–105` 在第一次 5-seed 评估中已经被查看，10-seed 结果属于扩展证据，
不是完全独立的 holdout。进入 P3 前应执行锁定的多训练种子鲁棒性协议：
[`P2_2_ROBUSTNESS_VALIDATION.md`](P2_2_ROBUSTNESS_VALIDATION.md)。
