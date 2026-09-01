# PSB-MARL P2.2-R：P3 前鲁棒性验证

P2.2-R 不修改 P2.1-U 的网络、近端分岔动力学或动作约束。它冻结当前
`supported_sector_q_gate + longitudinal_only + mean_only` 结构，只回答一个问题：
P2.1-U 的非劣结果能否跨训练随机种子、未参与选择的测试种子和不同场景复现。

此前 `intersection_2` 的 seeds `101–110` 结果可作为扩展证据，但其中 `101–105`
已经用于查看第一次 5-seed 结果，因此不能再被称为完全独立的 holdout。P2.2-R 在运行
前锁定所有配置和种子，验证过程中不得根据中间结果调参、换 checkpoint 或删减种子。

## 1. 固定实验合同

协议文件：

```text
configs/psb_marl/p2_2_r_holdout_protocol.json
```

训练配置：

```text
configs/psb_marl/p2_2_r_seed0.json
configs/psb_marl/p2_2_r_seed1.json
configs/psb_marl/p2_2_r_seed2.json
```

三个实验共享同一份冻结 Base policy/critic、P1 近端层、P2.1-U 网络结构和 30 轮预算，
仅 `training_seed` 分别取 `0/1/2`。这里的 `training_seed` 控制环境采样、网络初始化与
优化随机性；它不改变 Base checkpoint 的来源。

P2.2-R 使用三个同时成立的性能门：

\[
\operatorname{LCB}_{0.90}(\Delta R)\ge -0.002,
\]

\[
\operatorname{UCB}_{0.90}(\Delta C_{\mathrm{total}})\le 0.002,
\qquad
\operatorname{UCB}_{0.90}(\Delta C_{\mathrm{lane}})\le 0.001.
\]

其中差值均为 Candidate 减 Base。有限性、反对称性、近端残差、控制有界、尺度冻结、
steering 冻结和扇区界仍须全部通过。车道碰撞门是 P2.2-R 新增的独立门，避免总碰撞
下降掩盖车道碰撞恶化。

## 2. 手动训练（三个独立 run）

从项目根目录依次执行：

```bash
conda activate sigmarl-nod
PYTHONNOUSERSITE=1 python main_training.py \
  --config configs/psb_marl/p2_2_r_seed0.json

PYTHONNOUSERSITE=1 python main_training.py \
  --config configs/psb_marl/p2_2_r_seed1.json

PYTHONNOUSERSITE=1 python main_training.py \
  --config configs/psb_marl/p2_2_r_seed2.json
```

不要添加 `--resume` 或 `--iterations`。分别记录控制台输出的三个绝对目录为
`<RUN0>`、`<RUN1>`、`<RUN2>`。共享输出根目录中的 `latest_run.json` 只指向最后完成的
一次训练，不能代替这三个显式目录。

## 3. 锁定 holdout 测试

每个 run 必须使用与其训练种子匹配的配置，并完整执行下面两组测试。以 seed 0 为例：

```bash
PYTHONNOUSERSITE=1 python main_testing.py \
  --config configs/psb_marl/p2_2_r_seed0.json \
  --run-dir <RUN0> \
  --checkpoint <RUN0>/candidate_policy.pth \
  --scenario CPM_mixed \
  --max-steps 600 \
  --episodes 4 \
  --seeds 201 202 203 204 205 206 207 208 209 210 \
  --no-render \
  --compare-base \
  --psb-report-label holdout_cpm_mixed

PYTHONNOUSERSITE=1 python main_testing.py \
  --config configs/psb_marl/p2_2_r_seed0.json \
  --run-dir <RUN0> \
  --checkpoint <RUN0>/candidate_policy.pth \
  --scenario intersection_2 \
  --max-steps 600 \
  --episodes 4 \
  --seeds 301 302 303 304 305 306 307 308 309 310 \
  --no-render \
  --compare-base \
  --psb-report-label holdout_intersection_2
```

对 `<RUN1>` 和 `<RUN2>` 原样重复，只把配置分别换成 `seed1.json`、`seed2.json`。
不要添加 `--promote-if-noninferior`：P2.2-R 先收集证据，不改变 Base fallback。

每个 run 完成后应包含：

```text
p2_manual_validation_holdout_cpm_mixed.json
p2_manual_validation_holdout_intersection_2.json
```

## 4. 三训练种子汇总

六组测试全部完成后执行：

```bash
PYTHONNOUSERSITE=1 python -m utilities.psb_marl.p2_robustness \
  --protocol configs/psb_marl/p2_2_r_holdout_protocol.json \
  --run-dirs <RUN0> <RUN1> <RUN2> \
  --output outputs/psb_marl/p2_2_r_urgency_supported_sector/p2_2_r_summary.json
```

汇总器会拒绝错误的训练种子、训练轮数、runtime contract、场景、测试种子顺序、
episode 数、步数、报告标签，以及跨训练种子字节完全相同的 candidate policy，并报告
各场景在三个训练种子上的均值与样本标准差。

只有 `p2_2_r_summary.json` 中顶层 `"passed": true`，才能进入 P3。该条件表示三个训练
种子在两组锁定场景上都通过奖励、总碰撞、车道碰撞和全部结构门。若失败，应保留全部
报告并先判断是训练方差、迁移失败还是车道碰撞门失败；不能继续使用这批 holdout 种子
调参后再把它们称为 holdout。

## 5. 代码级验证

P2.2-R 的实现不需要环境训练即可检查：

```bash
PYTHONNOUSERSITE=1 python -m unittest discover -s tests/psb_marl -v
```

单元测试覆盖训练种子隔离、车道碰撞独立门、锁定协议校验和三训练种子汇总。

## 6. 正式验证结果

最终使用的三个独立训练 run：

```text
seed0: psb-p2-seed0-20260831T201718440272Z-1bc40e3f
seed1: psb-p2-seed1-20260901T044027570590Z-717d80b4
seed2: psb-p2-seed2-20260901T045843259542Z-754c5357
```

三个 `candidate_policy.pth` 的 SHA-256 各不相同，训练曲线也不相同。每个 run 均完成
30 轮训练，并在两个锁定场景的全部 10 个测试种子上完成 4-episode paired Base 评估。

六组独立门结果均通过。跨训练种子的均值和样本标准差如下，差值均为 Candidate 减
Base：

| 场景 | 奖励差 | 车辆碰撞差 | 车道碰撞差 | 总碰撞差 |
|---|---:|---:|---:|---:|
| `CPM_mixed` | `+0.000207 ± 0.000114` | `-0.000017 ± 0.000124` | `+0.000014 ± 0.000024` | `-0.000003 ± 0.000101` |
| `intersection_2` | `-0.000280 ± 0.000497` | `+0.000169 ± 0.000215` | `-0.000278 ± 0.000379` | `-0.000095 ± 0.000175` |

各单组最接近奖励门限的是 seed1 / `intersection_2`，奖励下置信界为 `-0.001782`；
最接近车道碰撞门限的是 seed0 / `intersection_2`，上置信界为 `+0.000702`。二者仍分别
满足 `-0.002` 和 `+0.001` 的预注册门限。所有结构检查在六组评估中均通过。

正式汇总：

```text
outputs/psb_marl/p2_2_r_urgency_supported_sector/p2_2_r_summary.json
passed: true
```

因此 P2.1-U 已获得跨训练随机种子、训练场景和迁移场景的鲁棒非劣证据，可以进入 P3。
该结果不能表述为所有指标均显著改善：`CPM_mixed` 整体接近 Base，`intersection_2`
表现为平均总碰撞和车道碰撞下降、平均奖励轻微下降且车辆碰撞分项轻微上升。P3 应在
保留现有安全和结构合同的前提下寻求更明确的性能增益。
