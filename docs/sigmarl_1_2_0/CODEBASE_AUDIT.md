# SigmaRL 1.2.0 源码与运行环境核对记录

> 核对日期：2026-08-23  
> 目标目录：`/Users/zhangxiaotong/Code/sigmarl-nod`  
> 上游基线：SigmaRL tag `1.2.0`  
> 基线 commit：`5fe715bdfba4ff3e33d901d69dfa220f1222c060`

## 1. 源码核对结论

已将当前目录与基线 commit 的完整文件树逐文件比较，排除当前项目自有的
`.git/`、`docs/` 和 macOS `.DS_Store`。

结果：

- `assets/`、`scenarios/`、`utilities/` 内容与 1.2.0 完全一致；
- `config.json`、`requirements.txt`、`main_training.py`、`main_testing.py`、
  `README.md`、`LICENSE.txt` 等核心文件完全一致；
- 当前源码中没有 Opinion、TSC topology、action predictor 或旧测试代码残留。

当前缺少四项上游隐藏工程文件：

```text
.gitignore
.pre-commit-config.yaml
.python-version
.vscode/
```

它们不影响 SigmaRL 算法和仿真语义。其中上游 `.python-version` 内容是 pyenv
环境名 `vmas`，不是 Python 版本号；当前项目使用 Conda 环境
`sigmarl-nod`，因此不应盲目复制该文件。

## 2. 当前 Conda 环境

正确的项目解释器是：

```text
Conda environment: sigmarl-nod
Python:            3.9.25
torch:             2.1.0
torchrl:           0.2.1
tensordict:        0.2.1
vmas:              1.4.1
```

在未激活环境时，可显式运行：

```bash
conda run -n sigmarl-nod python <command>
```

不要使用当前 base Conda 的 `python`；它在导入 `torch` 时会因
`typing_extensions.TypeIs` 版本冲突失败。当前项目也没有 `.venv/`，因此文档
和后续脚本不得写死 `.venv/bin/python`。

## 3. 已通过的运行验证

使用 `sigmarl-nod` 环境在 `CPM_mixed` 上真实创建两个 VMAS 环境：

```text
reset observation       [2,4,32]
action spec             [2,4,2]
3-step random rollout   成功
rollout tensors         全部有限
```

`info` 的实测字段和 shape：

```text
pos                         [2,4,2]
rot                         [2,4,1]
vel                         [2,4,2]
act_vel                     [2,4,1]
act_steer                   [2,4,1]
ref                         [2,4,6]
distance_ref                [2,4,1]
distance_left_b             [2,4,1]
distance_right_b            [2,4,1]
is_collision_with_agents    [2,4,1]
is_collision_with_lanelets  [2,4,1]
```

## 4. 尚未通过的可复现性 Gate

当前 Conda 环境并非完全隔离：

- 部分包从 `~/.local/lib/python3.9/site-packages` 加载；
- 其中存在指向另一个项目目录的 editable `sigmarl 1.2.0`；
- `python -m pip check` 当前失败；
- 设置 `PYTHONNOUSERSITE=1` 后，环境内部缺少 `typing_extensions` 和多个
  间接依赖。

因此当前只能说“源码一致且 smoke 可运行”，不能宣称环境已完全可复现。
R0 状态应保持“进行中”，直到：

1. 原地修复现有 `sigmarl-nod` Conda 环境，不创建或切换到其他环境；
2. `sigmarl-nod` 环境内部独立安装 `requirements.txt` 的精确版本；
3. 不依赖 user-site 仍能导入核心包；
4. `pip check` 无冲突；
5. reset、3-step rollout 和 `TanhNormal` 有限值检查重新通过。

## 5. 当前结论

```text
源码基线：通过
环境 smoke：通过（但尚依赖 user-site）
隔离依赖：未通过
R0 总状态：进行中
下一动作：修复隔离环境后重跑 R0，再执行 R1 Base 基线
```
