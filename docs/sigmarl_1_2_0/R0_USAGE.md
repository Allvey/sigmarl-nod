# R0 标准训练与测试入口

> 状态：R0 实现已完成，训练结果由用户手动验证  
> Conda 环境：`sigmarl-nod`

## 1. 激活环境

```bash
conda activate sigmarl-nod
```

R0 已在该环境中设置 `PYTHONNOUSERSITE=1`，并按根目录 `requirements.txt` 在环境
内部补齐依赖。不使用项目内 `.venv`。

## 2. 完整训练

先在根目录 `config.json` 中设置训练参数和输出目录：

```json
{
    "where_to_save": "outputs/testing_random/"
}
```

然后运行：

```bash
python main_training.py
```

R0 使用 SigmaRL 1.2.0 原始 `mappo_cavs()` 训练流程，不改变 Actor、Critic、环境、
reward 或 PPO。

## 3. 测试已训练模型

训练完成后运行：

```bash
python main_testing.py
```

R0 已取消 `main_testing.py` 中旧的硬编码模型目录。测试入口会读取 `config.json` 的
`where_to_save`，再从训练保存的 JSON 恢复参数和模型。

测试默认沿用训练 checkpoint 中的 `scenario_type`，避免硬编码切换到 agent 数不同
的场景后造成中央 Critic 维度不匹配。跨场景测试将在后续 M10 使用显式评估配置实现。

## 4. 当前边界

- `python main_training.py`：训练原始 Base-MAPPO；
- `python main_testing.py`：测试该 Base 模型；
- R0 不包含 Opinion Dynamics；
- R0 不声明性能提升；
- 用户负责实际训练、测试和性能判断。
