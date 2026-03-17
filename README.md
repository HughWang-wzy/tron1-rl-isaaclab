# TRON1-RL-IsaacLab

基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 和 [RSL-RL](https://github.com/leggedrobotics/rsl_rl) 的 LimX TRON-1A 双足机器人强化学习运动控制框架。支持轮足（WheelFoot）、点足（PointFoot）和板足（SoleFoot）三种足端构型，涵盖平地行走、崎岖地形、跳跃、纯步态行走等多种运动任务。

## 项目结构

```
tron1-rl-isaaclab/
├── exts/bipedal_locomotion/       # 主扩展包
│   └── bipedal_locomotion/
│       ├── assets/                # 机器人 USD 模型与配置
│       │   ├── config/            # WF/PF/SF 机器人配置
│       │   └── usd/              # URDF/USD 模型文件
│       └── tasks/locomotion/
│           ├── cfg/               # 基础环境配置（场景、观测、奖励、命令）
│           ├── mdp/               # MDP 组件
│           │   ├── rewards.py     # 奖励函数
│           │   ├── observations.py# 观测函数
│           │   ├── commands/      # 命令生成器（速度、步态、跳跃、身高）
│           │   ├── terminations.py# 终止条件
│           │   ├── events.py      # 域随机化事件
│           │   └── curriculums.py # 课程学习策略
│           ├── robots/            # 各机器人环境配置与 Gym 注册
│           └── agents/            # PPO 训练超参数配置
├── rsl_rl/                        # RSL-RL 训练框架
├── scripts/rsl_rl/               # 训练与回放脚本
│   ├── train.py
│   └── play.py
├── train.sh                       # 训练入口脚本
├── play.sh                        # 回放入口脚本
└── logs/                          # 实验日志
```

## 支持的机器人

| 构型 | 模型 | 说明 |
|------|------|------|
| WheelFoot (WF) | WF_TRON1A | 6 腿关节 + 2 轮关节，支持轮式滑行与纯步态行走 |
| PointFoot (PF) | PF_TRON1A | 点足构型 |
| SoleFoot (SF) | SF_TRON1A | 板足构型 |

## 注册环境

### WheelFoot
| 环境 ID | 说明 |
|---------|------|
| `Isaac-Limx-WF-Blind-Flat-v0` | 平地（无高度感知） |
| `Isaac-Limx-WF-Rough-v0` | 崎岖地形 |
| `Isaac-Limx-WF-Jump-Flat-v0` | 平地跳跃 |
| `Isaac-Limx-WF-Jump-Rough-v0` | 崎岖地形跳跃 |
| `Isaac-Limx-WF-Gait-Flat-v0` | 平地纯步态行走（禁用车轮） |
| `Isaac-Limx-WF-Gait-Rough-v0` | 崎岖地形纯步态行走 |
| `Isaac-Limx-WF-MoE-Flat-v0` | Mixture of Experts |

### PointFoot / SoleFoot
| 环境 ID | 说明 |
|---------|------|
| `Isaac-Limx-PF-Blind-Flat-v0` | 点足平地 |
| `Isaac-Limx-SF-Blind-Flat-v0` | 板足平地 |

> 所有环境均有对应的 `-Play-v0` 变体用于推理回放（32 环境、无随机化）。

## 快速开始

### 前置依赖

- NVIDIA Isaac Sim / Isaac Lab
- PyTorch
- RSL-RL（已包含在 `rsl_rl/` 目录中）

### 训练

```bash
# 使用 train.sh
bash train.sh

# 或直接调用训练脚本
python scripts/rsl_rl/train.py --headless --task=Isaac-Limx-WF-Gait-Flat-v0 --num_envs 128
```

常用参数：
- `--task`: 环境 ID（见上表）
- `--num_envs`: 并行环境数量
- `--headless`: 无头模式（无 GUI）
- `--resume`: 从最近的 checkpoint 恢复训练
- `--load_run`: 指定加载的实验目录
- `--checkpoint`: 指定 checkpoint 文件名

### 回放

```bash
# 使用 play.sh
bash play.sh

# 或直接调用
python scripts/rsl_rl/play.py --task=Isaac-Limx-WF-Gait-Flat-Play-v0 --num_envs 32
```

### 训练日志

实验日志保存在 `logs/rsl_rl/<experiment_name>/<timestamp>/` 目录下。

## 核心特性

### 奖励设计
- **速度跟踪奖励**：线速度/角速度指令跟踪
- **步态奖励**：基于力/速度的步态周期跟踪
- **跳跃奖励**：跳跃高度、腾空速度、收腿姿态
- **运动惩罚**：动作平滑度、关节力矩/加速度、功率消耗
- **条件奖励**：根据跳跃/行走状态自适应切换奖励

### 课程学习
- 跳跃概率课程：5% → 50%
- 辅助力课程：逐步衰减至零
- 地形难度递增

### 域随机化
- 随机推力扰动
- 质量参数变化
- 摩擦系数随机化

## 许可证

Apache 2.0
