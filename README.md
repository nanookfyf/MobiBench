# MobiBench

一个用于评估 AI Agent 在移动应用自动化测试能力的综合性基准测试平台。

[English Version](README_EN.md) | 中文版本


## 🎯 概述

MobiBench 提供了一个标准化的测试框架，用于评估各种 AI Agent（包括基于大语言模型的智能体）在移动应用上的自动化操作能力。支持购物、社交、出行、娱乐等多个类别的流行应用，采用有限状态机（FSM）进行客观的任务完成度评估。


## 📁 项目结构

```
MobiBench/
├── agents/          # AI Agent实现
├── env/             # 环境与FSM逻辑
├── collect/         # 数据收集工具
├── data/            # 测试数据与配置
├── utils/           # 工具函数与模型
├── results/         # 评估结果
└── requirements.txt # 依赖包
```

## 📊 数据存储格式

评测数据存储在 `data/rawdata/` 目录中，按以下层级结构组织：

```
rawdata/
├── <应用名称>/
│   ├── <任务类型>/
│   │   ├── 1/
│   │   │   ├── 1.jpg          # 第1个操作前的截图
│   │   │   ├── 2.jpg          # 第2个操作前的截图
│   │   │   ├── ...
│   │   │   └── actions.json   # 操作记录和任务信息
│   │   ├── 2/
│   │   │   └── ...            # 第2条数据
│   │   ├── task.json          # 任务描述
│   │   └── ...
│   └── <其他任务类型>/
└── <其他应用名称>/
```
[数据集](https://huggingface.co/datasets/IPADS-SAI/MobiFlow/tree/main,"点击查看MobiFlow数据集")
每个数据样本包含：

- **截图序列**：记录每个操作步骤前的界面状态
- **actions.json**：包含完整的操作序列、任务描述和应用信息
- **task.json**：包含任务描述和元数据


## 🔧 环境配置

### 系统要求

- Python 3.10.18
- Android设备或模拟器（用于实际测试）

### 安装步骤

```bash
# 创建虚拟环境
conda create -n MobiBench python=3.10.18
conda activate MobiBench

# 安装依赖
pip install -r requirements.txt

# 下载模型
cd MobiBench
modelscope download --model AI-ModelScope/OmniParser-v2.0 --local_dir ./utils/models/weights/OmniParser-v2.0
modelscope download --model sentence-transformers/paraphrase-MiniLM-L6-v2 --local_dir ./utils/models/weights/paraphrase-MiniLM-L6-v2
```


## 🚀 快速开始

### 运行Agent评估

```bash
# MobiMind Agent
python -m MobiBench.agents.MobiMind.bench \
    --service_ip 123.60.91.241 \
    --decider_port 9001 \
    --datapath /path/to/MobiBench/data \
    --task_json /path/to/MobiBench/data/base.json \
    --result_dir /path/to/MobiBench/results/dev \
    --log_dir /path/to/MobiBench/agents/MobiMind/data \
    --e2e 

# UI-TARS Agent
python -m MobiBench.agents.UI_TARS.bench \
    --service_ip 192.168.12.165 \
    --port 8000 \
    --datapath /path/to/MobiBench/data \
    --task_json /path/to/MobiBench/data/base.json \
    --result_dir /path/to/MobiBench/results/dev \
    --log_dir /path/to/MobiBench/agents/UI_TARS/data
```

### 参数说明

- `--service_ip <str>`: Agent服务的IP地址，默认`123.60.91.241`。
- `--port <int>` / `--decider_port <int>`: 服务端口号，默认`9003`。
- `--datapath <path>`: 评估数据目录路径，默认`MobiBench/data`。
- `--task_json <path>`: 任务定义JSON文件 (`data`下的任务json文件)。
- `--result_dir <path>`: 评估结果保存目录，默认`MobiBench/results`。
- `--log_dir <path>`: 执行日志保存目录，默认`agents/MobiMind/log`。
- `--e2e <on|off>`: 是否端到端推理模式，减少grounder调用（默认：`on`）。
- `--e2e_flag`: 端到端推理模式选择，（默认：`e2e_v1`） v1:不考虑图像位置，v2：考虑图像上下文本，以及系统提示。

## 📊 评估机制

### FSM评估

使用有限状态机跟踪Agent在预定义状态中的进展来评估任务完成情况。

### 评估指标

- **成功率**: 完成任务的比例
- **平均步数**: 每任务平均操作次数
- **状态匹配率**: 与参考轨迹的匹配程度
- **响应时间**: Agent推理与执行时间

## 💡 使用示例

### 评估单个Agent

```python
from MobiBench.agents.autoglm.bench import bench
from MobiBench.env.fsm import build_AppFSM

fsm = build_AppFSM(app="淘宝", task="type1", data_path="./data")
results = bench(fsm, app="淘宝", task="type1", instruction="在淘宝搜索iPhone手机")
print(results)
```

### 批量评估

```python
import json
from MobiBench.agents.autoglm.bench import bench
from MobiBench.env.fsm import build_AppFSM

with open('data/alldata.json', 'r', encoding='utf-8') as f:
    alldata = json.load(f)

for app, task_types in alldata.items():
    for task_type in task_types:
        fsm = build_AppFSM(app=app, task=task_type, data_path="./data")
        results = bench(fsm, app=app, task=task_type, instruction="...")
        save_results(results, app, task_type)
```


---
