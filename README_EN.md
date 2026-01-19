# MobiBench

A comprehensive benchmark for evaluating AI Agents' capabilities in mobile app automation testing.

## 📋 Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Supported Agents](#supported-agents)
- [Evaluation](#evaluation)
- [Examples](#examples)

## 🎯 Overview

MobiBench provides a standardized framework to evaluate AI Agents' (including LLM-based agents) automation capabilities on mobile apps. It supports popular apps across shopping, social, travel, and entertainment categories, using Finite State Machines (FSM) for objective task completion assessment.

### Supported Apps

- **Shopping**: Taobao, JD.com, Meituan, Ele.me
- **Social**: Weibo, Xiaohongshu, Zhihu, QQ
- **Travel**: Gaode, Ctrip, Tongcheng
- **Entertainment**: Bilibili, NetEase Cloud Music

Each app supports 21 task types covering common operations like search, browse, order, comment, and share.

## ✨ Key Features

- 🤖 **Multi-Agent Support**: Various LLM-based agent implementations
- 📱 **Real Scenarios**: Based on actual mobile app operation trajectories
- 🔍 **FSM Evaluation**: Objective assessment via Finite State Machines
- 📊 **Complete Pipeline**: Data collection, annotation, and evaluation tools
- 🎨 **Visualization**: Screenshot annotation and trajectory visualization
- 📈 **Performance Metrics**: Steps, success rate, response time analysis

## 📁 Project Structure

```
MobiBench/
├── agents/          # AI Agent implementations
├── env/             # Environment & FSM logic
├── collect/         # Data collection tools
├── data/            # Test data & configurations
├── utils/           # Utilities & models
├── results/         # Evaluation outputs
└── requirements.txt # Dependencies
```

## 🔧 Setup

### Requirements

- Python 3.10.18
- Android device/emulator (for actual testing)

### Installation

```bash
# Create environment
conda create -n MobiBench python=3.10.18
conda activate MobiBench

# Install dependencies
pip install -r requirements.txt

# Download models
cd MobiBench
modelscope download --model AI-ModelScope/OmniParser-v2.0 --local_dir ./utils/models/weights/OmniParser-v2.0
modelscope download --model sentence-transformers/paraphrase-MiniLM-L6-v2 --local_dir ./utils/models/weights/paraphrase-MiniLM-L6-v2
```

### Key Dependencies

- **DL Frameworks**: torch, torchvision
- **Vision/OCR**: paddlepaddle, paddleocr, ultralytics
- **Image**: Pillow, opencv-python
- **Mobile Automation**: uiautomator2
- **API**: fastapi, langchain

## 🚀 Quick Start

### Run Agent Evaluation

```bash
# MobiMind Agent
python -m MobiBench.agents.MobiMind.bench \
    --service_ip 123.60.91.241 \
    --decider_port 9001 \
    --datapath /path/to/MobiBench/data \
    --task_json /path/to/MobiBench/data/base.json \
    --result_dir /path/to/MobiBench/results/dev \
    --log_dir /path/to/MobiBench/agents/MobiMind/data \
    --use_flag e2e_v1

# UI-TARS Agent
python -m MobiBench.agents.UI_TARS.bench \
    --service_ip 192.168.12.165 \
    --port 8000 \
    --datapath /path/to/MobiBench/data \
    --task_json /path/to/MobiBench/data/base.json \
    --result_dir /path/to/MobiBench/results/dev \
    --log_dir /path/to/MobiBench/agents/UI_TARS/data
```

### Parameter Explanations

- `--service_ip`: IP address of the agent service
- `--port` / `--decider_port`: Service port number
- `--datapath` : Path to  data directory
- `--task_json`: JSON file containing task definitions
- `--result_dir`: Directory to save evaluation results
- `--log_dir`: Directory to save execution logs
- `--use_flag`: Special flag for agent configuration (e.g., `e2e_v1`)

## 🤖 Supported Agents

1. **AutoGLM**: Based on AutoGLM model
2. **Claude**: Based on Anthropic Claude
3. **Gemini**: Based on Google Gemini (2.5-flash, 2.5-pro, 3-flash-preview, 3-pro)
4. **GPT-5**: Based on OpenAI GPT-5
5. **Grok-4**: Based on Grok-4 model
6. **MobileAgent v2/v3**: Multi-agent collaborative assistant
7. **MobiMind**: Hierarchical decision-making agent
8. **UI-TARS**: Based on UI-TARS framework

## 📊 Evaluation

### FSM Assessment

Tasks are evaluated using Finite State Machines that track the agent's progress through predefined states.

### Metrics

- **Success Rate**: Percentage of completed tasks
- **Average Steps**: Mean operations per task
- **State Match Rate**: Alignment with reference trajectories
- **Response Time**: Agent reasoning + execution time

## 💡 Examples

### Evaluate Single Agent

```python
from MobiBench.agents.autoglm.bench import bench
from MobiBench.env.fsm import build_AppFSM

fsm = build_AppFSM(app="Taobao", task="type1", data_path="./data")
results = bench(fsm, app="Taobao", task="type1", instruction="Search iPhone on Taobao")
print(results)
```

### Batch Evaluation

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

## 📈 Results

Outputs in `results/` directory include:

- **CSV files**: Detailed evaluation tables
- **Trajectories**: Agent execution paths
- **Screenshots**: Step-by-step screenshots
- **Logs**: Detailed execution logs

Use provided tools for analysis and visualization.