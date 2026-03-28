# MobiGraph

A comprehensive benchmark platform for evaluating AI Agents in mobile application automation testing.

[中文版本](README_CN.md) | English Version

## 🎯 Overview

MobiGraph provides a standardized testing framework for evaluating various AI Agents (including LLM-based agents) on mobile application automation capabilities. It supports popular applications across multiple categories such as shopping, social media, travel, and entertainment, employing a Finite State Machine (FSM) for objective task completion assessment.

## 📁 Project Structure

```
MobiBench/
├── agents/          # AI Agent implementations
├── env/             # Environment and FSM logic
├── collect/         # Data collection tools
├── data/            # Test data and configurations
├── utils/           # Utility functions and models
├── results/         # Evaluation results
├── log_eval_auto.py # Automated evaluation script
└── requirements.txt # Dependencies
```

## 📊 Data Storage Format

Evaluation data is stored in the `data/rawdata/` directory, organized in the following hierarchical structure:

```
rawdata/
├── <app_name>/
│   ├── <task_type>/
│   │   ├── 1/
│   │   │   ├── 1.jpg          # Screenshot before the 1st action
│   │   │   ├── 2.jpg          # Screenshot before the 2nd action
│   │   │   ├── ...
│   │   │   └── actions.json   # Action records and task information
│   │   ├── 2/
│   │   │   └── ...            # 2nd data sample
│   │   ├── task.json          # Task description
│   │   └── ...
│   └── <other_task_type>/
└── <other_app_name>/
```


Each data sample contains:

- **Screenshot Sequence**: Records of the interface state before each action step
- **actions.json**: Contains the complete action sequence, task description, and app information
- **task.json**: Contains task description and metadata

## 🔧 Environment Setup

### System Requirements

- Python 3.10.18
- Android device or emulator (for actual testing)

### Installation Steps

```bash
# Create virtual environment
conda create -n MobiBench python=3.10.18
conda activate MobiBench

# Install dependencies
pip install -r requirements.txt

# Download models
cd MobiBench
modelscope download --model AI-ModelScope/OmniParser-v2.0 --local_dir ./utils/models/weights/OmniParser-v2.0
modelscope download --model sentence-transformers/paraphrase-MiniLM-L6-v2 --local_dir ./utils/models/weights/paraphrase-MiniLM-L6-v2
```

## 🚀 Quick Start

### Running Agent Evaluation

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

### Parameter Description

- `--service_ip <str>`: IP address of the Agent service, default `123.60.91.241`.
- `--port <int>` / `--decider_port <int>`: Service port number, default `9003`.
- `--datapath <path>`: Path to the evaluation data directory, default `MobiBench/data`.
- `--task_json <path>`: Task definition JSON file (task JSON file under `data`).
- `--result_dir <path>`: Directory to save evaluation results, default `MobiBench/results`.
- `--log_dir <path>`: Directory to save execution logs, default `agents/MobiMind/log`.
- `--e2e <on|off>`: Whether to use end-to-end inference mode to reduce grounder calls (default: `on`).
- `--e2e_flag`: End-to-end inference mode selection (default: `e2e_v1`). v1: does not consider image positions; v2: considers image context and system prompts.

## 📊 Evaluation Mechanism

### FSM Evaluation

Uses a Finite State Machine to track the Agent's progress through predefined states for task completion assessment.

### Evaluation Metrics

- **Success Rate**: Proportion of successfully completed tasks
- **Average Steps**: Average number of operations per task
- **State Matching Rate**: Degree of matching with reference trajectories
- **Response Time**: Agent inference and execution time

## 💡 Usage Examples

### Evaluating a Single Agent

```python
from MobiBench.agents.autoglm.bench import bench
from MobiBench.env.fsm import build_AppFSM

fsm = build_AppFSM(app="Taobao", task="type1", data_path="./data")
results = bench(fsm, app="Taobao", task="type1", instruction="Search for iPhone on Taobao")
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

---
