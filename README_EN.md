# MobiBench

A comprehensive benchmark platform for evaluating AI Agents' capabilities in mobile application automation testing.

[中文版本](README.md) | English Version

## 🎯 Overview

MobiBench provides a standardized testing framework to evaluate the automation capabilities of various AI Agents (including LLM-based agents) on mobile applications. It supports popular applications across categories such as shopping, social networking, travel, and entertainment, utilizing Finite State Machines (FSM) for objective task completion assessment.

## 📁 Project Structure

```
MobiBench/
├── agents/          # AI Agent implementations
├── env/             # Environment and FSM logic
├── collect/         # Data collection tools
├── data/            # Test data and configurations
├── utils/           # Utility functions and models
├── results/         # Evaluation results
└── requirements.txt # Dependencies
```

## 📊 Data Storage Format

Evaluation data is stored in the `data/rawdata/` directory, organized in the following hierarchical structure:

```
rawdata/
├── <application_name>/
│   ├── <task_type>/
│   │   ├── 1/
│   │   │   ├── 1.jpg          # Screenshot before the 1st operation
│   │   │   ├── 2.jpg          # Screenshot before the 2nd operation
│   │   │   ├── ...
│   │   │   └── actions.json   # Operation records and task information
│   │   ├── 2/
│   │   │   └── ...            # 2nd data sample
│   │   ├── task.json          # Task description
│   │   └── ...
│   └── <other_task_types>/
└── <other_application_names>/
```

Each data sample contains:

- **Screenshot Sequence**: Records the interface state before each operation step
- **actions.json**: Contains complete operation sequences, task descriptions, and application information
- **task.json**: Contains task descriptions and metadata

## 🔧 Environment Configuration

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
    --e2e \
    --prompt_mode e2e_v1

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
- `--datapath`: Path to evaluation data directory
- `--task_json`: JSON file containing task definitions
- `--result_dir`: Directory to save evaluation results
- `--log_dir`: Directory to save execution logs
- `--e2e`: End-to-end inference mode, reduces grounder calls (default: `True`)
- `--prompt_mode`: Agent prompt mode selection (default: `e2e_v1`)
  - `e2e_v1`: Combined decision and execution mode, expands action space, adds fusion operations like `click_input` combining `click` and `input`
  - `decider_en`: Decision-execution separation mode (uses two-stage architecture: decision module analyzes current state and generates high-level instructions, then execution module converts them to specific operations)

## 📊 Evaluation Mechanism

### FSM Evaluation

Uses Finite State Machines to track the agent's progress through predefined states to evaluate task completion.

### Evaluation Metrics

- **Success Rate**: Proportion of completed tasks
- **Average Steps**: Mean number of operations per task
- **State Match Rate**: Degree of alignment with reference trajectories
- **Response Time**: Agent reasoning and execution time

## 💡 Usage Examples

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

---
