# MobiBench

MobiBench 是一个全面的移动应用自动化测试基准测试平台，用于评估和比较不同 AI Agent 在移动应用上的自动化操作能力。

## 📋 目录

- [项目简介](#项目简介)
- [主要特性](#主要特性)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [支持的 Agent](#支持的-agent)
- [数据收集与标注](#数据收集与标注)
- [评估机制](#评估机制)
- [使用示例](#使用示例)
- [结果分析](#结果分析)


## 🎯 项目简介

MobiBench 提供了一个标准化的测试框架，用于评估各种 AI Agent（包括基于大语言模型的智能体）在移动应用上的自动化操作能力。项目支持多个主流移动应用，涵盖购物、社交、出行、娱乐等多个场景，通过有限状态机（FSM）机制对任务完成情况进行客观评估。

### 支持的应用

- **购物类**：淘宝、京东、美团、饿了么
- **社交类**：微博、小红书、知乎、QQ
- **出行类**：高德、携程、同城
- **娱乐类**：bilibili、网易云音乐

每个应用支持多种任务类型（type1-type21），覆盖搜索、浏览、下单、评论、分享等常见操作场景。

## ✨ 主要特性

- 🤖 **多 Agent 支持**：集成多种 AI Agent 实现，包括基于不同大语言模型的智能体
- 📱 **真实应用场景**：基于真实移动应用的操作轨迹数据
- 🔍 **FSM 评估机制**：使用有限状态机对任务完成情况进行客观评估
- 📊 **完整数据流程**：支持数据收集、标注、评估的完整流程
- 🎨 **可视化工具**：提供截图标注、轨迹可视化等工具
- 📈 **性能分析**：支持操作步数、成功率、响应时间等多维度性能分析

## 📁 项目结构

```
MobiBench/
├── agents/                    # 各种 AI Agent 实现
│   ├── autoglm/              # AutoGLM Agent
│   ├── claude/                # Claude Agent
│   ├── gemini/                # Gemini Agent
│   ├── gpt-5/                 # GPT-5 Agent
│   ├── grok4/                 # Grok-4 Agent
│   ├── MobileAgentv2/         # MobileAgent v2
│   ├── Mobileagentv3/         # MobileAgent v3
│   ├── MobiMind/              # MobiMind Agent
│   ├── UI_TARS/               # UI-TARS Agent
│   └── Base/                  # 基础 Agent 接口
├── env/                       # 环境相关代码
│   ├── fsm.py                 # 有限状态机实现
│   ├── parsedata2link.py      # 数据解析与轨迹链接
│   ├── multi_vis.py            # 多轨迹可视化
│   └── type_spaces.py          # 状态空间和动作空间定义
├── collect/                    # 数据收集与标注工具
│   ├── manual/                # 手动数据收集
│   ├── auto/                  # 自动数据收集
│   ├── annotate.py            # 数据标注
│   └── construct_sft.py      # SFT 数据构建
├── data/                       # 测试数据
│   ├── rawdata/               # 原始轨迹数据
│   ├── fsm_cache/             # FSM 缓存
│   └── *.json                 # 任务配置数据
├── utils/                      # 工具函数
│   ├── models/                # 模型相关工具
│   ├── draw_bounds.py         # 边界框绘制
│   ├── task_get.py            # 任务获取
│   └── score_proc.py          # 评分处理
├── results/                    # 评估结果
│   ├── dev/                   # 开发集结果
│   └── speed/                 # 速度测试结果
└── requirements.txt           # 依赖包列表
```

## 🔧 环境配置

### 系统要求

- Python 3.10.18
- 支持 Android 设备或模拟器（用于实际测试）
- 足够的存储空间（用于模型和数据）

### 安装步骤

1. **创建虚拟环境**

```bash
conda create -n MobiBench python=3.10.18
conda activate MobiBench
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **下载依赖模型权重**

```bash
cd MobiBench
# 下载 OmniParser-v2.0 模型
modelscope download --model AI-ModelScope/OmniParser-v2.0 --local_dir ./utils/models/weights/OmniParser-v2.0

# 下载 sentence-transformers 模型
modelscope download --model sentence-transformers/paraphrase-MiniLM-L6-v2 --local_dir ./utils/models/weights/paraphrase-MiniLM-L6-v2

# 下载 GroundingDINO 模型
modelscope download --model AI-ModelScope/GroundingDINO --local_dir ./utils/models/weights/GroundingDINO
```

### 主要依赖

- **深度学习框架**：torch, torchvision, torchaudio
- **OCR 与视觉**：paddlepaddle, paddleocr, ultralytics, transformers
- **图像处理**：Pillow, opencv-python
- **移动自动化**：uiautomator2
- **API 框架**：fastapi, uvicorn
- **其他**：langchain, pandas, sentence-transformers

## 🚀 快速开始

### 运行 Agent 评估

```bash
# 运行 AutoGLM Agent
python -m MobiBench.agents.autoglm.bench --datapath /home/feh/mobibench/MobiBench/data  --task_json /home/feh/mobibench/MobiBench/data/base.json --result_dir /home/feh/mobibench/MobiBench/results/dev --log_dir /home/feh/mobibench/MobiBench/agents/MobiMind/data


# 运行 MobiMind Agent
python -m MobiBench.agents.MobiMind.bench --datapath /home/feh/mobibench/MobiBench/data  --task_json /home/feh/mobibench/MobiBench/data/base.json --result_dir /home/feh/mobibench/MobiBench/results/dev --log_dir /home/feh/mobibench/MobiBench/agents/MobiMind/data

# 运行 MobileAgent v3
python -m MobiBench.agents.Mobileagentv3.bench --datapath /home/feh/mobibench/MobiBench/data  --task_json /home/feh/mobibench/MobiBench/data/base.json --result_dir /home/feh/mobibench/MobiBench/results/dev --log_dir /home/feh/mobibench/MobiBench/agents/MobiMind/data


# 运行 UI-TARS Agent
python -m MobiBench.agents.UI_TARS.bench --datapath /home/feh/mobibench/MobiBench/data  --task_json /home/feh/mobibench/MobiBench/data/base.json --result_dir /home/feh/mobibench/MobiBench/results/dev --log_dir /home/feh/mobibench/MobiBench/agents/MobiMind/data

```

### 配置 Agent

每个 Agent 都有自己的配置文件，通常需要设置：

- **API 配置**：模型 API 的基础 URL 和密钥
- **数据路径**：测试数据的位置
- **设备配置**：Android 设备连接信息
- **评估参数**：最大步数、超时时间等

## 🤖 支持的 Agent

### 1. AutoGLM
基于 AutoGLM 模型的智能体实现。

### 2. Claude
基于 Anthropic Claude 模型的智能体。

### 3. Gemini
基于 Google Gemini 模型的智能体，支持多个版本（2.5-flash, 2.5-pro, 3-flash-preview, 3-pro）。

### 4. GPT-5
基于 OpenAI GPT-5 模型的智能体。

### 5. Grok-4
基于 Grok-4 模型的智能体。

### 6. MobileAgent v2/v3
多 Agent 协作的移动设备操作助手。

### 7. MobiMind
基于分层决策架构的智能体。

### 8. UI-TARS
基于 UI-TARS 框架的智能体。



## 📝 数据收集与标注

### 手动数据收集

启动 Web 界面进行手动数据收集：

```bash
python -m collect.manual.server
```

访问 `http://localhost:9000` 进行数据收集操作。

### 自动数据收集

配置任务列表后，使用 AI Agent 自动收集数据：

```bash
python -m collect.auto.server --model <模型名称> --api_key <API密钥> --base_url <API基础URL> [--max_steps <最大步数>]
```


详细的数据收集和标注说明请参考 [collect/README.md](collect/README.md)。

## 📊 评估机制

### FSM 评估

MobiBench 使用有限状态机（FSM）机制对任务完成情况进行评估：

### 评估指标

- **成功率**：任务完成的比例
- **平均步数**：完成任务所需的平均操作步数
- **状态匹配率**：与标准轨迹的状态匹配程度
- **响应时间**：Agent 的推理和执行时间

## 💡 使用示例

### 示例 1：评估单个 Agent

```python
from MobiBench.agents.autoglm.bench import bench
from MobiBench.env.fsm import build_AppFSM

# 构建 FSM
fsm = build_AppFSM(app="淘宝", task="type1", data_path="./data")

# 运行评估
results = bench(fsm, app="淘宝", task="type1", instruction="在淘宝搜索iPhone手机")
print(results)
```

### 示例 2：批量评估

```python
import json
from MobiBench.agents.autoglm.bench import bench
from MobiBench.env.fsm import build_AppFSM

# 加载任务配置
with open('data/alldata.json', 'r', encoding='utf-8') as f:
    alldata = json.load(f)

# 遍历所有应用和任务类型
for app, task_types in alldata.items():
    for task_type in task_types:
        fsm = build_AppFSM(app=app, task=task_type, data_path="./data")
        results = bench(fsm, app=app, task=task_type, instruction="...")
        # 保存结果
        save_results(results, app, task_type)
```

## 📈 结果分析

评估结果保存在 `results/` 目录下，包括：

- **CSV 文件**：详细的评估结果表格
- **轨迹记录**：Agent 的执行轨迹
- **截图序列**：每个步骤的屏幕截图
- **日志文件**：详细的执行日志

可以使用提供的工具进行结果分析和可视化。

