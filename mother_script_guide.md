# 母脚本运行实验完整指南

## 概述

本文档详细说明如何使用母脚本（`run_all_experiments.py`）管理和运行大模型测评实验，包括单次运行、批量运行、拉丁方实验、并行执行、断点续传、错误处理等。

---

## 目录

0. [核心原则：独立上下文](#0-核心原则独立上下文)
1. [环境准备](#1-环境准备)
2. [配置检查](#2-配置检查)
3. [运行模式](#3-运行模式)
4. [参数说明](#4-参数说明)
5. [运行示例](#5-运行示例)
6. [断点续传](#6-断点续传)
7. [并行执行](#7-并行执行)
8. [结果管理](#8-结果管理)
9. [日志系统](#9-日志系统)
10. [错误处理](#10-错误处理)
11. [监控与调试](#11-监控与调试)
12. [常见问题](#12-常见问题)

---

## 0. 核心原则：独立上下文

### 0.1 原则说明

**每个模型进行每次判断时必须是独立的，动词和任务之间处于独立上下文，不可以有相互干扰。**

这是实验设计的核心原则，确保：
- 每次实验的结果不受之前实验的影响
- 模型对每个动词的判断是独立的
- 不同任务之间没有信息泄露
- 实验结果具有可重复性

### 0.2 实现方式

本框架通过以下方式确保独立上下文：

```python
# 每次API调用都是全新的会话
payload = {
    "model": model_path,
    "messages": [{"role": "user", "content": prompt}],  # 单条消息，无历史
    "temperature": 0.7,
    # ... 其他参数
}

# 不保存对话历史
response = client.chat.completions.create(...)
# 调用结束，上下文销毁
```

### 0.3 关键设计

| 设计要点 | 说明 |
|----------|------|
| 单次API调用 | 每次实验只调用一次API，不进行多轮对话 |
| 无历史记录 | 不向API传递任何历史对话记录 |
| 独立提示词 | 每个动词的提示词是独立生成的 |
| 独立会话 | vllm服务每次创建新的会话上下文 |

### 0.4 注意事项

**禁止行为**：
- ❌ 在提示词中包含之前的实验结果
- ❌ 让模型参考之前动词的编码结果
- ❌ 在同一会话中测试多个动词
- ❌ 保存并传递对话历史

**正确行为**：
- ✅ 每个动词单独生成提示词
- ✅ 每次调用使用独立的API请求
- ✅ 不在提示词中提及实验目的
- ✅ 保持提示词的一致性（仅替换动词）

### 0.5 代码示例

```python
# 正确：每个动词独立调用
for verb in verbs:
    prompt = template.replace("{verb}", verb)  # 独立生成提示词
    response = call_api(prompt)  # 独立API调用
    save_result(response)  # 保存结果
    # 上下文销毁，下次调用不会受到影响

# 错误：在同一会话中测试多个动词
conversation_history = []
for verb in verbs:
    prompt = f"基于之前的回答，现在判断：{verb}"
    conversation_history.append({"role": "user", "content": prompt})
    response = call_api(conversation_history)  # 包含历史，违反独立原则
```

### 0.6 验证方法

可以通过以下方式验证独立性：

```bash
# 1. 打乱动词顺序运行
python run_experiment.py --task task1 --verbs 扔 投 摔 丢 甩 抛
python run_experiment.py --task task1 --verbs 抛 甩 丢 摔 投 扔

# 2. 比较两次运行的结果
# 如果独立性良好，相同动词的结果应该一致（考虑随机种子）
```

---

## 1. 环境准备

### 1.1 依赖安装

```bash
pip install requests tqdm
```

### 1.2 vllm服务启动

```bash
# 基本启动
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --host localhost \
    --port 8000

# 完整配置
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --max-model-len 4096
```

### 1.3 验证vllm服务

```bash
curl http://localhost:8000/v1/models
```

---

## 2. 配置检查

### 2.1 模型配置（config.py）

```python
MODELS = [
    "/path/to/model1",
    "/path/to/model2",
]

VLLM_CONFIG = {
    "host": "localhost",
    "port": 8000,
}
```

### 2.2 动词配置

```python
# 中文动词
VERBS = ["投", "扔", "摔", "丢", "甩", "抛"]

# 拉丁方顺序
LATIN_SQUARE_ORDERS = [
    ["投", "扔", "摔", "丢", "甩", "抛"],
    ["扔", "摔", "丢", "甩", "抛", "投"],
    # ... 共6种顺序
]
```

### 2.3 任务配置

```python
TASKS = {
    "task1": {
        "name": "动作编码测评",
        "description": "对不同动词对应的动作进行五维度编码（JSON输出）",
        "prompt_template": "prompt_template.txt",
        "output_format": "json",
    },
    "task2": {
        "name": "动作描述测评",
        "description": "对不同动词对应的动作进行一句话描述",
        "prompt_template": "prompt_template_task2.txt",
        "output_format": "text",
    },
}
```

### 2.4 实验配置

```python
REPEAT_COUNT = 1          # 每个组合重复次数
SAVE_RAW_RESPONSE = True  # 是否保存原始响应
TIMEOUT = 120             # 超时时间（秒）
USE_LATIN_SQUARE = True   # 是否使用拉丁方顺序
```

---

## 3. 运行模式

### 3.1 单次模式（single）

运行单个任务、单个语言版本：

```bash
python run_all_experiments.py --mode single --task task1 --language zh
```

**适用场景**：
- 测试单个任务
- 运行特定语言版本
- 调试配置

### 3.2 批量模式（all）

运行所有任务、所有语言版本：

```bash
python run_all_experiments.py --mode all --languages zh en
```

**适用场景**：
- 完整实验
- 多语言对比
- 批量数据收集

### 3.3 拉丁方模式（latin-square）

运行拉丁方实验，自动遍历所有参与者ID：

```bash
python run_all_experiments.py --mode latin-square --task task1 --language zh
```

**适用场景**：
- 平衡顺序效应
- 系统性实验设计
- 需要6个参与者的数据

---

## 4. 参数说明

### 4.1 基本参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--mode` | 运行模式 | single | single/all/latin-square |
| `--task` | 任务ID | task1 | task1/task2 |
| `--language` | 语言版本 | zh | zh/en |
| `--languages` | 语言版本列表 | - | zh en |

### 4.2 模型参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--models` | 模型路径列表 | config.py | /path/to/model1 /path/to/model2 |
| `--host` | vllm服务地址 | localhost | 192.168.1.100 |
| `--port` | vllm服务端口 | 8000 | 8001 |

### 4.3 实验参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--repeat` | 重复次数 | 1 | 3 |
| `--participant-id` | 参与者ID | None（随机） | 1 |
| `--max-retries` | 最大重试次数 | 3 | 5 |
| `--retry-delay` | 重试延迟（秒） | 5.0 | 10.0 |

### 4.4 输出参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--output-dir` | 输出基础目录 | experiment_results | results/my_experiment |

### 4.5 控制参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--no-chat` | 使用Completion API | False |
| `--no-resume` | 禁用断点续传 | False |

---

## 5. 运行示例

### 5.1 基本运行

```bash
# 运行任务1，中文版
python run_all_experiments.py --mode single --task task1 --language zh

# 运行任务2，英文版
python run_all_experiments.py --mode single --task task2 --language en

# 运行所有任务和语言
python run_all_experiments.py --mode all --languages zh en
```

### 5.2 指定模型

```bash
# 运行单个模型
python run_all_experiments.py --mode single --task task1 --models /path/to/model1

# 运行多个模型
python run_all_experiments.py --mode single --task task1 \
    --models /path/to/model1 /path/to/model2 /path/to/model3
```

### 5.3 指定输出目录

```bash
python run_all_experiments.py --mode single --task task1 \
    --output-dir results/experiment_20240115
```

### 5.4 指定重复次数

```bash
python run_all_experiments.py --mode single --task task1 --repeat 3
```

### 5.5 指定参与者ID

```bash
python run_all_experiments.py --mode single --task task1 --participant-id 2
```

### 5.6 拉丁方实验

```bash
# 自动运行所有6个参与者ID
python run_all_experiments.py --mode latin-square --task task1 --language zh
```

### 5.7 完整实验示例

```bash
# 完整实验：所有任务、所有语言、3次重复
python run_all_experiments.py --mode all \
    --languages zh en \
    --repeat 3 \
    --output-dir results/full_experiment_20240115 \
    --max-retries 5 \
    --retry-delay 10
```

---

## 6. 断点续传

### 6.1 工作原理

- 每个输出目录下有`experiment_tracker.json`文件
- 记录已完成的实验（model_verb_repeat）
- 重新运行时自动跳过已完成的实验

### 6.2 使用方法

```bash
# 正常运行（默认启用断点续传）
python run_all_experiments.py --mode single --task task1

# 如果中断，重新运行相同的命令即可
python run_all_experiments.py --mode single --task task1
# 会自动跳过已完成的实验
```

### 6.3 禁用断点续传

```bash
python run_all_experiments.py --mode single --task task1 --no-resume
```

### 6.4 重新运行特定实验

```bash
# 删除tracker文件，重新运行所有实验
rm results/experiment_tracker.json
python run_all_experiments.py --mode single --task task1
```

### 6.5 Tracker文件格式

```json
{
  "completed": [
    "Llama-2-7b-chat-hf_投_0",
    "Llama-2-7b-chat-hf_扔_0",
    "Llama-2-7b-chat-hf_摔_0"
  ],
  "last_updated": "2024-01-15T10:30:00"
}
```

---

## 7. 并行执行

### 7.1 基本原理

通过启动多个母脚本进程实现并行：

```
终端1: 模型A → 结果目录A
终端2: 模型B → 结果目录B
终端3: 模型C → 结果目录C
```

### 7.2 关键规则

**必须遵守**：
1. 每个进程使用不同的输出目录
2. 控制vllm服务的并发负载
3. 监控GPU内存使用

### 7.3 单vllm服务并行

```bash
# 终端1
python run_all_experiments.py --mode single --task task1 \
    --models /path/to/model1 \
    --output-dir results/model1

# 终端2
python run_all_experiments.py --mode single --task task1 \
    --models /path/to/model2 \
    --output-dir results/model2

# 终端3
python run_all_experiments.py --mode single --task task1 \
    --models /path/to/model3 \
    --output-dir results/model3
```

**注意事项**：
- 建议同时运行2-3个进程
- 增加超时时间：`--retry-delay 10`
- 增加重试次数：`--max-retries 5`

### 7.4 多vllm服务并行

```bash
# 启动多个vllm服务
python -m vllm.entrypoints.openai.api_server --model /path/to/model1 --port 8000 &
python -m vllm.entrypoints.openai.api_server --model /path/to/model2 --port 8001 &
python -m vllm.entrypoints.openai.api_server --model /path/to/model3 --port 8002 &

# 每个进程连接不同的服务
python run_all_experiments.py --mode single --task task1 \
    --models /path/to/model1 --host localhost --port 8000 \
    --output-dir results/model1

python run_all_experiments.py --mode single --task task1 \
    --models /path/to/model2 --host localhost --port 8001 \
    --output-dir results/model2

python run_all_experiments.py --mode single --task task1 \
    --models /path/to/model3 --host localhost --port 8002 \
    --output-dir results/model3
```

### 7.5 GPU内存管理

```bash
# 查看GPU内存
nvidia-smi

# 限制每个服务的GPU内存使用
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model1 \
    --port 8000 \
    --gpu-memory-utilization 0.3
```

### 7.6 并行执行脚本

```bash
#!/bin/bash
# parallel_run.sh

MODELS=(
    "/path/to/model1"
    "/path/to/model2"
    "/path/to/model3"
)

TASK="task1"
LANGUAGE="zh"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for model in "${MODELS[@]}"; do
    model_name=$(basename "$model")
    output_dir="results/${model_name}_${TASK}_${LANGUAGE}_${TIMESTAMP}"
    
    echo "启动: $model_name -> $output_dir"
    python run_all_experiments.py \
        --mode single \
        --task $TASK \
        --language $LANGUAGE \
        --models "$model" \
        --output-dir "$output_dir" \
        --max-retries 5 \
        --retry-delay 10 &
done

echo "等待所有任务完成..."
wait
echo "所有任务完成！"
```

---

## 8. 结果管理

### 8.1 输出目录结构

```
experiment_results/
├── task1_zh_20240115_103000/
│   ├── Llama-2-7b-chat-hf_投_20240115_103000_123_abc123.json
│   ├── Llama-2-7b-chat-hf_扔_20240115_103001_456_def456.json
│   ├── ...
│   ├── experiment_tracker.json
│   └── summary_20240115_110000.json
├── task2_zh_20240115_103000/
│   └── ...
└── experiment_summary_20240115_103000.json
```

### 8.2 结果文件格式

#### 单次结果文件

```json
{
  "task_id": "task1",
  "model": "Llama-2-7b-chat-hf",
  "model_path": "/path/to/model",
  "verb": "投",
  "repeat_index": 0,
  "timestamp": "2024-01-15T10:30:00",
  "duration_seconds": 2.5,
  "is_valid": true,
  "parsed_result": {
    "FORCE": 4,
    "ARM": 1,
    "HAND": 8,
    "VD": 1,
    "HD": 1
  },
  "error_type": null,
  "raw_response": "..."
}
```

#### 汇总文件

```json
{
  "task_id": "task1",
  "task_name": "动作编码测评",
  "language": "zh",
  "experiment_time": {
    "start_time": "2024-01-15T10:30:00",
    "end_time": "2024-01-15T11:00:00",
    "duration_seconds": 1800
  },
  "total_experiments": 36,
  "valid_count": 30,
  "invalid_count": 6,
  "valid_rate": 83.3,
  "model_statistics": {
    "Llama-2-7b-chat-hf": {
      "total": 36,
      "valid": 30,
      "invalid": 6,
      "valid_rate": 83.3,
      "avg_duration": 2.5,
      "errors": {
        "E101": 3,
        "E103": 3
      }
    }
  },
  "error_statistics": {...},
  "results_by_model_verb": {...},
  "all_results": [...]
}
```

### 8.3 结果合并

当并行运行多个进程后，需要合并结果：

```python
# merge_results.py
import os
import json
from pathlib import Path
from datetime import datetime

def merge_results(input_dirs, output_dir):
    """合并多个目录的结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for input_dir in input_dirs:
        summary_files = list(Path(input_dir).glob("summary_*.json"))
        
        if summary_files:
            latest_summary = max(summary_files, key=lambda f: f.stat().st_mtime)
            with open(latest_summary, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                all_results.extend(summary.get("all_results", []))
    
    merged_summary = {
        "merged_at": datetime.now().isoformat(),
        "source_dirs": input_dirs,
        "total_results": len(all_results),
        "all_results": all_results
    }
    
    output_file = os.path.join(output_dir, f"merged_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_summary, f, ensure_ascii=False, indent=2)
    
    print(f"合并完成: {len(all_results)} 条结果 -> {output_file}")

if __name__ == "__main__":
    input_dirs = [
        "results/model1_task1_zh",
        "results/model2_task1_zh",
        "results/model3_task1_zh"
    ]
    merge_results(input_dirs, "results/merged")
```

### 8.4 数据筛查

```bash
# 筛查单个目录
python screen_data.py results/task1_zh_20240115 --task task1 --language zh

# 筛查合并后的数据
python screen_data.py results/merged --task task1 --language zh --output screening_report.json
```

---

## 9. 日志系统

### 9.1 日志文件位置

```
logs/
└── experiment_20240115_103000.log
```

### 9.2 日志内容

```
2024-01-15 10:30:00,123 - INFO - ============================================================
2024-01-15 10:30:00,123 - INFO - 任务: 动作编码测评
2024-01-15 10:30:00,123 - INFO - 描述: 对不同动词对应的动作进行五维度编码
2024-01-15 10:30:00,123 - INFO - 语言: zh
2024-01-15 10:30:00,123 - INFO - 模型数: 3
2024-01-15 10:30:00,123 - INFO - 动词数: 6
2024-01-15 10:30:00,123 - INFO - 总实验数: 18
2024-01-15 10:30:00,123 - INFO - ============================================================
2024-01-15 10:30:00,123 - INFO - 
[模型] Llama-2-7b-chat-hf
2024-01-15 10:30:00,123 - INFO - [路径] /path/to/model
2024-01-15 10:30:00,123 - INFO -   [动词] 投
2024-01-15 10:30:00,123 - INFO -     [1/18] 第1次重复...
2024-01-15 10:30:00,123 - INFO - 开始实验: 模型=Llama-2-7b-chat-hf, 动词=投, 重复=0
2024-01-15 10:30:02,456 - INFO - 实验完成: 有效, 耗时=2.33s
```

### 9.3 查看日志

```bash
# 查看最新日志
ls -lt logs/ | head -1

# 实时查看日志
tail -f logs/experiment_20240115_103000.log

# 搜索错误
grep "ERROR\|无效" logs/experiment_20240115_103000.log
```

---

## 10. 错误处理

### 10.1 错误类型

详见 `error_types_guide.md`：

| 错误码 | 类型 | 说明 |
|--------|------|------|
| E001-E004 | API错误 | 超时、连接、HTTP错误 |
| E101-E104 | 解析错误 | JSON、字段、值错误 |
| E201-E204 | 文件错误 | 读取、写入、权限错误 |
| E301-E303 | 配置错误 | 任务、模型、模板错误 |

### 10.2 自动重试

- API调用失败时自动重试
- 默认重试3次，间隔5秒
- 可通过参数调整：`--max-retries 5 --retry-delay 10`

### 10.3 错误统计

汇总报告中包含错误统计：

```json
{
  "error_statistics": {
    "E001": {
      "count": 5,
      "description": "API请求超时",
      "examples": [...]
    }
  }
}
```

### 10.4 处理失败实验

```bash
# 查看失败的实验
python -c "
import json
with open('results/task1_zh/summary_*.json') as f:
    data = json.load(f)
    for r in data['all_results']:
        if not r['is_valid']:
            print(f\"{r['model']} {r['verb']}: {r['error_type']}\")
"
```

---

## 11. 监控与调试

### 11.1 实时进度监控

```bash
# 统计已完成的实验数
ls results/task1_zh/*.json | grep -v "summary_" | grep -v "tracker" | wc -l

# 查看tracker文件
cat results/task1_zh/experiment_tracker.json | python -m json.tool
```

### 11.2 GPU监控

```bash
# 实时监控GPU
watch -n 5 nvidia-smi

# 查看GPU内存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 11.3 进程监控

```bash
# 查看运行中的实验进程
ps aux | grep "run_all_experiments\|run_experiment"

# 查看vllm服务
ps aux | grep "vllm"
```

### 11.4 网络监控

```bash
# 测试vllm服务连接
curl -s http://localhost:8000/v1/models | python -m json.tool

# 测试API调用
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"/path/to/model","messages":[{"role":"user","content":"test"}]}'
```

### 11.5 调试模式

```bash
# 单次实验调试
python run_experiment.py --task task1 --language zh --models /path/to/model --verbs 投 --repeat 1

# 查看详细日志
python run_all_experiments.py --mode single --task task1 2>&1 | tee debug.log
```

---

## 12. 常见问题

### Q1: 如何恢复中断的实验？

**A**: 重新运行相同的命令，断点续传会自动跳过已完成的实验。

```bash
# 原命令
python run_all_experiments.py --mode single --task task1 --language zh

# 中断后，重新运行相同命令即可
python run_all_experiments.py --mode single --task task1 --language zh
```

### Q2: 如何重新运行某个模型？

**A**: 删除该模型的tracker记录或整个目录。

```bash
# 方法1：删除tracker文件
rm results/task1_zh/experiment_tracker.json

# 方法2：删除整个目录
rm -rf results/task1_zh
```

### Q3: 如何查看实验进度？

**A**: 查看tracker文件或汇总文件。

```bash
# 查看tracker
cat results/task1_zh/experiment_tracker.json | python -c "import json,sys; print(len(json.load(sys.stdin)['completed']))"

# 查看汇总
ls results/task1_zh/summary_*.json
```

### Q4: 并行运行时某个任务失败了怎么办？

**A**: 
1. 查看对应目录的日志文件
2. 修复问题后重新运行该任务
3. 不需要重新运行其他任务

### Q5: 如何知道所有任务都完成了？

**A**:
1. 检查各目录的汇总文件
2. 查看日志中的"实验完成!"信息
3. 检查进程是否还在运行

### Q6: vllm服务响应很慢怎么办？

**A**:
1. 减少并发进程数
2. 增加超时时间
3. 检查GPU内存使用
4. 考虑使用多vllm服务

### Q7: 如何修改实验参数后继续运行？

**A**: 
- 如果修改了提示词模板或编码规则，需要删除tracker重新运行
- 如果只修改了重试参数等，可以直接继续运行

### Q8: 结果文件太大怎么办？

**A**:
1. 设置`SAVE_RAW_RESPONSE = False`不保存原始响应
2. 定期清理旧的结果文件
3. 使用数据筛查脚本提取有效数据

---

## 附录：快速参考

### 常用命令

```bash
# 基本运行
python run_all_experiments.py --mode single --task task1 --language zh

# 指定模型
python run_all_experiments.py --mode single --task task1 --models /path/to/model

# 拉丁方实验
python run_all_experiments.py --mode latin-square --task task1 --language zh

# 批量运行
python run_all_experiments.py --mode all --languages zh en

# 并行运行
python run_all_experiments.py --mode single --task task1 --models /path/to/model1 --output-dir results/model1 &
python run_all_experiments.py --mode single --task task1 --models /path/to/model2 --output-dir results/model2 &

# 数据筛查
python screen_data.py results/task1_zh --task task1 --language zh

# 合并结果
python merge_results.py
```

### 配置文件

- `config.py` - 中文配置
- `config_en.py` - 英文配置
- `prompt_template*.txt` - 提示词模板

### 输出文件

- `*.json` - 单次实验结果
- `summary_*.json` - 汇总文件
- `experiment_tracker.json` - 断点续传记录
- `logs/*.log` - 日志文件

### 指南文档

- `README.md` - 项目说明
- `data_screening_guide.md` - 数据筛查指南
- `error_types_guide.md` - 错误类型说明
- `parallel_execution_guide.md` - 并行执行指南
- `mother_script_guide.md` - 本文档
