# 实验策略补充文档（2026-06-22）

## 概述

本文档记录2026-06-22补充实验的策略说明，包括无效数据自动补采、tmux架构设计、补采自动化等内容。

---

## 1. 实验目标

- 8个模型 × 4个任务 × 6个动词 × 12次重复 = **2304条数据**
- 每个模型每个任务需要72条有效数据

---

## 2. 无效数据自动补采策略

### 2.1 核心原则

**如果数据无效（is_valid=False），必须自动补采，直到补采成功。**

### 2.2 补采机制

```
单次实验流程：
1. 运行实验 → 获取结果
2. 检查 is_valid
   - True → 标记完成，计入有效数据
   - False → 记录失败原因，立即重新尝试
3. 重复步骤1-2，直到 is_valid=True
```

### 2.3 补采记录

每个结果文件记录：
```json
{
  "retry_count": 3,           // 该动词重试次数
  "retry_history": [          // 重试历史
    {"attempt": 1, "is_valid": false, "error_type": "E104", "timestamp": "..."},
    {"attempt": 2, "is_valid": false, "error_type": "E104", "timestamp": "..."},
    {"attempt": 3, "is_valid": true, "error_type": null, "timestamp": "..."}
  ]
}
```

### 2.4 补采上限

- 单个动词最大重试次数：**无上限**（直到成功）
- 如果连续失败10次，记录警告日志，但继续尝试
- 如果连续失败50次，标记为异常，需要人工介入

---

## 3. tmux架构设计

### 3.1 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU 服务器                                │
├─────────────────────────────────────────────────────────────┤
│  tmux session: exp-model1                                   │
│  ├── 窗口1: vLLM服务 (port=8000, gpu-memory=0.35)          │
│  └── 窗口2: 数据采集脚本 (model1, 4任务×12次)               │
├─────────────────────────────────────────────────────────────┤
│  tmux session: exp-model2                                   │
│  ├── 窗口1: vLLM服务 (port=8001, gpu-memory=0.35)          │
│  └── 窗口2: 数据采集脚本 (model2, 4任务×12次)               │
├─────────────────────────────────────────────────────────────┤
│  ...                                                        │
├─────────────────────────────────────────────────────────────┤
│  tmux session: exp-monitor                                  │
│  └── 窗口1: 监控脚本 (nvidia-smi + 进度检查)               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 模型分配

| Session | 模型 | vLLM端口 | GPU内存 | 优先级 |
|---------|------|----------|---------|--------|
| exp-01 | HY-Embodied-0.5 | 8000 | 0.35 | high |
| exp-02 | Hunyuan-1.8B-Instruct | 8001 | 0.35 | high |
| exp-03 | Mimo-7B-SFT | 8002 | 0.35 | medium |
| exp-04 | Mimo-embodied-7B | 8003 | 0.35 | medium |
| exp-05 | Mimo-VL-7B-SFT-2508 | 8004 | 0.35 | medium |
| exp-06 | Qwen2.5-7B-Instruct | 8005 | 0.35 | medium |
| exp-07 | Qwen2.5-VL-7B-Instruct | 8006 | 0.35 | medium |
| exp-08 | RoboBrain2.0-7B | 8007 | 0.35 | medium |
| exp-monitor | 监控 | - | - | - |

### 3.3 单个Session结构

```bash
# 创建session
tmux new-session -d -s exp-01

# 窗口1: vLLM服务
tmux send-keys -t exp-01 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate NLP2" Enter
tmux send-keys -t exp-01 "python -m vllm.entrypoints.openai.api_server \
  --model /home/xitongzhang/models/HY-Embodied-0.5 \
  --host localhost \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 4096 \
  --trust-remote-code \
  2>&1 | tee logs/vllm_exp01.log" Enter

# 等待vLLM启动
sleep 30

# 窗口2: 数据采集脚本
tmux new-window -t exp-01
tmux send-keys -t exp-01 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate NLP2" Enter
tmux send-keys -t exp-01 "cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements" Enter
tmux send-keys -t exp-01 "python run_experiment.py \
  --task task1 --language zh \
  --models /home/xitongzhang/models/HY-Embodied-0.5 \
  --host localhost --port 8000 \
  --repeat 12 \
  --output-dir results/HY-Embodied-0.5_task1_zh \
  2>&1 | tee logs/exp01_task1_zh.log" Enter
```

---

## 4. 补采自动化设计

### 4.1 自动补采脚本

```python
#!/usr/bin/env python3
"""
auto_retry.py - 自动补采脚本
检查实验结果，对无效数据自动重新采集
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

class AutoRetry:
    def __init__(self, results_dir, model_path, host, port):
        self.results_dir = results_dir
        self.model_path = model_path
        self.host = host
        self.port = port
        self.model_name = os.path.basename(model_path)
        
    def find_invalid_results(self):
        """查找所有无效结果"""
        invalid = []
        for f in Path(self.results_dir).glob("*.json"):
            if "summary_" in f.name or "tracker" in f.name:
                continue
            with open(f) as fh:
                data = json.load(fh)
                if not data.get("is_valid", False):
                    invalid.append({
                        "file": str(f),
                        "verb": data["verb"],
                        "task": data["task_id"],
                        "error_type": data.get("error_type"),
                        "retry_count": data.get("retry_count", 0)
                    })
        return invalid
    
    def retry_experiment(self, verb, task_id, language, retry_count):
        """重试单个实验"""
        cmd = [
            "python", "run_experiment.py",
            "--task", task_id,
            "--language", language,
            "--models", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--verbs", verb,
            "--repeat", "1",
            "--output-dir", self.results_dir,
            "--no-resume"  # 强制重新运行
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def run(self):
        """执行自动补采"""
        invalid = self.find_invalid_results()
        
        if not invalid:
            print("没有发现无效数据，无需补采")
            return
        
        print(f"发现 {len(invalid)} 条无效数据，开始补采...")
        
        for item in invalid:
            verb = item["verb"]
            task_id = item["task"]
            language = "zh" if "_zh" in task_id else "en"
            retry_count = item["retry_count"] + 1
            
            print(f"补采: {self.model_name} / {verb} / {task_id} (第{retry_count}次)")
            
            success = self.retry_experiment(verb, task_id, language, retry_count)
            
            if success:
                print(f"  ✓ 补采成功")
            else:
                print(f"  ✗ 补采失败，将在下一轮重试")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    retry = AutoRetry(args.results_dir, args.model_path, args.host, args.port)
    retry.run()
```

### 4.2 自动补采流程

```
主循环：
1. 运行完整实验（12次重复）
2. 检查结果，统计无效数据
3. 如果有无效数据：
   a. 调用 auto_retry.py 补采
   b. 重新检查结果
   c. 重复步骤3，直到所有数据有效
4. 完成
```

### 4.3 集成到主脚本

```bash
#!/bin/bash
# run_with_retry.sh - 带自动补采的实验运行脚本

MODEL=$1
PORT=$2
MODEL_NAME=$(basename $MODEL)

echo "=== 开始实验: $MODEL_NAME ==="

# 启动vLLM
tmux new-session -d -s "vllm-$MODEL_NAME"
tmux send-keys -t "vllm-$MODEL_NAME" "python -m vllm.entrypoints.openai.api_server \
  --model $MODEL --port $PORT --gpu-memory-utilization 0.35 \
  --trust-remote-code 2>&1 | tee logs/vllm_${MODEL_NAME}.log" Enter

# 等待vLLM启动
sleep 30

# 运行4个任务
for task in task1 task2; do
  for lang in zh en; do
    echo "运行: $task $lang"
    python run_experiment.py \
      --task $task --language $lang \
      --models $MODEL --host localhost --port $PORT \
      --repeat 12 \
      --output-dir "results/${MODEL_NAME}_${task}_${lang}"
    
    # 自动补采
    python auto_retry.py \
      --results-dir "results/${MODEL_NAME}_${task}_${lang}" \
      --model-path $MODEL --host localhost --port $PORT
  done
done

echo "=== 实验完成: $MODEL_NAME ==="
```

---

## 5. 监控策略

### 5.1 进度监控脚本

```bash
#!/bin/bash
# monitor.sh - 实时监控所有实验进度

while true; do
  clear
  echo "=== 实验进度监控 ==="
  echo "时间: $(date)"
  echo ""
  
  for dir in results/*/; do
    if [ -d "$dir" ]; then
      total=$(ls "$dir"/*.json 2>/dev/null | grep -v "summary_" | grep -v "tracker" | wc -l)
      valid=$(python3 -c "
import json, glob
count = 0
for f in glob.glob('${dir}/*.json'):
    if 'summary_' in f or 'tracker' in f: continue
    with open(f) as fh:
        if json.load(fh).get('is_valid', False): count += 1
print(count)
")
      echo "$dir: $valid/$total 有效"
    fi
  done
  
  echo ""
  echo "GPU使用:"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
  
  sleep 10
done
```

### 5.2 告警机制

- 有效率低于80%时，发送告警
- GPU内存使用超过90%时，暂停新任务
- 连续失败超过10次时，记录异常日志

---

## 6. 数据合并

### 6.1 合并脚本

```python
#!/usr/bin/env python3
"""
merge_all_results.py - 合并所有实验结果
"""

import os
import json
from pathlib import Path
from datetime import datetime

def merge_all(base_dir="results", output_file="merged_results.json"):
    all_results = []
    
    for model_dir in Path(base_dir).iterdir():
        if not model_dir.is_dir():
            continue
        
        for result_file in model_dir.glob("*.json"):
            if "summary_" in result_file.name or "tracker" in result_file.name:
                continue
            
            with open(result_file) as f:
                data = json.load(f)
                if data.get("is_valid", False):
                    all_results.append(data)
    
    # 保存合并结果
    merged = {
        "merged_at": datetime.now().isoformat(),
        "total_valid": len(all_results),
        "results": all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"合并完成: {len(all_results)} 条有效数据 -> {output_file}")

if __name__ == "__main__":
    merge_all()
```

---

## 7. 执行计划

### 7.1 启动顺序

1. 先启动高优先级模型（HY-Embodied-0.5, Hunyuan-1.8B-Instruct）
2. 等待vLLM服务就绪
3. 启动中优先级模型（Mimo系列、Qwen系列、RoboBrain）
4. 启动监控session

### 7.2 预计时间

- 单个模型单个任务（72条）：约2-3小时
- 单个模型全部任务（288条）：约8-12小时
- 全部8个模型（2304条）：约12-24小时（并行执行）

### 7.3 检查点

- 每小时检查一次进度
- 每4小时检查一次数据质量
- 实验完成后运行数据合并和最终筛查

---

## 8. 应急预案

| 问题 | 处理方式 |
|------|----------|
| vLLM服务崩溃 | 重启服务，脚本自动断点续传 |
| GPU内存不足 | 降低gpu-memory-utilization，减少并发 |
| 磁盘空间不足 | 清理日志文件，或更换输出目录 |
| 网络中断 | 重启tmux session，断点续传 |
| 有效率过低 | 检查提示词，调整参数 |

---

## 9. 主动监控策略（2026-06-22补充）

### 9.1 核心原则

**必须主动检测并处理卡住的session，不能被动等待。**

### 9.2 检测规则

每隔10分钟检查一次，发现以下情况立即介入：

| 检测项 | 判断标准 | 处理方式 |
|--------|----------|----------|
| 进程卡死 | session无输出超过5分钟 | kill并重启 |
| GPU满载 | GPU内存>95% | 暂停新任务，等待释放 |
| 进度异常 | 单个实验耗时>300秒 | 检查是否OOM，考虑重启 |
| 连续失败 | 同一动词失败>10次 | 检查模型/提示词问题 |
| session空 | session只有shell提示符 | 进程已结束，检查日志 |

### 9.3 GPU分配原则

**每个GPU同一时间只运行1个模型**，避免内存竞争：

```
GPU 0: 模型A (task1_zh + task1_en + task2_zh + task2_en 并行)
GPU 1: 模型B (task1_zh + task1_en + task2_zh + task2_en 并行)
```

模型完成后释放GPU，再加载下一个模型。

### 9.4 监控命令

```bash
# 检查session活跃进程
for session in $(tmux list-sessions -F "#{session_name}"); do
    pid=$(tmux list-panes -t "$session" -F "#{pane_pid}" | head -1)
    child=$(pgrep -P $pid | head -1)
    if [ -n "$child" ]; then echo "$session: 运行中"; else echo "$session: 已结束"; fi
done

# 检查GPU使用
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

---

## 附录：快速启动命令

```bash
# 创建所有tmux session
for i in $(seq 1 8); do
  tmux new-session -d -s "exp-0$i"
done
tmux new-session -d -s "exp-monitor"

# 查看所有session
tmux list-sessions

# 进入某个session
tmux attach -t exp-01
```

---

## 10. 排除模型记录（2026-06-23）

### HY-Embodied-0.5

| 项目 | 内容 |
|------|------|
| **排除原因** | 模型依赖`flash_attn`库，当前环境nvcc版本(11.5)不满足编译要求(>=11.7) |
| **排除时间** | 2026-06-23 |
| **已有数据** | 保留在`results/HY-Embodied-0.5_*/`目录中，不要删除 |
| **恢复条件** | 升级CUDA toolkit到11.7以上，或找到预编译的flash_attn wheel |

### 排除原因详解

HY-Embodied-0.5使用了腾讯自研的**MoT（Mixture of Transformers）架构**，需要加载自定义Python代码：
- `configuration_hunyuan_vl_mot.py`
- `modeling_hunyuan_vl_mot.py`
- `processing_hunyuan_vl_mot.py`

这些代码硬依赖`flash_attn`库，而`flash_attn`需要CUDA >= 11.7编译。当前环境：
- nvcc: 11.5（不满足）
- PyTorch CUDA: 12.6
- GPU驱动 CUDA: 12.4
