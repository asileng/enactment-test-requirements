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

### 2.5 错误记录要求

**即使自动补采成功，也必须保留错误统计。**

每个任务的错误统计格式：
```json
{
  "errors": {
    "E104": 8,    // 任务2描述值无效，出现8次
    "E101": 2     // JSON解析失败，出现2次
  }
}
```

记录位置：
1. `task_tracker.json` 的 `data_status[model][task].errors` 字段
2. `task_tracker.json` 的 `summary.error_statistics` 字段
3. 每个结果文件的 `retry_count` 和 `retry_history` 字段

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

## 9. 主动监控策略（2026-06-22补充，2026-06-23更新）

### 9.1 核心原则

**必须主动检测并处理异常，不能被动等待用户发现。**

**关键判断逻辑**：
- GPU没有占用 = 没有推理在进行
- tmux session存在 ≠ 任务在运行
- 脚本输出"完成" ≠ 任务真正完成

### 9.2 主动监控实现方案

#### 9.2.1 操作后验证机制

每次执行操作后，必须立即验证：

```
操作 → 等待N秒 → 验证结果 → 汇报状态
```

验证内容：
1. **GPU内存占用**：启动实验后30秒检查GPU是否有内存占用
2. **进程存在性**：检查python进程是否真的在运行
3. **日志输出**：检查日志文件是否有新内容
4. **session名称匹配**：tmux session名称是否与预期一致

#### 9.2.2 定期巡检机制

每5分钟执行一次巡检：

| 检查项 | 判断标准 | 处理方式 |
|--------|----------|----------|
| GPU空闲但标记忙碌 | memory.used < 1GB | 重置状态，重新调度 |
| session无进程 | 无python子进程 | 检查日志，判断是否完成 |
| 日志无更新 | 5分钟内无新行 | 检查进程是否卡死 |
| 进度停滞 | 进度条不变 | 检查是否OOM或异常 |
| 错误累积 | 连续5次失败 | 汇报并暂停该任务 |

#### 9.2.3 异常自动恢复

```bash
# 伪代码
while 有任务未完成:
    检查GPU状态
    if GPU空闲 && 有待运行任务:
        启动下一个任务
        等待30秒
        验证GPU是否真的在运行
        if 未运行:
            检查错误日志
            重试或跳过
    elif GPU忙碌:
        验证进程是否真的在运行
        if 未运行:
            重置GPU状态
            重新调度
    等待5分钟
```

### 9.3 tmux session命名规范

**问题**：tmux会将点号`.`替换为下划线`_`，导致session名称不匹配

**解决方案**：统一使用下划线命名

```bash
# 错误写法
SESSION_NAME="Qwen2.5-7B-Instruct_task1_zh"  # 实际变成 Qwen2_5-7B-Instruct_task1_zh

# 正确写法
SESSION_NAME=$(echo "$model_name" | sed 's/\./_/g; s/-/_/g')_${task}_${lang}
```

### 9.4 汇报机制

**必须汇报的情况**：
1. 任务完成时（包含错误统计）
2. 检测到异常时（GPU空闲但标记忙碌、进程消失等）
3. 每次巡检发现异常时
4. 启动新任务后30秒验证结果

**汇报格式**：
```
[时间] [事件类型] 详情
[时间] [验证结果] 操作是否成功
[时间] [当前状态] GPU/进度/错误统计
```

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

## 11. 操作验证与托管规范（2026-06-23补充）

### 11.1 核心原则

**任何操作都必须经过验证才能托管，不能假设成功。**

### 11.2 操作流程规范

```
执行操作
    ↓
等待反馈（合理时间）
    ↓
验证结果
    ├─ 成功 → 汇报 → 托管
    └─ 失败 → 分析原因 → 修复 → 重新执行
```

### 11.3 实验启动验证清单

启动实验后，必须完成以下验证才能托管：

| 步骤 | 验证内容 | 判断标准 | 失败处理 |
|------|----------|----------|----------|
| 1 | tmux session创建 | session存在 | 重新创建 |
| 2 | 命令执行 | session中有输出 | 检查命令语法 |
| 3 | 模型加载 | 日志显示"模型加载完成" | 检查模型路径 |
| 4 | GPU内存占用 | memory.used > 1GB | 检查CUDA_VISIBLE_DEVICES |
| 5 | 实验进行 | 日志有进度更新 | 检查API/连接 |
| 6 | 进程存活 | ps aux有python进程 | 检查是否崩溃 |

### 11.4 定期巡检规范

即使已验证启动成功，仍需定期巡检：

| 频率 | 检查内容 | 异常判断 |
|------|----------|----------|
| 每5分钟 | GPU内存占用 | 内存下降>50%可能进程已结束 |
| 每10分钟 | 日志更新 | 无新输出可能卡死 |
| 每30分钟 | 进度统计 | 进度停滞可能OOM |
| 每任务完成 | 数据统计 | 有效率<80%需检查 |

### 11.5 反馈处理机制

**必须将脚本反馈作为输入处理，不能忽略：**

| 反馈类型 | 示例 | 处理方式 |
|----------|------|----------|
| 成功反馈 | "实验完成! 有效结果: 72" | 记录统计，继续下一个 |
| 警告反馈 | "GPU无内存占用" | 立即检查，可能需要重启 |
| 错误反馈 | "CUDA out of memory" | 停止任务，调整参数 |
| 进度反馈 | "进度: 50%" | 记录，用于预估完成时间 |

### 11.6 托管条件

**只有满足以下所有条件才能托管（不再主动监控）：**

1. ✅ GPU内存占用稳定（>1GB）
2. ✅ 进程正常运行（ps aux可见）
3. ✅ 日志持续更新（每分钟有新输出）
4. ✅ 无错误/警告信息
5. ✅ 进度正常推进

**不满足任何一条，都必须持续监控直到解决。**

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
