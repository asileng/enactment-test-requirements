# 并行执行指南

## 概述

本指南说明如何通过启动多个母脚本实现并行实验，以提高多模型测试效率。

---

## 基本原理

由于当前脚本采用顺序执行方式，当需要测试多个模型时，可以通过启动多个独立进程实现并行：

```
终端1: 模型A → [动词1, 动词2, ..., 动词6] → 结果目录A
终端2: 模型B → [动词1, 动词2, ..., 动词6] → 结果目录B
终端3: 模型C → [动词1, 动词2, ..., 动词6] → 结果目录C
```

---

## 使用方法

### 方法1：单模型并行

每个终端运行一个模型：

```bash
# 终端1 - 模型1
python run_all_experiments.py \
    --mode single \
    --task task1 \
    --language zh \
    --models /path/to/model1 \
    --output-dir results/model1_task1_zh

# 终端2 - 模型2
python run_all_experiments.py \
    --mode single \
    --task task1 \
    --language zh \
    --models /path/to/model2 \
    --output-dir results/model2_task1_zh

# 终端3 - 模型3
python run_all_experiments.py \
    --mode single \
    --task task1 \
    --language zh \
    --models /path/to/model3 \
    --output-dir results/model3_task1_zh
```

### 方法2：任务并行

同一模型的不同任务可以并行：

```bash
# 终端1 - 任务1
python run_all_experiments.py \
    --mode single \
    --task task1 \
    --language zh \
    --models /path/to/model1 \
    --output-dir results/model1_task1_zh

# 终端2 - 任务2
python run_all_experiments.py \
    --mode single \
    --task task2 \
    --language zh \
    --models /path/to/model1 \
    --output-dir results/model1_task2_zh
```

### 方法3：语言版本并行

同一模型同一任务的不同语言版本可以并行：

```bash
# 终端1 - 中文版
python run_all_experiments.py \
    --mode single \
    --task task1 \
    --language zh \
    --models /path/to/model1 \
    --output-dir results/model1_task1_zh

# 终端2 - 英文版
python run_all_experiments.py \
    --mode single \
    --task task1 \
    --language en \
    --models /path/to/model1 \
    --output-dir results/model1_task1_en
```

---

## 关键注意事项

### 1. 输出目录必须不同

**问题**：如果多个进程使用相同的输出目录，会导致：
- 文件名冲突（虽然已添加UUID，但仍可能混淆）
- tracker文件冲突（断点续传信息混乱）
- 汇总文件覆盖

**解决方案**：每个进程使用独立的输出目录，建议命名规范：

```
results/{模型名}_{任务}_{语言}_{时间戳}/
```

示例：
```
results/Llama-2-7b_task1_zh_20240115/
results/Qwen-7B_task1_zh_20240115/
results/Llama-2-7b_task2_en_20240115/
```

### 2. vllm服务并发限制

**问题**：如果多个进程同时调用同一个vllm服务，可能导致：
- 请求排队，响应变慢
- 内存溢出
- 服务崩溃

**解决方案**：

**方案A：单vllm服务，控制并发**
- 限制同时运行的进程数（建议2-3个）
- 增加`TIMEOUT`配置（建议120-180秒）
- 增加`--max-retries`参数（建议5次）

**方案B：多vllm服务，完全并行**
```bash
# 启动多个vllm服务，使用不同端口
python -m vllm.entrypoints.openai.api_server --model /path/to/model1 --port 8000
python -m vllm.entrypoints.openai.api_server --model /path/to/model2 --port 8001
python -m vllm.entrypoints.openai.api_server --model /path/to/model3 --port 8002

# 每个进程连接不同的vllm服务
python run_all_experiments.py --mode single --models /path/to/model1 --host localhost --port 8000 --output-dir results/model1
python run_all_experiments.py --mode single --models /path/to/model2 --host localhost --port 8001 --output-dir results/model2
python run_all_experiments.py --mode single --models /path/to/model3 --host localhost --port 8002 --output-dir results/model3
```

### 3. GPU内存管理

**问题**：多个vllm服务同时运行会占用大量GPU内存

**解决方案**：
- 使用`--gpu-memory-utilization`参数控制每个服务的GPU内存使用
- 使用`--tensor-parallel-size`参数控制张量并行
- 监控GPU内存使用：`nvidia-smi`

示例：3个模型共享24GB GPU内存
```bash
# 每个服务使用约8GB
python -m vllm.entrypoints.openai.api_server --model /path/to/model1 --port 8000 --gpu-memory-utilization 0.3
python -m vllm.entrypoints.openai.api_server --model /path/to/model2 --port 8001 --gpu-memory-utilization 0.3
python -m vllm.entrypoints.openai.api_server --model /path/to/model3 --port 8002 --gpu-memory-utilization 0.3
```

### 4. 日志分离

**问题**：多个进程的日志可能混在一起

**解决方案**：每个进程使用不同的日志目录：

```bash
# 方法1：使用环境变量（需要修改脚本）
LOG_DIR=logs/model1 python run_all_experiments.py ...

# 方法2：查看日志文件名
# 日志文件名格式：experiment_{时间戳}.log
# 每个进程会生成独立的日志文件
```

### 5. 断点续传独立性

**说明**：每个输出目录的tracker文件是独立的，互不影响

**注意事项**：
- 如果某个进程中断，只需重新运行该进程即可
- 不要手动复制tracker文件到其他目录
- 如果需要重新运行某个模型，删除该目录下的`experiment_tracker.json`

---

## 并行执行脚本模板

### Linux/Mac

```bash
#!/bin/bash
# parallel_run.sh - 并行运行多个模型

MODELS=(
    "/path/to/model1"
    "/path/to/model2"
    "/path/to/model3"
)

TASK="task1"
LANGUAGE="zh"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 启动并行任务
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

echo "所有任务已启动，等待完成..."
wait
echo "所有任务完成！"
```

### Windows (PowerShell)

```powershell
# parallel_run.ps1 - 并行运行多个模型

$MODELS = @(
    "/path/to/model1",
    "/path/to/model2",
    "/path/to/model3"
)

$TASK = "task1"
$LANGUAGE = "zh"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

# 启动并行任务
$jobs = @()
foreach ($model in $MODELS) {
    $model_name = Split-Path $model -Leaf
    $output_dir = "results/${model_name}_${TASK}_${LANGUAGE}_${TIMESTAMP}"
    
    Write-Host "启动: $model_name -> $output_dir"
    $jobs += Start-Job -ScriptBlock {
        param($model, $output_dir, $task, $language)
        python run_all_experiments.py --mode single --task $task --language $language --models $model --output-dir $output_dir --max-retries 5 --retry-delay 10
    } -ArgumentList $model, $output_dir, $TASK, $LANGUAGE
}

Write-Host "所有任务已启动，等待完成..."
$jobs | Wait-Job
Write-Host "所有任务完成！"
```

---

## 结果合并

并行执行完成后，需要合并各目录的结果：

```python
# merge_results.py - 合并多个目录的结果

import os
import json
from pathlib import Path
from datetime import datetime

def merge_results(input_dirs, output_dir):
    """合并多个目录的结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for input_dir in input_dirs:
        # 查找汇总文件
        summary_files = list(Path(input_dir).glob("summary_*.json"))
        
        if summary_files:
            # 使用最新的汇总文件
            latest_summary = max(summary_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_summary, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                all_results.extend(summary.get("all_results", []))
    
    # 保存合并后的结果
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

# 使用示例
if __name__ == "__main__":
    input_dirs = [
        "results/model1_task1_zh_20240115",
        "results/model2_task1_zh_20240115",
        "results/model3_task1_zh_20240115"
    ]
    output_dir = "results/merged"
    
    merge_results(input_dirs, output_dir)
```

---

## 监控脚本

### 实时监控进度

```bash
#!/bin/bash
# monitor_progress.sh - 监控所有并行任务的进度

echo "=== 并行任务监控 ==="
echo "时间: $(date)"
echo ""

for dir in results/*/; do
    if [ -d "$dir" ]; then
        # 统计已完成的实验数
        completed=$(ls "$dir"/*.json 2>/dev/null | grep -v "summary_" | grep -v "experiment_tracker" | wc -l)
        
        # 读取tracker文件
        tracker_file="$dir/experiment_tracker.json"
        if [ -f "$tracker_file" ]; then
            tracked=$(python -c "import json; print(len(json.load(open('$tracker_file'))['completed']))")
        else
            tracked=0
        fi
        
        echo "$dir: 已完成 $completed 个文件, tracker记录 $tracked 个"
    fi
done
```

### 监控GPU使用

```bash
# 每5秒刷新一次GPU状态
watch -n 5 nvidia-smi
```

---

## 常见问题

### Q1: 并行运行时某个任务失败了怎么办？

**A**: 
1. 查看对应目录的日志文件
2. 修复问题后重新运行该任务（断点续传会自动跳过已完成的实验）
3. 不需要重新运行其他任务

### Q2: 如何知道所有任务都完成了？

**A**:
1. 查看各目录的汇总文件（`summary_*.json`）
2. 检查日志文件中的"实验完成!"信息
3. 使用监控脚本查看进度

### Q3: 并行运行会影响结果质量吗？

**A**: 
- 不会，每个实验是独立的
- 唯一需要注意的是vllm服务的负载，如果过载可能导致响应变慢

### Q4: 最多可以同时运行多少个进程？

**A**:
- 取决于GPU内存和vllm服务配置
- 建议：单vllm服务时2-3个进程，多vllm服务时每个服务1个进程
- 监控GPU内存使用，确保不超过80%

### Q5: 如何设置优先级，让某些模型先完成？

**A**:
- 先启动优先级高的模型
- 或者使用不同的终端，手动控制进度
