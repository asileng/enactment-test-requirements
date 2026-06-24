#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate NLP2
cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements
mkdir -p logs

# 任务定义
TASKS=("Mimo-7B-SFT:task2:zh" "Mimo-embodied-7B:task1:en" "Qwen2.5-7B-Instruct:task1:zh" "Qwen2.5-7B-Instruct:task1:en" "Qwen2.5-7B-Instruct:task2:zh" "Qwen2.5-7B-Instruct:task2:en" "RoboBrain2.0-7B:task1:zh" "RoboBrain2.0-7B:task1:en" "RoboBrain2.0-7B:task2:zh" "RoboBrain2.0-7B:task2:en")

safe_name() { echo "$1" | sed 's/\./_/g; s/-/_/g'; }

gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 2>/dev/null; }

gpu_used_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null; }

check_process_running() {
    local sname=$1
    local pid=$(tmux list-panes -t "$sname" -F "#{pane_pid}" 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        local child=$(pgrep -P $pid 2>/dev/null | head -1)
        [ -n "$child" ] && return 0
    fi
    return 1
}

wait_for_gpu_memory() {
    local gpu_id=$1
    local timeout=120
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        local used=$(gpu_used_mb $gpu_id)
        if [ "$used" -gt 1000 ]; then
            echo "  ✓ GPU $gpu_id 内存占用: ${used}MB"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "  ✗ GPU $gpu_id 超时无内存占用"
    return 1
}

run_task() {
    local spec=$1
    local gpu_id=$2
    IFS=':' read -r model task lang <<< "$spec"
    local sname=$(safe_name "${model}_${task}_${lang}")
    local output_dir="results/${model}_${task}_${lang}"
    local model_path="/home/xitongzhang/models/${model}"
    
    # 检查是否已完成（通过日志判断）
    local log_file="logs/${model}_${task}_${lang}.log"
    if [ -f "$log_file" ] && grep -q "跳过（已完成）: 72" "$log_file" 2>/dev/null; then
        echo "  跳过: ${model} ${task}_${lang} (已完成)"
        return 0
    fi
    
    echo "  启动: ${model} ${task}_${lang} (GPU $gpu_id)"
    
    tmux kill-session -t "$sname" 2>/dev/null
    tmux new-session -d -s "$sname"
    tmux send-keys -t "$sname" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate NLP2 && cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements" Enter
    tmux send-keys -t "$sname" "CUDA_VISIBLE_DEVICES=$gpu_id python run_experiment_transformers.py --models $model_path --task $task --language $lang --repeat 12 --device auto --output-dir $output_dir 2>&1 | tee logs/${model}_${task}_${lang}.log" Enter
    
    # 等待GPU内存占用
    sleep 15
    if ! wait_for_gpu_memory $gpu_id; then
        echo "  ✗ 任务启动失败: ${model} ${task}_${lang}"
        return 1
    fi
    
    # 等待任务完成
    echo "  等待: ${model} ${task}_${lang} 完成..."
    while check_process_running "$sname"; do
        sleep 30
    done
    
    echo "  ✓ 完成: ${model} ${task}_${lang}"
    return 0
}

echo "=== 开始实验队列 ==="
echo "时间: $(date)"
echo "任务数: ${#TASKS[@]}"
echo ""

completed=0
failed=0

for spec in "${TASKS[@]}"; do
    IFS=':' read -r model task lang <<< "$spec"
    echo "[$((completed + failed + 1))/${#TASKS[@]}] 处理: $model $task $lang"
    
    # 选择GPU
    gpu_id=-1
    for g in 0 1; do
        free=$(gpu_free_mb $g)
        if [ "$free" -gt 20000 ]; then
            gpu_id=$g
            break
        fi
    done
    
    if [ $gpu_id -eq -1 ]; then
        echo "  等待GPU释放..."
        while [ $gpu_id -eq -1 ]; do
            sleep 60
            for g in 0 1; do
                free=$(gpu_free_mb $g)
                if [ "$free" -gt 20000 ]; then
                    gpu_id=$g
                    break
                fi
            done
        done
    fi
    
    if run_task "$spec" $gpu_id; then
        completed=$((completed + 1))
    else
        failed=$((failed + 1))
    fi
done

echo ""
echo "=== 实验队列完成 ==="
echo "完成: $completed"
echo "失败: $failed"
echo "时间: $(date)"
