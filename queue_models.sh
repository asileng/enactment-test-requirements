#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate NLP2
cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements
mkdir -p logs

declare -A MODEL_TASKS
MODEL_TASKS["Mimo-7B-SFT"]="task2:zh"
MODEL_TASKS["Mimo-embodied-7B"]="task1:en"
MODEL_TASKS["Qwen2.5-7B-Instruct"]="task1:zh task1:en task2:zh task2:en"
MODEL_TASKS["RoboBrain2.0-7B"]="task1:zh task1:en task2:zh task2:en"

MODEL_ORDER=("Mimo-7B-SFT" "Mimo-embodied-7B" "Qwen2.5-7B-Instruct" "RoboBrain2.0-7B")

declare -A GPU_BUSY
GPU_BUSY[0]=""
GPU_BUSY[1]=""

safe_session_name() {
    echo "$1" | sed 's/\./_/g; s/-/_/g'
}

check_gpu_memory() {
    local free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 2>/dev/null)
    if [ "$free_mb" -ge 20480 ]; then
        return 0
    else
        return 1
    fi
}

get_available_gpu() {
    for gpu_id in 0 1; do
        if [ -z "${GPU_BUSY[$gpu_id]}" ] && check_gpu_memory $gpu_id; then
            echo $gpu_id
            return 0
        fi
    done
    echo ""
    return 1
}

is_model_done() {
    local model_name=$1
    local tasks=${MODEL_TASKS[$model_name]}
    for task_lang in $tasks; do
        local task=$(echo $task_lang | cut -d: -f1)
        local lang=$(echo $task_lang | cut -d: -f2)
        local sname=$(safe_session_name "${model_name}_${task}_${lang}")
        local pids=$(tmux list-panes -t "$sname" -F "#{pane_pid}" 2>/dev/null)
        if [ -n "$pids" ]; then
            for pid in $pids; do
                local child=$(pgrep -P $pid 2>/dev/null | head -1)
                if [ -n "$child" ]; then
                    return 1
                fi
            done
        fi
    done
    return 0
}

run_model_tasks() {
    local model_name=$1
    local gpu_id=$2
    local tasks=${MODEL_TASKS[$model_name]}
    local model_path="/home/xitongzhang/models/${model_name}"
    
    echo ""
    echo "=========================================="
    echo "开始模型: $model_name (GPU $gpu_id)"
    echo "时间: $(date)"
    echo "=========================================="
    
    GPU_BUSY[$gpu_id]=$model_name
    
    for task_lang in $tasks; do
        local task=$(echo $task_lang | cut -d: -f1)
        local lang=$(echo $task_lang | cut -d: -f2)
        local sname=$(safe_session_name "${model_name}_${task}_${lang}")
        local output_dir="results/${model_name}_${task}_${lang}"
        
        tmux kill-session -t "$sname" 2>/dev/null
        tmux new-session -d -s "$sname"
        
        tmux send-keys -t "$sname" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate NLP2" Enter
        tmux send-keys -t "$sname" "cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements" Enter
        tmux send-keys -t "$sname" "CUDA_VISIBLE_DEVICES=$gpu_id python run_experiment_transformers.py --models $model_path --task $task --language $lang --repeat 12 --device auto --output-dir $output_dir 2>&1 | tee logs/${model_name}_${task}_${lang}.log" Enter
        
        echo "  启动: $sname"
    done
    
    # 验证启动
    sleep 30
    local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null)
    if [ "$mem_used" -gt 1000 ]; then
        echo "  ✓ 验证通过: GPU $gpu_id 内存占用 ${mem_used}MB"
    else
        echo "  ⚠ 警告: GPU $gpu_id 无内存占用，实验可能未启动"
    fi
}

echo "=== 开始模型排队运行 ==="
echo "时间: $(date)"
echo ""

model_idx=0
total_models=${#MODEL_ORDER[@]}

while [ $model_idx -lt $total_models ]; do
    available_gpu=$(get_available_gpu)
    
    if [ -n "$available_gpu" ]; then
        model=${MODEL_ORDER[$model_idx]}
        run_model_tasks "$model" "$available_gpu"
        model_idx=$((model_idx + 1))
        sleep 10
    else
        # 检查GPU状态
        for gpu_id in 0 1; do
            if [ -n "${GPU_BUSY[$gpu_id]}" ]; then
                model_name=${GPU_BUSY[$gpu_id]}
                local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null)
                if [ "$mem_used" -lt 1000 ]; then
                    echo "  ⚠ GPU $gpu_id 标记忙碌但无占用，重置状态"
                    GPU_BUSY[$gpu_id]=""
                elif is_model_done "$model_name"; then
                    echo "  ✓ $model_name 在GPU $gpu_id上完成"
                    GPU_BUSY[$gpu_id]=""
                fi
            fi
        done
        
        echo "  等待5分钟..."
        sleep 300
    fi
done

echo ""
echo "所有模型已启动，等待完成..."
while true; do
    all_done=true
    for gpu_id in 0 1; do
        if [ -n "${GPU_BUSY[$gpu_id]}" ]; then
            model_name=${GPU_BUSY[$gpu_id]}
            if ! is_model_done "$model_name"; then
                all_done=false
            else
                echo "  ✓ $model_name 完成"
                GPU_BUSY[$gpu_id]=""
            fi
        fi
    done
    if $all_done; then break; fi
    sleep 60
done

echo ""
echo "=== 所有模型完成 ==="
echo "时间: $(date)"
