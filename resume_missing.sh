#!/bin/bash
# resume_missing.sh - 补采缺失数据，每个GPU同一时间只运行1个模型

source ~/miniconda3/etc/profile.d/conda.sh
conda activate NLP2

cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements
mkdir -p logs

# 缺失数据模型列表（按GPU分组，顺序执行）
# GPU 0: Mimo-7B-SFT, Mimo-embodied-7B, Qwen2.5-7B-Instruct, RoboBrain2.0-7B
# GPU 1: Hunyuan-1.8B-Instruct

echo "=== 开始补采缺失数据 ==="
echo "时间: $(date)"

run_model_tasks() {
    local MODEL_NAME=$1
    local GPU_ID=$2
    local MODEL_PATH="/home/xitongzhang/models/${MODEL_NAME}"
    
    echo ""
    echo "=========================================="
    echo "开始模型: $MODEL_NAME (GPU $GPU_ID)"
    echo "时间: $(date)"
    echo "=========================================="
    
    # 并行启动4个任务
    for task in task1 task2; do
        for lang in zh en; do
            SESSION_NAME="${MODEL_NAME}_${task}_${lang}"
            
            tmux kill-session -t "$SESSION_NAME" 2>/dev/null
            tmux new-session -d -s "$SESSION_NAME"
            
            tmux send-keys -t "$SESSION_NAME" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate NLP2" Enter
            tmux send-keys -t "$SESSION_NAME" "cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements" Enter
            tmux send-keys -t "$SESSION_NAME" "CUDA_VISIBLE_DEVICES=$GPU_ID python run_experiment_transformers.py \
                --models $MODEL_PATH \
                --task $task --language $lang \
                --repeat 12 \
                --device auto \
                --output-dir results/${MODEL_NAME}_${task}_${lang} \
                2>&1 | tee logs/${MODEL_NAME}_${task}_${lang}.log" Enter
            
            echo "  启动: $SESSION_NAME"
        done
    done
    
    # 等待该模型所有任务完成
    echo "  等待 $MODEL_NAME 完成..."
    while true; do
        all_done=true
        for task in task1 task2; do
            for lang in zh en; do
                SESSION_NAME="${MODEL_NAME}_${task}_${lang}"
                pid=$(tmux list-panes -t "$SESSION_NAME" -F "#{pane_pid}" 2>/dev/null | head -1)
                child=$(pgrep -P $pid 2>/dev/null | head -1)
                if [ -n "$child" ]; then
                    all_done=false
                    break 2
                fi
            done
        done
        
        if $all_done; then
            echo "  ✓ $MODEL_NAME 完成"
            break
        fi
        
        sleep 60
    done
}

# GPU 0: 按顺序运行模型
for model in Mimo-7B-SFT Mimo-embodied-7B Qwen2.5-7B-Instruct RoboBrain2.0-7B; do
    run_model_tasks "$model" "0"
done

# GPU 1: Hunyuan
run_model_tasks "Hunyuan-1.8B-Instruct" "1"

echo ""
echo "=== 所有补采完成 ==="
echo "时间: $(date)"
