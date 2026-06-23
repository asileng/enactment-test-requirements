#!/bin/bash
# launch_parallel_tasks.sh - 每个模型并行运行4个任务

source ~/miniconda3/etc/profile.d/conda.sh
conda activate NLP2

cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements
mkdir -p logs

# 剩余模型配置：模型名:GPU编号
MODELS=(
    "Mimo-7B-SFT:0"
    "Mimo-embodied-7B:0"
    "Qwen2.5-7B-Instruct:0"
    "RoboBrain2.0-7B:0"
    "Hunyuan-1.8B-Instruct:1"
)

echo "=== 启动并行实验（每个模型4个任务并行）==="

for model_info in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $model_info | cut -d':' -f1)
    GPU_ID=$(echo $model_info | cut -d':' -f2)
    MODEL_PATH="/home/xitongzhang/models/${MODEL_NAME}"
    
    echo "启动: $MODEL_NAME (GPU $GPU_ID)"
    
    # 为每个任务创建独立的tmux窗口
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
            
            echo "  ✓ $SESSION_NAME 已启动"
        done
    done
done

# 启动监控
echo ""
echo "启动监控..."
tmux kill-session -t "exp-monitor" 2>/dev/null
tmux new-session -d -s "exp-monitor"
tmux send-keys -t "exp-monitor" "watch -n 60 'echo \"=== GPU状态 ===\" && nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader && echo \"\" && echo \"=== 实验进度 ===\" && for dir in results/*/; do if [ -d \"\$dir\" ]; then total=\$(ls \"\$dir\"/*.json 2>/dev/null | grep -v summary_ | grep -v tracker | wc -l); echo \"\$(basename \$dir): \$total 条\"; fi; done'" Enter

echo ""
echo "=== 并行实验已启动 ==="
echo "每个模型有4个并行任务（task1_zh, task1_en, task2_zh, task2_en）"
echo "使用 'tmux list-sessions' 查看所有session"
echo "使用 'tmux attach -t MODEL_TASK_LANG' 进入某个session"
