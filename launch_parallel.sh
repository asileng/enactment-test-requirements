#!/bin/bash
# launch_parallel.sh - 并行运行剩余模型

source ~/miniconda3/etc/profile.d/conda.sh
conda activate NLP2

cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements
mkdir -p logs

# 剩余模型配置：模型名:GPU编号
MODELS=(
    "HY-Embodied-0.5:0"
    "Mimo-7B-SFT:0"
    "Mimo-embodied-7B:0"
    "Qwen2.5-7B-Instruct:0"
    "RoboBrain2.0-7B:0"
)

echo "=== 启动并行实验 ==="

for model_info in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $model_info | cut -d':' -f1)
    GPU_ID=$(echo $model_info | cut -d':' -f2)
    MODEL_PATH="/home/xitongzhang/models/${MODEL_NAME}"
    SESSION_NAME="exp-${MODEL_NAME}"
    
    echo "启动: $MODEL_NAME (GPU $GPU_ID)"
    
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null
    tmux new-session -d -s "$SESSION_NAME"
    
    tmux send-keys -t "$SESSION_NAME" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate NLP2" Enter
    tmux send-keys -t "$SESSION_NAME" "cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements" Enter
    tmux send-keys -t "$SESSION_NAME" "CUDA_VISIBLE_DEVICES=$GPU_ID python run_experiment_transformers.py \
        --models $MODEL_PATH \
        --task task1 --language zh \
        --repeat 12 \
        --device auto \
        --output-dir results/${MODEL_NAME}_task1_zh \
        2>&1 | tee logs/${MODEL_NAME}_task1_zh.log" Enter
    
    echo "  ✓ $SESSION_NAME 已启动"
done

# 启动监控
echo ""
echo "启动监控..."
tmux new-session -d -s "exp-monitor"
tmux send-keys -t "exp-monitor" "watch -n 30 'echo \"=== GPU状态 ===\" && nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader && echo \"\" && echo \"=== 实验进度 ===\" && for dir in results/*/; do if [ -d \"\$dir\" ]; then total=\$(ls \"\$dir\"/*.json 2>/dev/null | grep -v summary_ | grep -v tracker | wc -l); echo \"\$(basename \$dir): \$total 条\"; fi; done'" Enter

echo ""
echo "=== 并行实验已启动 ==="
echo "使用 'tmux list-sessions' 查看所有session"
echo "使用 'tmux attach -t exp-MODEL_NAME' 进入某个session"
