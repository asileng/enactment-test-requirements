#!/bin/bash
# launch_experiments_transformers.sh - 使用transformers直接加载模型的实验启动脚本

source ~/miniconda3/etc/profile.d/conda.sh
conda activate NLP2

cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements

mkdir -p logs

# 模型配置：模型名 GPU编号
declare -A MODEL_GPU=(
    ["HY-Embodied-0.5"]="0"
    ["Hunyuan-1.8B-Instruct"]="0"
    ["Mimo-7B-SFT"]="1"
    ["Mimo-embodied-7B"]="1"
    ["Mimo-VL-7B-SFT-2508"]="0"
    ["Qwen2.5-7B-Instruct"]="1"
    ["Qwen2.5-VL-7B-Instruct"]="0"
    ["RoboBrain2.0-7B"]="1"
)

for MODEL_NAME in "${!MODEL_GPU[@]}"; do
    GPU_ID="${MODEL_GPU[$MODEL_NAME]}"
    MODEL_PATH="/home/xitongzhang/models/${MODEL_NAME}"
    SESSION_NAME="exp-${MODEL_NAME}"
    
    echo "=== 启动 ${SESSION_NAME}: ${MODEL_NAME} (GPU ${GPU_ID}) ==="
    
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null
    tmux new-session -d -s "${SESSION_NAME}"
    
    tmux send-keys -t "${SESSION_NAME}" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate NLP2" Enter
    tmux send-keys -t "${SESSION_NAME}" "cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements" Enter
    tmux send-keys -t "${SESSION_NAME}" "CUDA_VISIBLE_DEVICES=${GPU_ID} python run_experiment_transformers.py \
        --models ${MODEL_PATH} \
        --task task1 --language zh \
        --repeat 12 \
        --device auto \
        --output-dir results/${MODEL_NAME}_task1_zh \
        2>&1 | tee logs/${MODEL_NAME}_task1_zh.log" Enter
    
    echo "✓ ${SESSION_NAME} 已启动"
done

# 监控session
echo "=== 启动监控session ==="
tmux kill-session -t "exp-monitor" 2>/dev/null
tmux new-session -d -s "exp-monitor"
tmux send-keys -t "exp-monitor" "cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements" Enter
tmux send-keys -t "exp-monitor" "watch -n 30 'echo \"=== GPU状态 ===\" && nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader && echo \"\" && echo \"=== 实验进度 ===\" && for dir in results/*/; do if [ -d \"\$dir\" ]; then total=\$(ls \"\$dir\"/*.json 2>/dev/null | grep -v summary_ | grep -v tracker | wc -l); echo \"\$dir: \$total 条\"; fi; done'" Enter

echo "✓ exp-monitor 已启动"
echo ""
echo "=== 所有session已启动 ==="
