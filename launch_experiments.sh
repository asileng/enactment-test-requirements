#!/bin/bash
# launch_experiments.sh - 启动所有实验的tmux session

source ~/miniconda3/etc/profile.d/conda.sh
conda activate NLP2-vllm

cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements

# 创建日志目录
mkdir -p logs

# 模型配置：模型路径 GPU编号 端口 session名
declare -A MODEL_CONFIGS=(
    ["HY-Embodied-0.5"]="0 8000 exp-01"
    ["Hunyuan-1.8B-Instruct"]="0 8001 exp-02"
    ["Mimo-7B-SFT"]="0 8002 exp-03"
    ["Mimo-embodied-7B"]="0 8003 exp-04"
    ["Mimo-VL-7B-SFT-2508"]="1 8004 exp-05"
    ["Qwen2.5-7B-Instruct"]="1 8005 exp-06"
    ["Qwen2.5-VL-7B-Instruct"]="1 8006 exp-07"
    ["RoboBrain2.0-7B"]="1 8007 exp-08"
)

for MODEL_NAME in "${!MODEL_CONFIGS[@]}"; do
    read GPU_ID PORT SESSION_NAME <<< "${MODEL_CONFIGS[$MODEL_NAME]}"
    MODEL_PATH="/home/xitongzhang/models/${MODEL_NAME}"
    
    echo "=== 启动 ${SESSION_NAME}: ${MODEL_NAME} (GPU ${GPU_ID}, Port ${PORT}) ==="
    
    # 创建tmux session
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null
    tmux new-session -d -s "${SESSION_NAME}"
    
    # 窗口0: vLLM服务
    tmux send-keys -t "${SESSION_NAME}" "CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --host localhost \
        --port ${PORT} \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.35 \
        --max-model-len 4096 \
        --trust-remote-code \
        2>&1 | tee logs/vllm_${SESSION_NAME}.log" Enter
    
    # 等待vLLM启动
    echo "等待vLLM服务启动 (30秒)..."
    sleep 30
    
    # 窗口1: 数据采集脚本
    tmux new-window -t "${SESSION_NAME}"
    tmux send-keys -t "${SESSION_NAME}" "cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements" Enter
    tmux send-keys -t "${SESSION_NAME}" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate NLP2" Enter
    
    # 运行4个任务
    tmux send-keys -t "${SESSION_NAME}" "for task in task1 task2; do \
        for lang in zh en; do \
            echo \"=== 运行: \${task} \${lang} ===\"; \
            python run_experiment.py \
                --task \${task} --language \${lang} \
                --models ${MODEL_PATH} \
                --host localhost --port ${PORT} \
                --repeat 12 \
                --output-dir results/${MODEL_NAME}_\${task}_\${lang} \
                --max-retries 5 --retry-delay 10 \
                2>&1 | tee logs/${MODEL_NAME}_\${task}_\${lang}.log; \
            echo \"=== 完成: \${task} \${lang} ===\"; \
        done; \
    done; \
    echo \"=== 全部完成: ${MODEL_NAME} ===\"" Enter
    
    echo "✓ ${SESSION_NAME} 已启动"
done

# 创建监控session
echo "=== 启动监控session ==="
tmux kill-session -t "exp-monitor" 2>/dev/null
tmux new-session -d -s "exp-monitor"
tmux send-keys -t "exp-monitor" "cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements" Enter
tmux send-keys -t "exp-monitor" "watch -n 30 'echo \"=== 实验进度监控 ===\" && echo \"时间: \$(date)\" && echo \"\" && for dir in results/*/; do if [ -d \"\$dir\" ]; then total=\$(ls \"\$dir\"/*.json 2>/dev/null | grep -v summary_ | grep -v tracker | wc -l); valid=\$(python3 -c \"import json,glob; count=0; [count:=count+1 for f in glob.glob(\\\"\${dir}/*.json\\\") if \\\"summary_\\\" not in f and \\\"tracker\\\" not in f and json.load(open(f)).get(\\\"is_valid\\\",False)]\" 2>/dev/null || echo 0); echo \"\$dir: \$total 条\"; fi; done && echo \"\" && echo \"GPU使用:\" && nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader'" Enter

echo "✓ exp-monitor 已启动"
echo ""
echo "=== 所有session已启动 ==="
echo "使用 'tmux list-sessions' 查看所有session"
echo "使用 'tmux attach -t exp-01' 进入某个session"
