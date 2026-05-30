#!/bin/bash
# 批量 pilot 测试脚本
# 用法: bash run_all_pilots.sh

MODELS=(
    "Mimo-VL-7B-SFT-2508"
    "Mimo-7B-base"
    "HY-Embodied-0.5-X"
    "Hunyuan-1.8B-Instruct"
)

MODEL_DIR="/root/autodl-fs/model"
WORK_DIR="/root/EVtest/enactment-test-requirements"
PORT=8000
PARTICIPANT_ID=5

cd "$WORK_DIR"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始测试: $MODEL"
    echo "=========================================="
    
    # 清理 GPU
    pkill -9 -f "vllm" 2>/dev/null
    sleep 3
    # 强制清理残留 GPU 进程
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 2
    echo "GPU 已清理: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
    
    # 启动 vllm
    echo "启动 vllm..."
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR/$MODEL" \
        --host localhost --port $PORT \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code --max-model-len 4096 \
        > /tmp/vllm_${MODEL}.log 2>&1 &
    
    # 等待就绪
    echo "等待 vllm 就绪..."
    READY=0
    for i in $(seq 1 60); do
        if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q "$MODEL"; then
            READY=1
            echo "vllm 就绪! (${i}x5s)"
            break
        fi
        sleep 5
    done
    
    if [ "$READY" -eq 0 ]; then
        echo "ERROR: $MODEL 启动失败，跳过"
        continue
    fi
    
    # 运行 4 个实验
    for TASK in task1 task2; do
        for LANG in zh en; do
            echo "运行: ${TASK}_${LANG}..."
            timeout 600 python3 run_experiment.py \
                --task $TASK --language $LANG \
                --participant-id $PARTICIPANT_ID \
                --output-dir "pilot_results/${MODEL}/${TASK}_${LANG}" \
                2>&1 | grep -E "入组率|有效结果|无效结果"
        done
    done
    
    echo "$MODEL 测试完成!"
done

echo ""
echo "=========================================="
echo "所有模型测试完成!"
echo "=========================================="
