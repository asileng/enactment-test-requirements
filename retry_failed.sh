#!/bin/bash
# 重试失败实验脚本（修复版）
cd /root/EVtest/enactment-test-requirements

MODEL_DIR="/root/autodl-fs/model"

start_vllm() {
    local MODEL=$1
    local MODEL_PATH="$MODEL_DIR/$MODEL"
    
    # 清理旧进程
    pkill -9 -f "vllm" 2>/dev/null
    sleep 3
    
    # 确保 GPU 内存释放
    while true; do
        FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | awk '{print $1}')
        if [ "$FREE" -gt 40000 ]; then
            echo "GPU 已清理: ${FREE}MiB 可用"
            break
        fi
        echo "等待 GPU 释放... (${FREE}MiB)"
        sleep 5
    done
    
    # 启动 vllm
    echo "启动 vllm: $MODEL"
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --host localhost --port 8000 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code --max-model-len 4096 \
        > /tmp/vllm_retry.log 2>&1 &
    
    # 等待 vllm 完全就绪（检查日志中的 "Application startup complete"）
    echo "等待 vllm 就绪..."
    for i in $(seq 1 60); do
        if grep -q "Application startup complete" /tmp/vllm_retry.log 2>/dev/null; then
            echo "vllm 就绪! (${i}x5s)"
            return 0
        fi
        sleep 5
    done
    
    echo "ERROR: vllm 启动超时"
    return 1
}

retry_experiment() {
    local MODEL=$1
    local TASK=$2
    local LANG=$3
    local VERB=$4
    local OUTPUT_DIR="pilot_results/${MODEL}/${TASK}_${LANG}"
    
    # 删除跟踪器，强制重试
    rm -f "$OUTPUT_DIR/experiment_tracker.json"
    
    echo "  重试: ${TASK}_${LANG} ${VERB}"
    timeout 300 python3 run_experiment.py \
        --task $TASK --language $LANG \
        --models "$MODEL_DIR/$MODEL" \
        --verbs $VERB \
        --participant-id 5 \
        --output-dir $OUTPUT_DIR \
        --no-resume \
        2>&1 | grep -E "有效|无效|错误"
}

# Mimo-VL-7B-SFT-2508
echo ""
echo "=========================================="
echo "=== Mimo-VL-7B-SFT-2508 ==="
echo "=========================================="
start_vllm "Mimo-VL-7B-SFT-2508"
retry_experiment "Mimo-VL-7B-SFT-2508" "task1" "en" "摔"

# Mimo-7B-base
echo ""
echo "=========================================="
echo "=== Mimo-7B-base ==="
echo "=========================================="
start_vllm "Mimo-7B-base"
retry_experiment "Mimo-7B-base" "task1" "en" "扔"
for verb in 抛 扔 丢 甩; do
    retry_experiment "Mimo-7B-base" "task2" "en" "$verb"
done

# Hunyuan-1.8B-Instruct
echo ""
echo "=========================================="
echo "=== Hunyuan-1.8B-Instruct ==="
echo "=========================================="
start_vllm "Hunyuan-1.8B-Instruct"
for verb in 投 扔 摔 甩; do
    retry_experiment "Hunyuan-1.8B-Instruct" "task2" "en" "$verb"
done

# RoboBrain2.0-7B
echo ""
echo "=========================================="
echo "=== RoboBrain2.0-7B ==="
echo "=========================================="
start_vllm "RoboBrain2.0-7B"
for verb in 抛 扔; do
    retry_experiment "RoboBrain2.0-7B" "task1" "en" "$verb"
done
for verb in 扔 丢 甩; do
    retry_experiment "RoboBrain2.0-7B" "task2" "zh" "$verb"
done
for verb in 抛 扔 丢; do
    retry_experiment "RoboBrain2.0-7B" "task2" "en" "$verb"
done

# Qwen2.5-VL-7B-Instruct
echo ""
echo "=========================================="
echo "=== Qwen2.5-VL-7B-Instruct ==="
echo "=========================================="
start_vllm "Qwen2.5-VL-7B-Instruct"
retry_experiment "Qwen2.5-VL-7B-Instruct" "task2" "en" "扔"

echo ""
echo "=========================================="
echo "=== 重试完成 ==="
echo "=========================================="
