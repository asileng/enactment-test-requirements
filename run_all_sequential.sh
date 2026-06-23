#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate NLP2
cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements

echo "=== 开始顺序执行所有模型实验 ==="
echo "时间: $(date)"

# 模型列表: 模型名:GPU编号
MODELS=(
    "HY-Embodied-0.5:0"
    "Mimo-VL-7B-SFT-2508:0"
    "Qwen2.5-VL-7B-Instruct:0"
    "Hunyuan-1.8B-Instruct:0"
    "Mimo-7B-SFT:1"
    "Mimo-embodied-7B:1"
    "Qwen2.5-7B-Instruct:1"
    "RoboBrain2.0-7B:1"
)

for model_info in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $model_info | cut -d':' -f1)
    GPU_ID=$(echo $model_info | cut -d':' -f2)
    MODEL_PATH="/home/xitongzhang/models/${MODEL_NAME}"
    
    echo ""
    echo "=========================================="
    echo "开始模型: $MODEL_NAME (GPU $GPU_ID)"
    echo "时间: $(date)"
    echo "=========================================="
    
    for task in task1 task2; do
        for lang in zh en; do
            echo "--- 运行: $task $lang ---"
            CUDA_VISIBLE_DEVICES=$GPU_ID python run_experiment_transformers.py \
                --models $MODEL_PATH \
                --task $task --language $lang \
                --repeat 12 \
                --device auto \
                --output-dir results/${MODEL_NAME}_${task}_${lang} \
                2>&1 | tee logs/${MODEL_NAME}_${task}_${lang}.log
            echo "--- 完成: $task $lang ---"
        done
    done
    
    echo "=========================================="
    echo "完成模型: $MODEL_NAME"
    echo "时间: $(date)"
    echo "=========================================="
done

echo ""
echo "=== 所有实验完成 ==="
echo "时间: $(date)"
