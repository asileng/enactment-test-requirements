#!/bin/bash
# monitor_checkpoint.sh - 监控实验进度，每完成一个任务汇报一次，并记录错误统计

source ~/miniconda3/etc/profile.d/conda.sh
conda activate NLP2

cd /home/xitongzhang/vllm_verb_aquisition/enactment-test-requirements

# 记录已完成的任务
COMPLETED_MIMO7B=0
COMPLETED_MIMOEMB=0

# 错误统计
MIMO7B_E104=0
MIMOEMB_E104=0

echo "=== 开始监控实验进度 ==="
echo "时间: $(date)"
echo ""

while true; do
    # 检查Mimo-7B-SFT task2_zh
    if [ "$COMPLETED_MIMO7B" -eq 0 ]; then
        # 统计E104错误
        new_errors=$(grep -c "E104" logs/Mimo-7B-SFT_task2_zh.log 2>/dev/null)
        if [ "$new_errors" -gt "$MIMO7B_E104" ]; then
            MIMO7B_E104=$new_errors
            echo "[$(date +%H:%M:%S)] Mimo-7B-SFT task2_zh E104错误: $MIMO7B_E104 次"
        fi
        
        if grep -q "实验完成!" logs/Mimo-7B-SFT_task2_zh.log 2>/dev/null; then
            COMPLETED_MIMO7B=1
            echo ""
            echo "=========================================="
            echo "✓ 任务完成: Mimo-7B-SFT task2_zh"
            echo "时间: $(date)"
            echo "=========================================="
            total=$(ls results/Mimo-7B-SFT_task2_zh/*.json 2>/dev/null | grep -v summary_ | wc -l)
            echo "结果文件: $total 条"
            echo "E104错误: $MIMO7B_E104 次"
            echo ""
        else
            progress=$(grep -o "[0-9]*/72" logs/Mimo-7B-SFT_task2_zh.log 2>/dev/null | tail -1)
            echo "[$(date +%H:%M:%S)] Mimo-7B-SFT task2_zh: ${progress:-0/72}"
        fi
    fi
    
    # 检查Mimo-embodied-7B task1_en
    if [ "$COMPLETED_MIMOEMB" -eq 0 ]; then
        # 统计E104错误
        new_errors=$(grep -c "E104" logs/Mimo-embodied-7B_task1_en.log 2>/dev/null)
        if [ "$new_errors" -gt "$MIMOEMB_E104" ]; then
            MIMOEMB_E104=$new_errors
            echo "[$(date +%H:%M:%S)] Mimo-embodied-7B task1_en E104错误: $MIMOEMB_E104 次"
        fi
        
        if grep -q "实验完成!" logs/Mimo-embodied-7B_task1_en.log 2>/dev/null; then
            COMPLETED_MIMOEMB=1
            echo ""
            echo "=========================================="
            echo "✓ 任务完成: Mimo-embodied-7B task1_en"
            echo "时间: $(date)"
            echo "=========================================="
            total=$(ls results/Mimo-embodied-7B_task1_en/*.json 2>/dev/null | grep -v summary_ | wc -l)
            echo "结果文件: $total 条"
            echo "E104错误: $MIMOEMB_E104 次"
            echo ""
        else
            progress=$(grep -o "[0-9]*/72" logs/Mimo-embodied-7B_task1_en.log 2>/dev/null | tail -1)
            echo "[$(date +%H:%M:%S)] Mimo-embodied-7B task1_en: ${progress:-0/72}"
        fi
    fi
    
    # 检查是否所有任务都完成
    if [ "$COMPLETED_MIMO7B" -eq 1 ] && [ "$COMPLETED_MIMOEMB" -eq 1 ]; then
        echo ""
        echo "=== 所有监控的任务已完成 ==="
        echo "错误统计汇总:"
        echo "  Mimo-7B-SFT task2_zh E104: $MIMO7B_E104 次"
        echo "  Mimo-embodied-7B task1_en E104: $MIMOEMB_E104 次"
        break
    fi
    
    sleep 60
done
