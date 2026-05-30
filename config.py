# -*- coding: utf-8 -*-
"""
大模型测评任务配置文件
在此文件中定义所有可替换的变量
"""

# ==================== 模型配置 ====================
# vllm本地模型配置
# 模型路径或名称（用于vllm加载）
MODELS = [
    # 本地模型路径示例
    "/path/to/your/model1",
    "/path/to/your/model2",
    # 或使用Hugging Face模型名称
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen-7B-Chat",
    # 添加更多模型...
]

# vllm服务配置
VLLM_CONFIG = {
    "host": "localhost",           # vllm服务地址
    "port": 8000,                  # vllm服务端口
    "tensor_parallel_size": 1,     # 张量并行大小
    "gpu_memory_utilization": 0.9, # GPU内存使用率
    "max_model_len": 4096,         # 最大模型长度
    "trust_remote_code": True,     # 是否信任远程代码
}

# ==================== 动词配置 ====================
# 6个目标动词（中文）
VERBS = [
    "投",
    "扔",
    "摔",
    "丢",
    "甩",
    "抛",
]

# 拉丁方顺序配置（用于平衡顺序效应）
# 6个动词的拉丁方排列
LATIN_SQUARE_ORDERS = [
    ["投", "扔", "摔", "丢", "甩", "抛"],
    ["扔", "摔", "丢", "甩", "抛", "投"],
    ["摔", "丢", "甩", "抛", "投", "扔"],
    ["丢", "甩", "抛", "投", "扔", "摔"],
    ["甩", "抛", "投", "扔", "摔", "丢"],
    ["抛", "投", "扔", "摔", "丢", "甩"],
]

# ==================== 任务配置 ====================
# 支持的任务列表
TASKS = {
    "task1": {
        "name": "动作编码测评",
        "description": "对不同动词对应的动作进行五维度编码（JSON输出）",
        "prompt_template": "prompt_template.txt",
        "output_format": "json",
    },
    "task2": {
        "name": "动作描述测评",
        "description": "对不同动词对应的动作进行一句话描述",
        "prompt_template": "prompt_template_task2.txt",
        "output_format": "text",
    },
}

# 当前任务（默认任务1）
CURRENT_TASK = "task1"

# 语言版本
LANGUAGE = "zh"  # "zh" 或 "en"

# ==================== 输出配置 ====================
# 结果保存目录
RESULTS_DIR = "results"

# 结果文件名格式：{model}_{verb}_{timestamp}.json
RESULT_FILENAME_FORMAT = "{model}_{verb}_{timestamp}.json"

# 汇总文件名
SUMMARY_FILENAME = "summary_{timestamp}.json"

# ==================== 实验配置 ====================
# 每个组合重复实验次数（用于稳定性测试）
REPEAT_COUNT = 1

# 是否保存原始响应
SAVE_RAW_RESPONSE = True

# 超时时间（秒）
TIMEOUT = 120

# 使用拉丁方顺序（True）或固定顺序（False）
USE_LATIN_SQUARE = True

# ==================== vllm推理参数 ====================
INFERENCE_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 500,
    "stop": ["\n\n", "Human:", "Assistant:"],
}

# ==================== 任务1编码维度说明 ====================
CODING_DIMENSIONS = {
    "FORCE": {
        "description": "执行动作时施加于物体的力量大小",
        "scale": {
            5: "非常强",
            4: "强",
            3: "中等",
            2: "弱",
            1: "非常弱"
        }
    },
    "ARM": {
        "description": "动作开始前手臂的初始状态",
        "scale": {
            1: "手臂伸直",
            0: "手臂弯曲"
        }
    },
    "HAND": {
        "description": "动作开始前手部停留的初始高度",
        "scale": {
            0: "地面高度",
            "1-9": "地面高度与执行者身高之间的相对高度",
            "10": "执行者身高",
            "11": "高于执行者身高一个单位",
            "12": "高于执行者身高两个单位"
        }
    },
    "VD": {
        "description": "执行动作时手部的主要垂直运动方向",
        "scale": {
            1: "向下",
            0: "向上"
        }
    },
    "HD": {
        "description": "执行动作时手部的主要水平运动方向",
        "scale": {
            1: "向前",
            0: "向侧方"
        }
    }
}

# ==================== 任务2描述维度说明 ====================
DESCRIPTION_DIMENSIONS = {
    "FORCE": {
        "description": "动作施加于物体的力量大小",
        "options": ["非常强", "强", "中等", "弱", "非常弱"]
    },
    "ARM": {
        "description": "动作开始前手臂的初始状态",
        "options": ["手臂伸直", "手臂弯曲"]
    },
    "HAND": {
        "description": "动作开始前手部停留的大致相对高度",
        "options": ["接近地面", "膝盖高度", "腰部高度", "胸部高度", "肩部高度", "头部高度", "高于头部"]
    },
    "VD": {
        "description": "动作过程中手部的主要垂直运动方向",
        "options": ["向下", "向上"]
    },
    "HD": {
        "description": "动作过程中手部的主要水平运动方向",
        "options": ["向前", "向侧方"]
    }
}
