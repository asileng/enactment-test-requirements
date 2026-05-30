# -*- coding: utf-8 -*-
"""
Large Language Model Evaluation Task Configuration (English Version)
Define all replaceable variables in this file
"""

# ==================== Model Configuration ====================
# vllm local model configuration
# Model paths or names (for vllm loading)
MODELS = [
    # Local model path examples
    "/path/to/your/model1",
    "/path/to/your/model2",
    # Or use Hugging Face model names
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen-7B-Chat",
    # Add more models...
]

# vllm service configuration
VLLM_CONFIG = {
    "host": "localhost",           # vllm service address
    "port": 8000,                  # vllm service port
    "tensor_parallel_size": 1,     # Tensor parallel size
    "gpu_memory_utilization": 0.9, # GPU memory utilization
    "max_model_len": 4096,         # Maximum model length
    "trust_remote_code": True,     # Whether to trust remote code
}

# ==================== Verb Configuration ====================
# 6 target verbs (English)
VERBS = [
    "fling",
    "chuck",
    "cast",
    "throw",
    "hurl",
    "toss",
]

# Latin Square order configuration (for balancing order effects)
# 6 verbs in Latin Square arrangement
LATIN_SQUARE_ORDERS = [
    ["fling", "chuck", "cast", "throw", "hurl", "toss"],
    ["chuck", "cast", "throw", "hurl", "toss", "fling"],
    ["cast", "throw", "hurl", "toss", "fling", "chuck"],
    ["throw", "hurl", "toss", "fling", "chuck", "cast"],
    ["hurl", "toss", "fling", "chuck", "cast", "throw"],
    ["toss", "fling", "chuck", "cast", "throw", "hurl"],
]

# ==================== Task Configuration ====================
# Supported tasks list
TASKS = {
    "task1": {
        "name": "Action Encoding Evaluation",
        "description": "Encode actions corresponding to different verbs across five dimensions (JSON output)",
        "prompt_template": "prompt_template_en.txt",
        "output_format": "json",
    },
    "task2": {
        "name": "Action Description Evaluation",
        "description": "Describe actions corresponding to different verbs in one sentence",
        "prompt_template": "prompt_template_task2_en.txt",
        "output_format": "text",
    },
}

# Current task (default task1)
CURRENT_TASK = "task1"

# Language version
LANGUAGE = "en"  # "zh" or "en"

# ==================== Output Configuration ====================
# Results save directory
RESULTS_DIR = "results_en"

# Result filename format: {model}_{verb}_{timestamp}.json
RESULT_FILENAME_FORMAT = "{model}_{verb}_{timestamp}.json"

# Summary filename
SUMMARY_FILENAME = "summary_{timestamp}.json"

# ==================== Experiment Configuration ====================
# Number of repetitions per combination (for stability testing)
REPEAT_COUNT = 1

# Whether to save raw responses
SAVE_RAW_RESPONSE = True

# Timeout in seconds
TIMEOUT = 120

# Use Latin Square order (True) or fixed order (False)
USE_LATIN_SQUARE = True

# ==================== vllm Inference Parameters ====================
INFERENCE_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 500,
    "stop": ["\n\n", "Human:", "Assistant:"],
}

# ==================== Task 1 Coding Dimensions Description ====================
CODING_DIMENSIONS = {
    "FORCE": {
        "description": "The amount of force applied to the object when performing the action",
        "scale": {
            5: "very strong",
            4: "strong",
            3: "moderate",
            2: "weak",
            1: "very weak"
        }
    },
    "ARM": {
        "description": "The initial state of the arm before the action begins",
        "scale": {
            1: "arm extended",
            0: "arm bent"
        }
    },
    "HAND": {
        "description": "The initial height where the hand rests before the action begins",
        "scale": {
            0: "ground level",
            "1-9": "relative height between ground level and the performer's height",
            "10": "performer's height",
            "11": "one unit above the performer's height",
            "12": "two units above the performer's height"
        }
    },
    "VD": {
        "description": "The primary vertical direction of hand movement during the action",
        "scale": {
            1: "downward",
            0: "upward"
        }
    },
    "HD": {
        "description": "The primary horizontal direction of hand movement during the action",
        "scale": {
            1: "forward",
            0: "sideways"
        }
    }
}

# ==================== Task 2 Description Dimensions ====================
DESCRIPTION_DIMENSIONS = {
    "FORCE": {
        "description": "The amount of force applied to the object when performing the action",
        "options": ["very strong", "strong", "moderate", "weak", "very weak"]
    },
    "ARM": {
        "description": "The initial state of the arm before the action begins",
        "options": ["arm extended", "arm bent"]
    },
    "HAND": {
        "description": "The approximate relative height where the hand rests before the action begins",
        "options": ["near ground", "knee height", "waist height", "chest height", "shoulder height", "head height", "above head"]
    },
    "VD": {
        "description": "The primary vertical direction of hand movement during the action",
        "options": ["downward", "upward"]
    },
    "HD": {
        "description": "The primary horizontal direction of hand movement during the action",
        "options": ["forward", "sideways"]
    }
}
