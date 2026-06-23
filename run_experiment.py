# -*- coding: utf-8 -*-
"""
大模型测评任务执行脚本
支持vllm本地模型推理，自动遍历所有模型和动词组合，记录测评结果
支持任务1（JSON编码）和任务2（文本描述）
具备断点续传、超时重试、详细日志、进度条等功能
"""

import os
import json
import time
import argparse
import random
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re
import requests
from pathlib import Path

# 尝试导入tqdm，如果没有安装则使用简单进度显示
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 导入配置
from config import (
    MODELS, VERBS, TASKS, CURRENT_TASK,
    RESULTS_DIR, RESULT_FILENAME_FORMAT, SUMMARY_FILENAME,
    REPEAT_COUNT, SAVE_RAW_RESPONSE, TIMEOUT,
    VLLM_CONFIG, INFERENCE_PARAMS, USE_LATIN_SQUARE, LATIN_SQUARE_ORDERS,
    LANGUAGE, CODING_DIMENSIONS, DESCRIPTION_DIMENSIONS
)


# ==================== 错误类型定义 ====================
class ExperimentError:
    """错误类型枚举"""
    # API相关错误
    API_TIMEOUT = "E001"
    API_CONNECTION_ERROR = "E002"
    API_HTTP_ERROR = "E003"
    API_UNKNOWN_ERROR = "E004"
    
    # 数据解析错误
    PARSE_JSON_ERROR = "E101"
    PARSE_MISSING_FIELDS = "E102"
    PARSE_INVALID_VALUES = "E103"
    PARSE_TASK2_OPTION_ERROR = "E104"
    
    # 文件操作错误
    FILE_READ_ERROR = "E201"
    FILE_WRITE_ERROR = "E202"
    FILE_PERMISSION_ERROR = "E203"
    FILE_DISK_FULL = "E204"
    
    # 配置错误
    CONFIG_TASK_ERROR = "E301"
    CONFIG_MODEL_ERROR = "E302"
    CONFIG_TEMPLATE_ERROR = "E303"
    
    # 其他错误
    UNKNOWN_ERROR = "E999"


# ==================== 日志配置 ====================
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    logger = logging.getLogger("ExperimentLogger")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ==================== 断点续传跟踪器 ====================
class ExperimentTracker:
    """实验进度跟踪器，支持断点续传"""
    
    def __init__(self, tracker_file: str):
        self.tracker_file = tracker_file
        self.completed_experiments = set()
        self.load_tracker()
    
    def load_tracker(self):
        """加载已完成的实验记录"""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.completed_experiments = set(data.get("completed", []))
            except Exception:
                self.completed_experiments = set()
    
    def save_tracker(self):
        """保存实验记录"""
        try:
            os.makedirs(os.path.dirname(self.tracker_file), exist_ok=True)
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "completed": list(self.completed_experiments),
                    "last_updated": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 保存失败不影响实验继续
            pass
    
    def get_experiment_key(self, model: str, verb: str, repeat_idx: int) -> str:
        """生成实验唯一标识"""
        return f"{model}_{verb}_{repeat_idx}"
    
    def is_completed(self, model: str, verb: str, repeat_idx: int) -> bool:
        """检查实验是否已完成"""
        key = self.get_experiment_key(model, verb, repeat_idx)
        return key in self.completed_experiments
    
    def mark_completed(self, model: str, verb: str, repeat_idx: int):
        """标记实验为已完成"""
        key = self.get_experiment_key(model, verb, repeat_idx)
        self.completed_experiments.add(key)
        self.save_tracker()


# ==================== vllm客户端 ====================
class VLLMClient:
    """vllm本地模型客户端"""

    def __init__(self, model_path: str, host: str = None, port: int = None, 
                 max_retries: int = 3, retry_delay: float = 5.0):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.host = host or VLLM_CONFIG.get("host", "localhost")
        self.port = port or VLLM_CONFIG.get("port", 8000)
        self.base_url = f"http://{self.host}:{self.port}"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def call(self, prompt: str, logger: logging.Logger = None) -> Tuple[str, Optional[str]]:
        """调用vllm API进行推理，支持重试
        
        Returns:
            Tuple[str, Optional[str]]: (响应内容, 错误类型)
        """
        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}/v1/completions"

                payload = {
                    "model": self.model_path,
                    "prompt": prompt,
                    "temperature": INFERENCE_PARAMS.get("temperature", 0.7),
                    "top_p": INFERENCE_PARAMS.get("top_p", 0.9),
                    "max_tokens": INFERENCE_PARAMS.get("max_tokens", 500),
                    "stop": INFERENCE_PARAMS.get("stop", []),
                }

                headers = {"Content-Type": "application/json"}

                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=TIMEOUT
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["text"], None
                else:
                    error_msg = f"HTTP {response.status_code} - {response.text}"
                    if logger:
                        logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {error_msg}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return f"ERROR: {error_msg}", ExperimentError.API_HTTP_ERROR

            except requests.exceptions.Timeout:
                if logger:
                    logger.warning(f"请求超时 (尝试 {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return "ERROR: Request timeout", ExperimentError.API_TIMEOUT
            except requests.exceptions.ConnectionError:
                if logger:
                    logger.warning(f"连接失败 (尝试 {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return f"ERROR: Cannot connect to vllm server at {self.base_url}", ExperimentError.API_CONNECTION_ERROR
            except Exception as e:
                if logger:
                    logger.error(f"未知错误 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return f"ERROR: {str(e)}", ExperimentError.API_UNKNOWN_ERROR

        return "ERROR: Max retries exceeded", ExperimentError.API_UNKNOWN_ERROR


class VLLMChatClient:
    """vllm本地模型Chat格式客户端（适用于Chat模型）"""

    def __init__(self, model_path: str, host: str = None, port: int = None,
                 max_retries: int = 3, retry_delay: float = 5.0):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.host = host or VLLM_CONFIG.get("host", "localhost")
        self.port = port or VLLM_CONFIG.get("port", 8000)
        self.base_url = f"http://{self.host}:{self.port}"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def call(self, prompt: str, logger: logging.Logger = None) -> Tuple[str, Optional[str]]:
        """调用vllm Chat API进行推理，支持重试
        
        Returns:
            Tuple[str, Optional[str]]: (响应内容, 错误类型)
        """
        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}/v1/chat/completions"

                payload = {
                    "model": self.model_path,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": INFERENCE_PARAMS.get("temperature", 0.7),
                    "top_p": INFERENCE_PARAMS.get("top_p", 0.9),
                    "max_tokens": INFERENCE_PARAMS.get("max_tokens", 500),
                    "stop": INFERENCE_PARAMS.get("stop", []),
                }

                headers = {"Content-Type": "application/json"}

                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=TIMEOUT
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"], None
                else:
                    error_msg = f"HTTP {response.status_code} - {response.text}"
                    if logger:
                        logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {error_msg}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return f"ERROR: {error_msg}", ExperimentError.API_HTTP_ERROR

            except requests.exceptions.Timeout:
                if logger:
                    logger.warning(f"请求超时 (尝试 {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return "ERROR: Request timeout", ExperimentError.API_TIMEOUT
            except requests.exceptions.ConnectionError:
                if logger:
                    logger.warning(f"连接失败 (尝试 {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return f"ERROR: Cannot connect to vllm server at {self.base_url}", ExperimentError.API_CONNECTION_ERROR
            except Exception as e:
                if logger:
                    logger.error(f"未知错误 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return f"ERROR: {str(e)}", ExperimentError.API_UNKNOWN_ERROR

        return "ERROR: Max retries exceeded", ExperimentError.API_UNKNOWN_ERROR


def get_client(model_path: str, use_chat: bool = True, 
               max_retries: int = 3, retry_delay: float = 5.0) -> VLLMClient:
    """根据配置返回对应的vllm客户端"""
    if use_chat:
        return VLLMChatClient(model_path, max_retries=max_retries, retry_delay=retry_delay)
    else:
        return VLLMClient(model_path, max_retries=max_retries, retry_delay=retry_delay)


# ==================== 提示词加载 ====================
def load_prompt_template(task_id: str = None, language: str = None) -> str:
    """加载提示词模板"""
    task_id = task_id or CURRENT_TASK
    language = language or LANGUAGE

    task_config = TASKS.get(task_id)
    if not task_config:
        raise ValueError(f"{ExperimentError.CONFIG_TASK_ERROR}: 未知任务ID: {task_id}")

    template_path = task_config["prompt_template"]

    # 根据语言版本调整模板路径
    if language == "en" and not template_path.endswith("_en.txt"):
        en_template_path = template_path.replace(".txt", "_en.txt")
        if os.path.exists(en_template_path):
            template_path = en_template_path

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"{ExperimentError.CONFIG_TEMPLATE_ERROR}: 无法加载模板文件 {template_path}: {str(e)}")


def generate_prompt(template: str, verb: str) -> str:
    """生成完整的提示词"""
    return template.replace("{verb}", verb)


def get_verb_order(participant_id: int = None, language: str = None) -> List[str]:
    """获取动词顺序（支持拉丁方平衡）"""
    language = language or LANGUAGE
    
    # 根据语言加载正确的动词和拉丁方配置
    if language == "en":
        from config_en import VERBS as lang_verbs, LATIN_SQUARE_ORDERS as lang_orders
    else:
        from config import VERBS as lang_verbs, LATIN_SQUARE_ORDERS as lang_orders
    
    if USE_LATIN_SQUARE and lang_orders:
        if participant_id is not None:
            order_index = participant_id % len(lang_orders)
        else:
            order_index = random.randint(0, len(lang_orders) - 1)
        return lang_orders[order_index]
    else:
        return lang_verbs.copy()


# ==================== 响应解析 ====================
def extract_thinking(response: str) -> Tuple[str, Optional[str]]:
    """提取模型响应中的 <think> 内容，返回 (清理后的响应, 思考内容)
    
    Returns:
        Tuple[str, Optional[str]]: (cleaned_response, thinking_content)
    """
    thinking_match = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
    thinking_content = thinking_match.group(1).strip() if thinking_match else None
    
    # 去除 <think>...</think> 标签及其内容
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    cleaned = cleaned.strip()
    
    return cleaned, thinking_content


def parse_task1_response(response: str) -> Tuple[Optional[Dict], Optional[str]]:
    """解析任务1的模型响应，提取JSON对象
    
    Returns:
        Tuple[Optional[Dict], Optional[str]]: (解析结果, 错误类型)
    """
    # 先去除 thinking 标签
    response, _ = extract_thinking(response)
    
    try:
        return json.loads(response), None
    except json.JSONDecodeError:
        pass

    json_pattern = r'\{[^{}]+\}'
    matches = re.findall(json_pattern, response)

    for match in matches:
        try:
            data = json.loads(match)
            required_fields = ["FORCE", "ARM", "HAND", "VD", "HD"]
            if all(field in data for field in required_fields):
                return data, None
        except json.JSONDecodeError:
            continue

    return None, ExperimentError.PARSE_JSON_ERROR


def validate_task1_result(result: Dict) -> Tuple[bool, Optional[str]]:
    """验证任务1结果是否符合编码标准
    
    Returns:
        Tuple[bool, Optional[str]]: (是否有效, 错误类型)
    """
    if not result:
        return False, ExperimentError.PARSE_MISSING_FIELDS

    required_fields = ["FORCE", "ARM", "HAND", "VD", "HD"]
    if not all(field in result for field in required_fields):
        return False, ExperimentError.PARSE_MISSING_FIELDS

    if not (1 <= result.get("FORCE", 0) <= 5):
        return False, ExperimentError.PARSE_INVALID_VALUES
    if result.get("ARM") not in [0, 1]:
        return False, ExperimentError.PARSE_INVALID_VALUES
    if not (0 <= result.get("HAND", -1) <= 12):
        return False, ExperimentError.PARSE_INVALID_VALUES
    if result.get("VD") not in [0, 1]:
        return False, ExperimentError.PARSE_INVALID_VALUES
    if result.get("HD") not in [0, 1]:
        return False, ExperimentError.PARSE_INVALID_VALUES

    return True, None


def parse_task2_response(response: str, language: str = None) -> Tuple[Optional[Dict], Optional[str]]:
    """解析任务2的模型响应，提取文本描述中的维度信息
    
    Returns:
        Tuple[Optional[Dict], Optional[str]]: (解析结果, 错误类型)
    """
    # 先去除 thinking 标签
    response, _ = extract_thinking(response)
    
    language = language or LANGUAGE
    
    # 根据语言加载对应的维度配置
    if language == "en":
        from config_en import DESCRIPTION_DIMENSIONS as dimensions
    else:
        dimensions = DESCRIPTION_DIMENSIONS

    result = {}
    response_lower = response.lower()
    # 去除标点符号后分词，用于模糊匹配
    response_clean = re.sub(r'[^\w\s]', ' ', response_lower)
    response_words = set(response_clean.split())

    for dim_name, dim_config in dimensions.items():
        options = dim_config["options"]
        found = False

        # 第一轮：精确子串匹配
        for option in options:
            if language == "en":
                if option.lower() in response_lower:
                    result[dim_name] = option
                    found = True
                    break
            else:
                if option in response:
                    result[dim_name] = option
                    found = True
                    break

        # 第二轮：模糊匹配（单词级别，忽略顺序）
        if not found and language == "en":
            for option in options:
                option_words = set(option.lower().split())
                if option_words.issubset(response_words):
                    result[dim_name] = option
                    found = True
                    break

        if not found:
            result[dim_name] = None

    return result, None


def validate_task2_result(result: Dict) -> Tuple[bool, Optional[str]]:
    """验证任务2结果是否包含所有维度
    
    Returns:
        Tuple[bool, Optional[str]]: (是否有效, 错误类型)
    """
    if not result:
        return False, ExperimentError.PARSE_MISSING_FIELDS

    required_dims = ["FORCE", "ARM", "HAND", "VD", "HD"]
    for dim in required_dims:
        if dim not in result or result[dim] is None:
            return False, ExperimentError.PARSE_TASK2_OPTION_ERROR

    return True, None


def parse_response(response: str, task_id: str = None, language: str = None) -> Tuple[Optional[Dict], Optional[str]]:
    """根据任务类型解析响应"""
    task_id = task_id or CURRENT_TASK

    if task_id == "task1":
        return parse_task1_response(response)
    elif task_id == "task2":
        return parse_task2_response(response, language)
    else:
        raise ValueError(f"{ExperimentError.CONFIG_TASK_ERROR}: 未知任务ID: {task_id}")


def validate_result(result: Dict, task_id: str = None) -> Tuple[bool, Optional[str]]:
    """根据任务类型验证结果"""
    task_id = task_id or CURRENT_TASK

    if task_id == "task1":
        return validate_task1_result(result)
    elif task_id == "task2":
        return validate_task2_result(result)
    else:
        raise ValueError(f"{ExperimentError.CONFIG_TASK_ERROR}: 未知任务ID: {task_id}")


# ==================== 实验执行 ====================
def run_single_experiment(
    client: VLLMClient,
    verb: str,
    template: str,
    task_id: str = None,
    language: str = None,
    repeat_index: int = 0,
    logger: logging.Logger = None
) -> Dict:
    """运行单次实验"""
    task_id = task_id or CURRENT_TASK
    language = language or LANGUAGE

    prompt = generate_prompt(template, verb)

    if logger:
        logger.info(f"开始实验: 模型={client.model_name}, 动词={verb}, 重复={repeat_index}")

    start_time = time.time()
    raw_response, api_error = client.call(prompt, logger)
    end_time = time.time()

    # 解析响应
    parsed_result, parse_error = parse_response(raw_response, task_id, language)
    
    # 验证结果
    is_valid = False
    validation_error = None
    if parsed_result:
        is_valid, validation_error = validate_result(parsed_result, task_id)

    # 确定错误类型
    error_type = api_error or parse_error or validation_error

    experiment_result = {
        "task_id": task_id,
        "model": client.model_name,
        "model_path": client.model_path,
        "verb": verb,
        "repeat_index": repeat_index,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(end_time - start_time, 2),
        "is_valid": is_valid,
        "parsed_result": parsed_result,
        "error_type": error_type,
    }

    if SAVE_RAW_RESPONSE:
        cleaned_response, thinking_content = extract_thinking(raw_response)
        experiment_result["raw_response"] = cleaned_response
        if thinking_content:
            experiment_result["thinking"] = thinking_content

    if logger:
        status = "有效" if is_valid else "无效"
        error_info = f" (错误: {error_type})" if error_type else ""
        logger.info(f"实验完成: {status}{error_info}, 耗时={experiment_result['duration_seconds']}s")

    return experiment_result


def save_result(result: Dict, output_dir: str, logger: logging.Logger = None) -> Optional[str]:
    """保存单次实验结果
    
    Returns:
        Optional[str]: 文件路径，失败返回None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # 使用毫秒+随机数避免文件名冲突
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        unique_id = uuid.uuid4().hex[:6]
        filename = RESULT_FILENAME_FORMAT.format(
            model=result["model"],
            verb=result["verb"],
            timestamp=f"{timestamp}_{unique_id}"
        )

        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return filepath
    except PermissionError as e:
        if logger:
            logger.error(f"{ExperimentError.FILE_PERMISSION_ERROR}: 权限错误 - {str(e)}")
        return None
    except OSError as e:
        if "No space left" in str(e):
            if logger:
                logger.error(f"{ExperimentError.FILE_DISK_FULL}: 磁盘空间不足")
        else:
            if logger:
                logger.error(f"{ExperimentError.FILE_WRITE_ERROR}: 文件写入错误 - {str(e)}")
        return None
    except Exception as e:
        if logger:
            logger.error(f"{ExperimentError.FILE_WRITE_ERROR}: 未知文件错误 - {str(e)}")
        return None


# ==================== 统计函数 ====================
def calculate_model_statistics(results: List[Dict]) -> Dict:
    """计算每个模型的统计信息"""
    model_stats = {}
    
    for result in results:
        model = result["model"]
        if model not in model_stats:
            model_stats[model] = {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "errors": {},
                "avg_duration": 0,
                "durations": []
            }
        
        model_stats[model]["total"] += 1
        
        if result["is_valid"]:
            model_stats[model]["valid"] += 1
        else:
            model_stats[model]["invalid"] += 1
        
        # 统计错误类型
        error_type = result.get("error_type")
        if error_type:
            if error_type not in model_stats[model]["errors"]:
                model_stats[model]["errors"][error_type] = 0
            model_stats[model]["errors"][error_type] += 1
        
        # 记录耗时
        model_stats[model]["durations"].append(result.get("duration_seconds", 0))
    
    # 计算平均耗时和入组率
    for model in model_stats:
        durations = model_stats[model]["durations"]
        model_stats[model]["avg_duration"] = round(sum(durations) / len(durations), 2) if durations else 0
        model_stats[model]["valid_rate"] = round(model_stats[model]["valid"] / model_stats[model]["total"] * 100, 1) if model_stats[model]["total"] > 0 else 0
        del model_stats[model]["durations"]  # 删除原始耗时列表以节省空间
    
    return model_stats


def calculate_error_statistics(results: List[Dict]) -> Dict:
    """计算错误类型统计"""
    error_stats = {}
    
    for result in results:
        error_type = result.get("error_type")
        if error_type:
            if error_type not in error_stats:
                error_stats[error_type] = {
                    "count": 0,
                    "description": "",
                    "examples": []
                }
            error_stats[error_type]["count"] += 1
            
            # 保留最多3个示例
            if len(error_stats[error_type]["examples"]) < 3:
                error_stats[error_type]["examples"].append({
                    "model": result["model"],
                    "verb": result["verb"],
                    "raw_response": result.get("raw_response", "")[:100] if result.get("raw_response") else ""
                })
    
    # 添加错误描述
    error_descriptions = {
        ExperimentError.API_TIMEOUT: "API请求超时",
        ExperimentError.API_CONNECTION_ERROR: "API连接失败",
        ExperimentError.API_HTTP_ERROR: "API返回HTTP错误",
        ExperimentError.API_UNKNOWN_ERROR: "API未知错误",
        ExperimentError.PARSE_JSON_ERROR: "JSON解析失败",
        ExperimentError.PARSE_MISSING_FIELDS: "缺少必需字段",
        ExperimentError.PARSE_INVALID_VALUES: "编码值超出范围",
        ExperimentError.PARSE_TASK2_OPTION_ERROR: "任务2描述值无效",
        ExperimentError.FILE_READ_ERROR: "文件读取错误",
        ExperimentError.FILE_WRITE_ERROR: "文件写入错误",
        ExperimentError.FILE_PERMISSION_ERROR: "文件权限错误",
        ExperimentError.FILE_DISK_FULL: "磁盘空间不足",
        ExperimentError.CONFIG_TASK_ERROR: "任务配置错误",
        ExperimentError.CONFIG_MODEL_ERROR: "模型配置错误",
        ExperimentError.CONFIG_TEMPLATE_ERROR: "模板配置错误",
        ExperimentError.UNKNOWN_ERROR: "未知错误",
    }
    
    for error_type in error_stats:
        error_stats[error_type]["description"] = error_descriptions.get(error_type, "未知错误类型")
    
    return error_stats


# ==================== 主实验函数 ====================
def run_experiment(
    models: List[str] = None,
    verbs: List[str] = None,
    output_dir: str = None,
    participant_id: int = None,
    use_chat: bool = True,
    language: str = None,
    task_id: str = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    resume: bool = True,
    logger: logging.Logger = None
):
    """运行完整实验，支持断点续传"""
    models = models or MODELS
    output_dir = output_dir or RESULTS_DIR
    language = language or LANGUAGE
    task_id = task_id or CURRENT_TASK

    # 设置日志
    if logger is None:
        logger = setup_logging()

    # 记录实验开始时间
    experiment_start_time = datetime.now()

    # 获取任务配置
    task_config = TASKS.get(task_id)
    if not task_config:
        logger.error(f"{ExperimentError.CONFIG_TASK_ERROR}: 未知任务ID: {task_id}")
        return []

    # 获取动词顺序（支持拉丁方）
    verb_order = get_verb_order(participant_id, language)

    template = load_prompt_template(task_id, language)
    all_results = []

    # 断点续传跟踪器
    tracker_file = os.path.join(output_dir, "experiment_tracker.json")
    tracker = ExperimentTracker(tracker_file) if resume else None

    total_experiments = len(models) * len(verb_order) * REPEAT_COUNT
    completed = 0
    skipped = 0

    logger.info("=" * 60)
    logger.info(f"任务: {task_config['name']}")
    logger.info(f"描述: {task_config['description']}")
    logger.info(f"语言: {language}")
    logger.info(f"模型数: {len(models)}")
    logger.info(f"动词数: {len(verb_order)}")
    logger.info(f"动词顺序: {verb_order}")
    logger.info(f"重复次数: {REPEAT_COUNT}")
    logger.info(f"总实验数: {total_experiments}")
    logger.info(f"使用Chat API: {use_chat}")
    logger.info(f"断点续传: {resume}")
    logger.info(f"最大重试次数: {max_retries}")
    logger.info(f"开始时间: {experiment_start_time.isoformat()}")
    logger.info("=" * 60)

    # 创建进度条
    if HAS_TQDM:
        pbar = tqdm(total=total_experiments, desc="实验进度", unit="exp")
    else:
        pbar = None

    for model_path in models:
        model_name = os.path.basename(model_path)
        logger.info(f"\n[模型] {model_name}")
        logger.info(f"[路径] {model_path}")

        client = get_client(model_path, use_chat, max_retries, retry_delay)

        for verb in verb_order:
            logger.info(f"  [动词] {verb}")

            for repeat_idx in range(REPEAT_COUNT):
                completed += 1

                # 检查是否已完成（断点续传）
                if tracker and tracker.is_completed(model_name, verb, repeat_idx):
                    skipped += 1
                    logger.info(f"    [{completed}/{total_experiments}] 跳过（已完成）")
                    if pbar:
                        pbar.update(1)
                    continue

                logger.info(f"    [{completed}/{total_experiments}] 第{repeat_idx+1}次重复...")

                # 重试机制：无效数据自动补采
                retry_count = 0
                retry_history = []
                max_auto_retries = 50  # 单个实验最大自动重试次数
                
                while True:
                    result = run_single_experiment(
                        client, verb, template, task_id, language, repeat_idx, logger
                    )
                    retry_count += 1
                    
                    # 记录重试历史
                    retry_history.append({
                        "attempt": retry_count,
                        "is_valid": result["is_valid"],
                        "error_type": result.get("error_type"),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # 保存结果（带错误处理）
                    filepath = save_result(result, output_dir, logger)
                    if filepath:
                        all_results.append(result)
                    else:
                        logger.warning(f"结果保存失败，跳过此实验记录")
                        # 保存失败不标记为已完成，允许重试
                        if pbar:
                            pbar.update(1)
                        break
                    
                    # 只有有效结果才标记为已完成
                    if result["is_valid"]:
                        # 记录重试信息到结果
                        result["retry_count"] = retry_count
                        result["retry_history"] = retry_history
                        
                        # 更新已保存的文件
                        if filepath and os.path.exists(filepath):
                            with open(filepath, 'w', encoding='utf-8') as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)
                        
                        if tracker:
                            tracker.mark_completed(model_name, verb, repeat_idx)
                        if retry_count > 1:
                            logger.info(f"    ✓ 第{retry_count}次尝试成功")
                        break
                    else:
                        # 无效结果：记录警告，继续重试
                        logger.warning(f"    ✗ 第{retry_count}次尝试无效 (错误: {result.get('error_type')})，自动补采...")
                        
                        # 检查是否超过最大重试次数
                        if retry_count >= max_auto_retries:
                            logger.error(f"    ✗ 已达最大重试次数({max_auto_retries})，跳过此实验")
                            # 标记为已完成（避免无限循环），但记录为异常
                            if tracker:
                                tracker.mark_completed(model_name, verb, repeat_idx)
                            break
                        
                        # 删除无效结果文件，准备重试
                        if filepath and os.path.exists(filepath):
                            os.remove(filepath)
                            if result in all_results:
                                all_results.remove(result)

                if pbar:
                    pbar.update(1)

    # 关闭进度条
    if pbar:
        pbar.close()

    # 记录实验结束时间
    experiment_end_time = datetime.now()
    experiment_duration = (experiment_end_time - experiment_start_time).total_seconds()

    # 计算统计信息
    model_stats = calculate_model_statistics(all_results)
    error_stats = calculate_error_statistics(all_results)

    # 保存汇总结果
    summary_path = save_summary(
        all_results, output_dir, participant_id, task_id,
        experiment_start_time, experiment_end_time, experiment_duration,
        model_stats, error_stats
    )

    logger.info(f"\n" + "=" * 60)
    logger.info(f"实验完成!")
    logger.info(f"总实验数: {total_experiments}")
    logger.info(f"本次完成: {len(all_results)}")
    logger.info(f"跳过（已完成）: {skipped}")
    logger.info(f"有效结果: {sum(1 for r in all_results if r['is_valid'])}")
    logger.info(f"无效结果: {sum(1 for r in all_results if not r['is_valid'])}")
    logger.info(f"总耗时: {experiment_duration:.1f}秒")
    logger.info(f"开始时间: {experiment_start_time.isoformat()}")
    logger.info(f"结束时间: {experiment_end_time.isoformat()}")
    logger.info(f"汇总文件: {summary_path}")
    logger.info("=" * 60)

    # 打印模型统计摘要
    logger.info("\n" + "=" * 60)
    logger.info("模型统计摘要:")
    for model, stats in model_stats.items():
        logger.info(f"  {model}: 总计={stats['total']}, 有效={stats['valid']}, "
                    f"无效={stats['invalid']}, 入组率={stats['valid_rate']}%, "
                    f"平均耗时={stats['avg_duration']}s")
    logger.info("=" * 60)

    # 打印错误统计摘要
    if error_stats:
        logger.info("\n" + "=" * 60)
        logger.info("错误类型统计:")
        for error_type, stats in error_stats.items():
            logger.info(f"  {error_type}: {stats['description']} - 出现{stats['count']}次")
        logger.info("=" * 60)

    return all_results


def save_summary(
    results: List[Dict], 
    output_dir: str, 
    participant_id: int = None, 
    task_id: str = None,
    start_time: datetime = None,
    end_time: datetime = None,
    duration: float = None,
    model_stats: Dict = None,
    error_stats: Dict = None
) -> str:
    """保存汇总结果"""
    task_id = task_id or CURRENT_TASK
    task_config = TASKS.get(task_id, {})

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = SUMMARY_FILENAME.format(timestamp=timestamp)
    summary_path = os.path.join(output_dir, summary_filename)

    # 按模型和动词组织结果
    summary = {
        "task_id": task_id,
        "task_name": task_config.get("name", ""),
        "task_description": task_config.get("description", ""),
        "language": LANGUAGE,
        "timestamp": datetime.now().isoformat(),
        "experiment_time": {
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "duration_seconds": duration,
        },
        "participant_id": participant_id,
        "use_latin_square": USE_LATIN_SQUARE,
        "total_experiments": len(results),
        "valid_count": sum(1 for r in results if r["is_valid"]),
        "invalid_count": sum(1 for r in results if not r["is_valid"]),
        "valid_rate": round(sum(1 for r in results if r["is_valid"]) / len(results) * 100, 1) if results else 0,
        "model_statistics": model_stats or {},
        "error_statistics": error_stats or {},
        "results_by_model_verb": {},
        "all_results": results
    }

    for result in results:
        model = result["model"]
        verb = result["verb"]

        if model not in summary["results_by_model_verb"]:
            summary["results_by_model_verb"][model] = {}

        if verb not in summary["results_by_model_verb"][model]:
            summary["results_by_model_verb"][model][verb] = []

        summary["results_by_model_verb"][model][verb].append({
            "parsed_result": result["parsed_result"],
            "is_valid": result["is_valid"],
            "duration_seconds": result["duration_seconds"],
            "error_type": result.get("error_type")
        })

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary_path


# ==================== 命令行入口 ====================
def main():
    global REPEAT_COUNT, VLLM_CONFIG
    parser = argparse.ArgumentParser(description="大模型测评任务执行脚本（支持vllm）")
    parser.add_argument(
        "--models",
        nargs="+",
        help="要测评的模型路径列表（默认使用config.py中的配置）"
    )
    parser.add_argument(
        "--verbs",
        nargs="+",
        help="要测评的动词列表（默认使用config.py中的配置）"
    )
    parser.add_argument(
        "--output-dir",
        default=RESULTS_DIR,
        help=f"结果输出目录（默认: {RESULTS_DIR}）"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=REPEAT_COUNT,
        help=f"每个组合重复次数（默认: {REPEAT_COUNT}）"
    )
    parser.add_argument(
        "--participant-id",
        type=int,
        default=None,
        help="参与者ID（用于拉丁方顺序分配）"
    )
    parser.add_argument(
        "--use-chat",
        action="store_true",
        default=True,
        help="使用Chat API格式（默认: True）"
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="使用Completion API格式"
    )
    parser.add_argument(
        "--language",
        choices=["zh", "en"],
        default=LANGUAGE,
        help=f"语言版本（默认: {LANGUAGE}）"
    )
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()),
        default=CURRENT_TASK,
        help=f"任务ID（默认: {CURRENT_TASK}）"
    )
    parser.add_argument(
        "--host",
        default=VLLM_CONFIG.get("host", "localhost"),
        help="vllm服务地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=VLLM_CONFIG.get("port", 8000),
        help="vllm服务端口"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="最大重试次数（默认: 3）"
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="重试延迟时间（秒，默认: 5.0）"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用断点续传功能"
    )

    args = parser.parse_args()

    # 更新重复次数
    REPEAT_COUNT = args.repeat

    # 更新vllm配置
    VLLM_CONFIG["host"] = args.host
    VLLM_CONFIG["port"] = args.port

    # 确定是否使用Chat API
    use_chat = not args.no_chat

    run_experiment(
        models=args.models,
        verbs=args.verbs,
        output_dir=args.output_dir,
        participant_id=args.participant_id,
        use_chat=use_chat,
        language=args.language,
        task_id=args.task,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
