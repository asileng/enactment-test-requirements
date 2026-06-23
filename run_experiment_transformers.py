# -*- coding: utf-8 -*-
"""
大模型测评任务执行脚本（Transformers版本）
使用transformers直接加载模型进行推理，无需vLLM服务
支持任务1（JSON编码）和任务2（文本描述）
具备断点续传、无效数据自动补采、详细日志、进度条等功能
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
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText, AutoProcessor

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
    INFERENCE_PARAMS, USE_LATIN_SQUARE, LATIN_SQUARE_ORDERS,
    LANGUAGE, CODING_DIMENSIONS, DESCRIPTION_DIMENSIONS
)


# ==================== 错误类型定义 ====================
class ExperimentError:
    """错误类型枚举"""
    # 模型相关错误
    MODEL_LOAD_ERROR = "E001"
    MODEL_INFERENCE_ERROR = "E002"
    CUDA_OOM_ERROR = "E003"
    
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


# ==================== Transformers客户端 ====================
class TransformersClient:
    """Transformers本地模型客户端"""
    
    # 类级别缓存，避免重复加载模型
    _model_cache = {}
    _tokenizer_cache = {}
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self, logger: logging.Logger = None):
        """加载模型和分词器，自动检测VLM模型"""
        if self.model_path in TransformersClient._model_cache:
            self.model = TransformersClient._model_cache[self.model_path]
            self.tokenizer = TransformersClient._tokenizer_cache[self.model_path]
            if logger:
                logger.info(f"使用缓存的模型: {self.model_name}")
            return
        
        if logger:
            logger.info(f"开始加载模型: {self.model_path}")
        
        try:
            # 检测模型类型
            config_path = os.path.join(self.model_path, "config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            architectures = config.get("architectures", [])
            is_vlm = any("VL" in arch or "VLMoT" in arch or "MultiModal" in arch for arch in architectures)
            
            if is_vlm:
                # VLM模型使用AutoModelForImageTextToText
                if logger:
                    logger.info(f"检测到VLM模型: {architectures}")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                # VLM模型可能还有processor
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_path,
                        trust_remote_code=True
                    )
                except:
                    self.processor = None
            else:
                # 普通LLM模型
                if logger:
                    logger.info(f"检测到LLM模型: {architectures}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True
                )
            
            # 缓存模型
            TransformersClient._model_cache[self.model_path] = self.model
            TransformersClient._tokenizer_cache[self.model_path] = self.tokenizer
            
            if logger:
                logger.info(f"模型加载完成: {self.model_name}")
                
        except Exception as e:
            if logger:
                logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def call(self, prompt: str, logger: logging.Logger = None) -> Tuple[str, Optional[str]]:
        """使用transformers进行推理
        
        Returns:
            Tuple[str, Optional[str]]: (响应内容, 错误类型)
        """
        try:
            # 确保模型已加载
            if self.model is None:
                self.load_model(logger)
            
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # 推理参数
            gen_kwargs = {
                "max_new_tokens": INFERENCE_PARAMS.get("max_tokens", 500),
                "temperature": INFERENCE_PARAMS.get("temperature", 0.7),
                "top_p": INFERENCE_PARAMS.get("top_p", 0.9),
                "do_sample": True,
            }
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # 解码输出（只取新生成的部分）
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return response, None
            
        except torch.cuda.OutOfMemoryError:
            if logger:
                logger.error("CUDA内存不足")
            torch.cuda.empty_cache()
            return "ERROR: CUDA out of memory", ExperimentError.CUDA_OOM_ERROR
            
        except Exception as e:
            if logger:
                logger.error(f"推理失败: {str(e)}")
            return f"ERROR: {str(e)}", ExperimentError.MODEL_INFERENCE_ERROR


def get_client(model_path: str, device: str = "auto") -> TransformersClient:
    """返回transformers客户端"""
    return TransformersClient(model_path, device=device)


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
    if language == "en":
        template_path = template_path.replace(".txt", "_en.txt") if "_en" not in template_path else template_path
    
    template_full_path = os.path.join(os.path.dirname(__file__), template_path)
    
    if not os.path.exists(template_full_path):
        raise FileNotFoundError(f"{ExperimentError.CONFIG_TEMPLATE_ERROR}: 模板文件不存在: {template_full_path}")

    with open(template_full_path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_prompt(template: str, verb: str) -> str:
    """生成完整提示词"""
    return template.replace("{verb}", verb)


# ==================== 拉丁方顺序 ====================
def get_verb_order(participant_id: int = None, language: str = None) -> List[str]:
    """获取动词顺序（支持拉丁方）"""
    language = language or LANGUAGE
    
    if language == "en":
        from config_en import LATIN_SQUARE_ORDERS as en_orders
        orders = en_orders
        default_verbs = ["fling", "chuck", "cast", "throw", "hurl", "toss"]
    else:
        orders = LATIN_SQUARE_ORDERS
        default_verbs = VERBS
    
    if not USE_LATIN_SQUARE:
        return default_verbs

    if participant_id is not None:
        idx = participant_id % len(orders)
        return orders[idx]

    return random.choice(orders)


# ==================== thinking标签提取 ====================
def extract_thinking(response: str) -> Tuple[str, Optional[str]]:
    """提取thinking标签内容"""
    thinking_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if thinking_match:
        thinking_content = thinking_match.group(1).strip()
        cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        return cleaned_response, thinking_content
    return response.strip(), None


# ==================== 响应解析 ====================
def parse_task1_response(response: str) -> Tuple[Optional[Dict], Optional[str]]:
    """解析任务1的JSON响应"""
    response, _ = extract_thinking(response)
    
    try:
        # 尝试直接解析JSON
        data = json.loads(response)
        
        # 检查是否是单个JSON对象
        if isinstance(data, dict):
            required_fields = ["FORCE", "ARM", "HAND", "VD", "HD"]
            if all(field in data for field in required_fields):
                return data, None
        
        return None, ExperimentError.PARSE_MISSING_FIELDS
    except json.JSONDecodeError:
        # 尝试从文本中提取JSON
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                required_fields = ["FORCE", "ARM", "HAND", "VD", "HD"]
                if all(field in data for field in required_fields):
                    return data, None
            except json.JSONDecodeError:
                pass
        
        return None, ExperimentError.PARSE_JSON_ERROR


def validate_task1_result(result: Dict) -> Tuple[bool, Optional[str]]:
    """验证任务1结果的编码值范围"""
    if not result:
        return False, ExperimentError.PARSE_MISSING_FIELDS

    try:
        force = int(result.get("FORCE", -1))
        arm = int(result.get("ARM", -1))
        hand = int(result.get("HAND", -1))
        vd = int(result.get("VD", -1))
        hd = int(result.get("HD", -1))

        if not (1 <= force <= 5):
            return False, ExperimentError.PARSE_INVALID_VALUES
        if arm not in [0, 1]:
            return False, ExperimentError.PARSE_INVALID_VALUES
        if not (0 <= hand <= 12):
            return False, ExperimentError.PARSE_INVALID_VALUES
        if vd not in [0, 1]:
            return False, ExperimentError.PARSE_INVALID_VALUES
        if hd not in [0, 1]:
            return False, ExperimentError.PARSE_INVALID_VALUES

    except (ValueError, TypeError):
        return False, ExperimentError.PARSE_INVALID_VALUES

    return True, None


def parse_task2_response(response: str, language: str = None) -> Tuple[Optional[Dict], Optional[str]]:
    """解析任务2的模型响应，提取文本描述中的维度信息"""
    response, _ = extract_thinking(response)
    
    language = language or LANGUAGE
    
    if language == "en":
        from config_en import DESCRIPTION_DIMENSIONS as dimensions
    else:
        dimensions = DESCRIPTION_DIMENSIONS

    result = {}
    response_lower = response.lower()
    response_clean = re.sub(r'[^\w\s]', ' ', response_lower)
    response_words = set(response_clean.split())

    for dim_name, dim_config in dimensions.items():
        options = dim_config["options"]
        found = False

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
    """验证任务2结果是否包含所有维度"""
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
    client: TransformersClient,
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

    parsed_result, parse_error = parse_response(raw_response, task_id, language)
    
    is_valid = False
    validation_error = None
    if parsed_result:
        is_valid, validation_error = validate_result(parsed_result, task_id)

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
    """保存单次实验结果"""
    try:
        os.makedirs(output_dir, exist_ok=True)

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
        
        error_type = result.get("error_type")
        if error_type:
            if error_type not in model_stats[model]["errors"]:
                model_stats[model]["errors"][error_type] = 0
            model_stats[model]["errors"][error_type] += 1
        
        model_stats[model]["durations"].append(result.get("duration_seconds", 0))
    
    for model in model_stats:
        durations = model_stats[model]["durations"]
        model_stats[model]["avg_duration"] = round(sum(durations) / len(durations), 2) if durations else 0
        model_stats[model]["valid_rate"] = round(model_stats[model]["valid"] / model_stats[model]["total"] * 100, 1) if model_stats[model]["total"] > 0 else 0
        del model_stats[model]["durations"]
    
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
                    "examples": []
                }
            error_stats[error_type]["count"] += 1
            
            if len(error_stats[error_type]["examples"]) < 3:
                error_stats[error_type]["examples"].append({
                    "model": result["model"],
                    "verb": result["verb"],
                    "raw_response": result.get("raw_response", "")[:200] if result.get("raw_response") else None
                })
    
    return error_stats


# ==================== 汇总保存 ====================
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


# ==================== 主实验函数 ====================
def run_experiment(
    models: List[str] = None,
    verbs: List[str] = None,
    output_dir: str = None,
    participant_id: int = None,
    language: str = None,
    task_id: str = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    resume: bool = True,
    device: str = "auto",
    logger: logging.Logger = None
):
    """运行完整实验，支持断点续传和无效数据自动补采"""
    models = models or MODELS
    output_dir = output_dir or RESULTS_DIR
    language = language or LANGUAGE
    task_id = task_id or CURRENT_TASK

    if logger is None:
        logger = setup_logging()

    experiment_start_time = datetime.now()

    task_config = TASKS.get(task_id)
    if not task_config:
        logger.error(f"{ExperimentError.CONFIG_TASK_ERROR}: 未知任务ID: {task_id}")
        return []

    verb_order = get_verb_order(participant_id, language)

    template = load_prompt_template(task_id, language)
    all_results = []

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
    logger.info(f"断点续传: {resume}")
    logger.info(f"设备: {device}")
    logger.info(f"开始时间: {experiment_start_time.isoformat()}")
    logger.info("=" * 60)

    if HAS_TQDM:
        pbar = tqdm(total=total_experiments, desc="实验进度", unit="exp")
    else:
        pbar = None

    for model_path in models:
        model_name = os.path.basename(model_path)
        logger.info(f"\n[模型] {model_name}")
        logger.info(f"[路径] {model_path}")

        client = get_client(model_path, device=device)

        # 预加载模型
        try:
            client.load_model(logger)
        except Exception as e:
            logger.error(f"模型加载失败，跳过: {str(e)}")
            # 跳过该模型的所有实验
            for _ in verb_order:
                for _ in range(REPEAT_COUNT):
                    completed += 1
                    if pbar:
                        pbar.update(1)
            continue

        for verb in verb_order:
            logger.info(f"  [动词] {verb}")

            for repeat_idx in range(REPEAT_COUNT):
                completed += 1

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
                max_auto_retries = 50
                
                while True:
                    result = run_single_experiment(
                        client, verb, template, task_id, language, repeat_idx, logger
                    )
                    retry_count += 1
                    
                    retry_history.append({
                        "attempt": retry_count,
                        "is_valid": result["is_valid"],
                        "error_type": result.get("error_type"),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    filepath = save_result(result, output_dir, logger)
                    if filepath:
                        all_results.append(result)
                    else:
                        logger.warning(f"结果保存失败，跳过此实验记录")
                        if pbar:
                            pbar.update(1)
                        break
                    
                    if result["is_valid"]:
                        result["retry_count"] = retry_count
                        result["retry_history"] = retry_history
                        
                        if filepath and os.path.exists(filepath):
                            with open(filepath, 'w', encoding='utf-8') as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)
                        
                        if tracker:
                            tracker.mark_completed(model_name, verb, repeat_idx)
                        if retry_count > 1:
                            logger.info(f"    ✓ 第{retry_count}次尝试成功")
                        break
                    else:
                        logger.warning(f"    ✗ 第{retry_count}次尝试无效 (错误: {result.get('error_type')})，自动补采...")
                        
                        if retry_count >= max_auto_retries:
                            logger.error(f"    ✗ 已达最大重试次数({max_auto_retries})，跳过此实验")
                            if tracker:
                                tracker.mark_completed(model_name, verb, repeat_idx)
                            break
                        
                        if filepath and os.path.exists(filepath):
                            os.remove(filepath)
                            if result in all_results:
                                all_results.remove(result)

                if pbar:
                    pbar.update(1)

        # 释放模型显存
        del client.model
        del client.tokenizer
        if model_path in TransformersClient._model_cache:
            del TransformersClient._model_cache[model_path]
        if model_path in TransformersClient._tokenizer_cache:
            del TransformersClient._tokenizer_cache[model_path]
        torch.cuda.empty_cache()

    if pbar:
        pbar.close()

    experiment_end_time = datetime.now()
    experiment_duration = (experiment_end_time - experiment_start_time).total_seconds()

    model_stats = calculate_model_statistics(all_results)
    error_stats = calculate_error_statistics(all_results)

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
    logger.info(f"汇总文件: {summary_path}")
    logger.info("=" * 60)

    return all_results


# ==================== 命令行入口 ====================
def main():
    global REPEAT_COUNT
    parser = argparse.ArgumentParser(description="大模型测评任务执行脚本（Transformers版本）")
    parser.add_argument("--models", nargs="+", help="要测评的模型路径列表")
    parser.add_argument("--verbs", nargs="+", help="要测评的动词列表")
    parser.add_argument("--output-dir", default=RESULTS_DIR, help=f"结果输出目录")
    parser.add_argument("--repeat", type=int, default=REPEAT_COUNT, help=f"每个组合重复次数")
    parser.add_argument("--participant-id", type=int, default=None, help="参与者ID")
    parser.add_argument("--language", choices=["zh", "en"], default=LANGUAGE, help="语言版本")
    parser.add_argument("--task", choices=list(TASKS.keys()), default=CURRENT_TASK, help="任务ID")
    parser.add_argument("--device", default="auto", help="设备（auto/cpu/cuda/cuda:0/cuda:1）")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="重试延迟时间（秒）")
    parser.add_argument("--no-resume", action="store_true", help="禁用断点续传功能")

    args = parser.parse_args()

    REPEAT_COUNT = args.repeat

    run_experiment(
        models=args.models,
        verbs=args.verbs,
        output_dir=args.output_dir,
        participant_id=args.participant_id,
        language=args.language,
        task_id=args.task,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        resume=not args.no_resume,
        device=args.device
    )


if __name__ == "__main__":
    main()
