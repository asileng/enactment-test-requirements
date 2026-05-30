# -*- coding: utf-8 -*-
"""
task6 可逆性2实验脚本
使用各模型自己的task1参数化结果作为基准值，让模型反向推断动词
"""

import os
import json
import time
import argparse
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests

# 配置
INFERENCE_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 50,
    "stop": [],
}

ZH_VERBS = ["扔", "丢", "抛", "投", "摔", "甩"]
EN_VERBS = ["throw", "toss", "fling", "cast", "chuck", "hurl"]


def load_prompts(model_name: str) -> Dict:
    """加载指定模型的task6提示词"""
    prompt_file = f"task6_prompts_{model_name}.json"
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"提示词文件不存在: {prompt_file}，请先运行 generate_task6_prompts.py")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return json.load(f)


def call_vllm(prompt: str, model_path: str, host: str = "localhost", port: int = 8000,
              max_retries: int = 3, retry_delay: float = 5.0) -> Tuple[str, Optional[str]]:
    """调用 vllm API"""
    base_url = f"http://{host}:{port}"
    
    for attempt in range(max_retries):
        try:
            url = f"{base_url}/v1/chat/completions"
            payload = {
                "model": model_path,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": INFERENCE_PARAMS["temperature"],
                "top_p": INFERENCE_PARAMS["top_p"],
                "max_tokens": INFERENCE_PARAMS["max_tokens"],
                "stop": INFERENCE_PARAMS["stop"],
            }
            
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip(), None
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return f"ERROR: {error_msg}", "E003"
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return "ERROR: Timeout", "E001"
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return "ERROR: Connection failed", "E002"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return f"ERROR: {str(e)}", "E004"
    
    return "ERROR: Max retries exceeded", "E004"


def extract_verb(response: str, expected_verbs: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """从响应中提取动词"""
    response = response.strip().lower()
    
    # 去除 thinking 标签
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    
    # 直接匹配
    for verb in expected_verbs:
        if response == verb.lower():
            return verb, None
    
    # 模糊匹配
    for verb in expected_verbs:
        if verb.lower() in response:
            return verb, None
    
    return None, "PARSE_ERROR"


def run_single_experiment(prompt: str, expected_verb: str, model_path: str, 
                         host: str, port: int, language: str) -> Dict:
    """运行单次实验"""
    expected_verbs = ZH_VERBS if language == "zh" else EN_VERBS
    
    start_time = time.time()
    raw_response, api_error = call_vllm(prompt, model_path, host, port)
    end_time = time.time()
    
    parsed_verb, parse_error = extract_verb(raw_response, expected_verbs)
    
    is_correct = parsed_verb == expected_verb if parsed_verb else False
    is_valid = is_correct and api_error is None
    
    error_type = api_error or parse_error
    
    return {
        "expected_verb": expected_verb,
        "parsed_verb": parsed_verb,
        "is_correct": is_correct,
        "is_valid": is_valid,
        "error_type": error_type,
        "raw_response": raw_response,
        "duration_seconds": round(end_time - start_time, 2),
        "timestamp": datetime.now().isoformat(),
    }


def run_task6_experiment(model_path: str, model_name: str, language: str,
                         output_dir: str, host: str = "localhost", port: int = 8000,
                         max_retries: int = 10) -> Dict:
    """运行 task6 实验"""
    try:
        prompts = load_prompts(model_name)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return {"correct": 0, "total": 0, "accuracy": 0}
    
    lang_prompts = prompts[language]
    expected_verbs = ZH_VERBS if language == "zh" else EN_VERBS
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for verb in expected_verbs:
        if verb not in lang_prompts:
            print(f"  跳过 {verb}: 无对应提示词")
            continue
            
        prompt = lang_prompts[verb]
        
        print(f"  运行 {verb}...")
        
        for attempt in range(1, max_retries + 1):
            result = run_single_experiment(prompt, verb, model_path, host, port, language)
            
            if result["is_valid"]:
                print(f"    {verb}: ✓ 正确 (尝试 {attempt})")
                break
            else:
                print(f"    {verb}: ✗ 错误 (尝试 {attempt}, 预期={verb}, 实际={result['parsed_verb']})")
        
        results.append(result)
    
    # 保存结果
    for result in results:
        filename = f"{model_name}_{result['expected_verb']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        output = {
            "task_id": "task6",
            "task_name": "可逆性2",
            "model": model_name,
            "model_path": model_path,
            "language": language,
            "expected_verb": result["expected_verb"],
            "parsed_verb": result["parsed_verb"],
            "is_correct": result["is_correct"],
            "is_valid": result["is_valid"],
            "error_type": result["error_type"],
            "raw_response": result["raw_response"],
            "duration_seconds": result["duration_seconds"],
            "timestamp": result["timestamp"],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    
    # 统计
    correct_count = sum(1 for r in results if r["is_correct"])
    total_count = len(results)
    
    return {
        "correct": correct_count,
        "total": total_count,
        "accuracy": round(correct_count / total_count * 100, 1) if total_count > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="task6 可逆性2实验")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--model-name", required=True, help="模型名称")
    parser.add_argument("--language", choices=["zh", "en"], required=True, help="语言")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--host", default="localhost", help="vllm 地址")
    parser.add_argument("--port", type=int, default=8000, help="vllm 端口")
    parser.add_argument("--max-retries", type=int, default=10, help="最大重试次数")
    
    args = parser.parse_args()
    
    print(f"运行 task6 {args.language} 实验...")
    print(f"模型: {args.model_name}")
    
    stats = run_task6_experiment(
        model_path=args.model,
        model_name=args.model_name,
        language=args.language,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        max_retries=args.max_retries,
    )
    
    print(f"\n结果: {stats['correct']}/{stats['total']} 正确, 准确率={stats['accuracy']}%")


if __name__ == "__main__":
    main()
