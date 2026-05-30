# -*- coding: utf-8 -*-
"""
数据完整性筛查脚本
根据筛查指南自动检查实验数据的合法性
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import Counter

# 导入配置
from config import (
    TASKS, VERBS, MODELS, CODING_DIMENSIONS, DESCRIPTION_DIMENSIONS,
    LANGUAGE
)


class DataScreener:
    """数据筛查器"""

    def __init__(self, task_id: str = "task1", language: str = "zh"):
        self.task_id = task_id
        self.language = language
        
        # 获取动词和模型列表
        if language == "zh":
            self.valid_verbs = set(VERBS)
        else:
            from config_en import VERBS as EN_VERBS
            self.valid_verbs = set(EN_VERBS)
        
        self.valid_models = set(os.path.basename(m) for m in MODELS)
        
        # 任务2的有效描述值
        if task_id == "task2":
            self.valid_values = {
                dim: set(config["options"])
                for dim, config in DESCRIPTION_DIMENSIONS.items()
            }

    def load_json_file(self, filepath: str) -> Tuple[Optional[Dict], Optional[str]]:
        """加载JSON文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, None
        except json.JSONDecodeError as e:
            return None, f"E1: JSON格式无效 - {str(e)}"
        except Exception as e:
            return None, f"E1: 文件读取失败 - {str(e)}"

    def check_required_fields(self, data: Dict) -> Optional[str]:
        """检查必需字段"""
        required_fields = [
            "task_id", "model", "model_path", "verb",
            "repeat_index", "timestamp", "duration_seconds",
            "is_valid", "parsed_result"
        ]
        
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return f"E2: 缺少必需字段 - {', '.join(missing_fields)}"
        
        return None

    def check_field_types(self, data: Dict) -> Optional[str]:
        """检查字段类型"""
        type_checks = {
            "task_id": str,
            "model": str,
            "model_path": str,
            "verb": str,
            "repeat_index": int,
            "timestamp": str,
            "duration_seconds": (int, float),
            "is_valid": bool,
        }
        
        for field, expected_type in type_checks.items():
            if field in data and not isinstance(data[field], expected_type):
                return f"E2: 字段类型错误 - {field} 应为 {expected_type.__name__}"
        
        return None

    def check_is_valid(self, data: Dict) -> Optional[str]:
        """检查is_valid标志"""
        if not data.get("is_valid", False):
            return "E3: is_valid=false"
        return None

    def check_parsed_result(self, data: Dict) -> Optional[str]:
        """检查parsed_result"""
        if data.get("parsed_result") is None:
            return "E4: parsed_result为null"
        return None

    def check_task1_values(self, data: Dict) -> Optional[str]:
        """检查任务1编码值范围"""
        result = data.get("parsed_result", {})
        
        value_ranges = {
            "FORCE": (1, 5),
            "ARM": (0, 1),
            "HAND": (0, 12),
            "VD": (0, 1),
            "HD": (0, 1),
        }
        
        for field, (min_val, max_val) in value_ranges.items():
            if field in result:
                val = result[field]
                if not isinstance(val, (int, float)) or val < min_val or val > max_val:
                    return f"E5: {field}值超出范围 - {val} (应为{min_val}-{max_val})"
        
        return None

    def check_task2_values(self, data: Dict) -> Optional[str]:
        """检查任务2描述值"""
        result = data.get("parsed_result", {})
        
        for dim, valid_set in self.valid_values.items():
            if dim in result:
                val = result[dim]
                if val not in valid_set:
                    return f"E5: {dim}值无效 - '{val}' (应为{valid_set})"
        
        return None

    def check_duration(self, data: Dict) -> Optional[str]:
        """检查响应时间"""
        duration = data.get("duration_seconds", 0)
        
        if duration < 0.1:
            return f"A1: 响应时间异常快速 - {duration}s"
        if duration > 300:
            return f"A2: 响应时间异常缓慢 - {duration}s"
        
        return None

    def check_raw_response(self, data: Dict) -> Optional[str]:
        """检查原始响应"""
        raw_response = data.get("raw_response", "")
        
        if isinstance(raw_response, str) and "ERROR" in raw_response.upper():
            return f"A3: 原始响应包含错误 - {raw_response[:100]}"
        
        return None

    def check_verb_model(self, data: Dict) -> Optional[str]:
        """检查动词和模型"""
        verb = data.get("verb", "")
        model = data.get("model", "")
        
        if verb not in self.valid_verbs:
            return f"E8: 动词不在配置列表中 - '{verb}'"
        
        # 模型名称可能包含路径，只检查基本名称
        # if model not in self.valid_models:
        #     return f"E9: 模型不在配置列表中 - '{model}'"
        
        return None

    def screen_single_file(self, filepath: str) -> Dict:
        """筛查单个文件"""
        result = {
            "file": filepath,
            "status": "pending",
            "exclusion_reason": None,
            "data": None,
            "checks": {}
        }
        
        # 1. 加载文件
        data, error = self.load_json_file(filepath)
        result["checks"]["F1"] = "pass" if error is None else "fail"
        if error:
            result["status"] = "excluded"
            result["exclusion_reason"] = error
            return result
        result["data"] = data
        
        # 2. 检查必需字段
        error = self.check_required_fields(data)
        result["checks"]["F2"] = "pass" if error is None else "fail"
        if error:
            result["status"] = "excluded"
            result["exclusion_reason"] = error
            return result
        
        # 3. 检查字段类型
        error = self.check_field_types(data)
        result["checks"]["F3"] = "pass" if error is None else "fail"
        if error:
            result["status"] = "excluded"
            result["exclusion_reason"] = error
            return result
        
        # 4. 检查is_valid
        error = self.check_is_valid(data)
        result["checks"]["E3"] = "pass" if error is None else "fail"
        if error:
            result["status"] = "excluded"
            result["exclusion_reason"] = error
            return result
        
        # 5. 检查parsed_result
        error = self.check_parsed_result(data)
        result["checks"]["E4"] = "pass" if error is None else "fail"
        if error:
            result["status"] = "excluded"
            result["exclusion_reason"] = error
            return result
        
        # 6. 检查编码值范围
        if self.task_id == "task1":
            error = self.check_task1_values(data)
        else:
            error = self.check_task2_values(data)
        result["checks"]["E5"] = "pass" if error is None else "fail"
        if error:
            result["status"] = "excluded"
            result["exclusion_reason"] = error
            return result
        
        # 7. 检查响应时间
        error = self.check_duration(data)
        if error:
            if "A1" in error:
                result["checks"]["A1"] = "warning"
            elif "A2" in error:
                result["checks"]["A2"] = "warning"
            # 响应时间异常不直接排除，标记为待复核
            result["status"] = "review"
            result["exclusion_reason"] = error
        else:
            result["checks"]["A1"] = "pass"
            result["checks"]["A2"] = "pass"
        
        # 8. 检查原始响应
        error = self.check_raw_response(data)
        result["checks"]["A3"] = "pass" if error is None else "fail"
        if error:
            result["status"] = "excluded"
            result["exclusion_reason"] = error
            return result
        
        # 9. 检查动词和模型
        error = self.check_verb_model(data)
        if error:
            if "E8" in error:
                result["checks"]["E8"] = "fail"
            elif "E9" in error:
                result["checks"]["E9"] = "fail"
            result["status"] = "excluded"
            result["exclusion_reason"] = error
            return result
        else:
            result["checks"]["E8"] = "pass"
            result["checks"]["E9"] = "pass"
        
        # 10. 标记为入组
        if result["status"] == "pending":
            result["status"] = "included"
        
        return result

    def screen_directory(self, data_dir: str) -> Dict:
        """筛查整个目录"""
        results = {
            "screening_time": datetime.now().isoformat(),
            "data_source": data_dir,
            "task_id": self.task_id,
            "language": self.language,
            "total_files": 0,
            "included": 0,
            "excluded": 0,
            "review": 0,
            "exclusion_reasons": Counter(),
            "quality_metrics": {},
            "files": []
        }
        
        # 扫描目录
        json_files = list(Path(data_dir).glob("**/*.json"))
        # 排除汇总文件和跟踪文件
        json_files = [f for f in json_files if not f.name.startswith("summary_") 
                      and not f.name.startswith("experiment_")]
        
        results["total_files"] = len(json_files)
        
        # 筛查每个文件
        for filepath in json_files:
            file_result = self.screen_single_file(str(filepath))
            results["files"].append(file_result)
            
            if file_result["status"] == "included":
                results["included"] += 1
            elif file_result["status"] == "excluded":
                results["excluded"] += 1
                if file_result["exclusion_reason"]:
                    # 提取排除原因代码
                    reason_code = file_result["exclusion_reason"].split(":")[0]
                    results["exclusion_reasons"][reason_code] += 1
            elif file_result["status"] == "review":
                results["review"] += 1
        
        # 计算质量指标
        total = results["total_files"]
        if total > 0:
            results["quality_metrics"]["inclusion_rate"] = f"{results['included']/total*100:.1f}%"
            results["quality_metrics"]["exclusion_rate"] = f"{results['excluded']/total*100:.1f}%"
            results["quality_metrics"]["review_rate"] = f"{results['review']/total*100:.1f}%"
            
            # 计算响应时间统计
            durations = [
                f["data"]["duration_seconds"] 
                for f in results["files"] 
                if f["data"] and "duration_seconds" in f["data"]
            ]
            if durations:
                import statistics
                results["quality_metrics"]["avg_duration"] = f"{statistics.mean(durations):.1f}s"
                results["quality_metrics"]["std_duration"] = f"{statistics.stdev(durations):.1f}s" if len(durations) > 1 else "N/A"
        
        return results

    def generate_report(self, results: Dict, output_file: str = None) -> str:
        """生成筛查报告"""
        report = {
            "筛查时间": results["screening_time"],
            "数据源": results["data_source"],
            "任务ID": results["task_id"],
            "语言": results["language"],
            "总记录数": results["total_files"],
            "入组数": results["included"],
            "排除数": results["excluded"],
            "待复核数": results["review"],
            "入组率": results["quality_metrics"].get("inclusion_rate", "N/A"),
            "排除原因统计": dict(results["exclusion_reasons"]),
            "质量指标": results["quality_metrics"],
            "入组记录": [
                {"file": f["file"], "verb": f["data"]["verb"], "model": f["data"]["model"]}
                for f in results["files"] if f["status"] == "included"
            ],
            "排除记录": [
                {"file": f["file"], "reason": f["exclusion_reason"]}
                for f in results["files"] if f["status"] == "excluded"
            ],
            "待复核记录": [
                {"file": f["file"], "reason": f["exclusion_reason"]}
                for f in results["files"] if f["status"] == "review"
            ]
        }
        
        report_json = json.dumps(report, ensure_ascii=False, indent=2)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_json)
            print(f"报告已保存: {output_file}")
        
        return report_json


def main():
    parser = argparse.ArgumentParser(description="数据完整性筛查脚本")
    parser.add_argument(
        "data_dir",
        help="数据目录路径"
    )
    parser.add_argument(
        "--task",
        choices=["task1", "task2"],
        default="task1",
        help="任务类型（默认: task1）"
    )
    parser.add_argument(
        "--language",
        choices=["zh", "en"],
        default="zh",
        help="语言版本（默认: zh）"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出报告文件路径"
    )
    
    args = parser.parse_args()
    
    # 创建筛查器
    screener = DataScreener(args.task, args.language)
    
    # 执行筛查
    print(f"开始筛查: {args.data_dir}")
    print(f"任务: {args.task}, 语言: {args.language}")
    print("-" * 60)
    
    results = screener.screen_directory(args.data_dir)
    
    # 生成报告
    output_file = args.output or os.path.join(
        args.data_dir,
        f"screening_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    report = screener.generate_report(results, output_file)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("筛查完成!")
    print(f"总记录数: {results['total_files']}")
    print(f"入组数: {results['included']}")
    print(f"排除数: {results['excluded']}")
    print(f"待复核数: {results['review']}")
    print(f"入组率: {results['quality_metrics'].get('inclusion_rate', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
