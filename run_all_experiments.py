# -*- coding: utf-8 -*-
"""
大模型测评任务母脚本
管理整个实验流程，支持批量运行多个任务、多个语言版本
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict
from pathlib import Path

# 导入配置
from config import (
    MODELS, TASKS, LANGUAGE, RESULTS_DIR, VLLM_CONFIG,
    REPEAT_COUNT, USE_LATIN_SQUARE, LATIN_SQUARE_ORDERS
)


class ExperimentManager:
    """实验管理器"""

    def __init__(self, output_base_dir: str = None):
        self.output_base_dir = output_base_dir or "experiment_results"
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}

    def run_single_task(
        self,
        task_id: str,
        language: str,
        models: List[str] = None,
        participant_id: int = None,
        repeat: int = None,
        use_chat: bool = True,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        resume: bool = True
    ) -> Dict:
        """运行单个任务"""
        models = models or MODELS
        repeat = repeat or REPEAT_COUNT

        # 创建输出目录
        output_dir = os.path.join(
            self.output_base_dir,
            f"{task_id}_{language}_{self.experiment_id}"
        )

        # 构建命令
        cmd = [
            sys.executable, "run_experiment.py",
            "--task", task_id,
            "--language", language,
            "--output-dir", output_dir,
            "--repeat", str(repeat),
            "--max-retries", str(max_retries),
            "--retry-delay", str(retry_delay),
        ]

        if models:
            cmd.extend(["--models"] + models)

        if participant_id is not None:
            cmd.extend(["--participant-id", str(participant_id)])

        if not use_chat:
            cmd.append("--no-chat")

        if not resume:
            cmd.append("--no-resume")

        print(f"\n{'='*60}")
        print(f"运行任务: {task_id}")
        print(f"语言: {language}")
        print(f"输出目录: {output_dir}")
        print(f"命令: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        # 执行命令
        start_time = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                check=True
            )
            success = True
            error_msg = None
        except subprocess.CalledProcessError as e:
            success = False
            error_msg = str(e)
        except Exception as e:
            success = False
            error_msg = str(e)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        task_result = {
            "task_id": task_id,
            "language": language,
            "output_dir": output_dir,
            "success": success,
            "error_msg": error_msg,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
        }

        self.results[f"{task_id}_{language}"] = task_result

        if success:
            print(f"\n✓ 任务 {task_id} ({language}) 完成，耗时 {duration:.1f}s")
        else:
            print(f"\n✗ 任务 {task_id} ({language}) 失败: {error_msg}")

        return task_result

    def run_all_tasks(
        self,
        languages: List[str] = None,
        task_ids: List[str] = None,
        models: List[str] = None,
        participant_id: int = None,
        repeat: int = None,
        use_chat: bool = True,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        resume: bool = True
    ) -> Dict:
        """运行所有任务"""
        languages = languages or [LANGUAGE]
        task_ids = task_ids or list(TASKS.keys())

        print(f"\n{'='*60}")
        print(f"开始批量实验")
        print(f"实验ID: {self.experiment_id}")
        print(f"任务数: {len(task_ids)}")
        print(f"语言数: {len(languages)}")
        print(f"总组合数: {len(task_ids) * len(languages)}")
        print(f"{'='*60}\n")

        for task_id in task_ids:
            for language in languages:
                self.run_single_task(
                    task_id=task_id,
                    language=language,
                    models=models,
                    participant_id=participant_id,
                    repeat=repeat,
                    use_chat=use_chat,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    resume=resume
                )

        # 保存汇总
        self.save_summary()

        return self.results

    def run_latin_square_experiment(
        self,
        task_id: str,
        language: str,
        models: List[str] = None,
        repeat: int = None,
        use_chat: bool = True,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        resume: bool = True
    ) -> Dict:
        """运行拉丁方实验（所有参与者ID）"""
        models = models or MODELS
        repeat = repeat or REPEAT_COUNT

        # 获取拉丁方顺序数量
        if language == "zh":
            from config import LATIN_SQUARE_ORDERS
        else:
            from config_en import LATIN_SQUARE_ORDERS

        num_orders = len(LATIN_SQUARE_ORDERS)

        print(f"\n{'='*60}")
        print(f"开始拉丁方实验")
        print(f"任务: {task_id}")
        print(f"语言: {language}")
        print(f"拉丁方顺序数: {num_orders}")
        print(f"{'='*60}\n")

        for participant_id in range(num_orders):
            print(f"\n--- 参与者ID: {participant_id} ---")
            self.run_single_task(
                task_id=task_id,
                language=language,
                models=models,
                participant_id=participant_id,
                repeat=repeat,
                use_chat=use_chat,
                max_retries=max_retries,
                retry_delay=retry_delay,
                resume=resume
            )

        # 保存汇总
        self.save_summary()

        return self.results

    def save_summary(self):
        """保存实验汇总"""
        os.makedirs(self.output_base_dir, exist_ok=True)

        summary_file = os.path.join(
            self.output_base_dir,
            f"experiment_summary_{self.experiment_id}.json"
        )

        summary = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(self.results),
            "successful_tasks": sum(1 for r in self.results.values() if r["success"]),
            "failed_tasks": sum(1 for r in self.results.values() if not r["success"]),
            "results": self.results
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n实验汇总已保存: {summary_file}")

        return summary_file


def main():
    parser = argparse.ArgumentParser(description="大模型测评任务母脚本")
    parser.add_argument(
        "--mode",
        choices=["single", "all", "latin-square"],
        default="single",
        help="运行模式: single=单个任务, all=所有任务, latin-square=拉丁方实验"
    )
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()),
        default=None,
        help="任务ID（single和latin-square模式必需）"
    )
    parser.add_argument(
        "--language",
        choices=["zh", "en"],
        default=LANGUAGE,
        help=f"语言版本（默认: {LANGUAGE}）"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=["zh", "en"],
        default=None,
        help="语言版本列表（all模式使用）"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="模型路径列表"
    )
    parser.add_argument(
        "--output-dir",
        default="experiment_results",
        help="输出基础目录（默认: experiment_results）"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=REPEAT_COUNT,
        help=f"重复次数（默认: {REPEAT_COUNT}）"
    )
    parser.add_argument(
        "--participant-id",
        type=int,
        default=None,
        help="参与者ID（用于拉丁方顺序分配）"
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="使用Completion API格式"
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

    # 验证参数
    if args.mode in ["single", "latin-square"] and args.task is None:
        parser.error(f"--mode {args.mode} 需要指定 --task 参数")

    # 创建实验管理器
    manager = ExperimentManager(args.output_dir)

    # 根据模式运行
    if args.mode == "single":
        manager.run_single_task(
            task_id=args.task,
            language=args.language,
            models=args.models,
            participant_id=args.participant_id,
            repeat=args.repeat,
            use_chat=not args.no_chat,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            resume=not args.no_resume
        )
    elif args.mode == "all":
        languages = args.languages or [args.language]
        manager.run_all_tasks(
            languages=languages,
            models=args.models,
            participant_id=args.participant_id,
            repeat=args.repeat,
            use_chat=not args.no_chat,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            resume=not args.no_resume
        )
    elif args.mode == "latin-square":
        manager.run_latin_square_experiment(
            task_id=args.task,
            language=args.language,
            models=args.models,
            repeat=args.repeat,
            use_chat=not args.no_chat,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            resume=not args.no_resume
        )


if __name__ == "__main__":
    main()
