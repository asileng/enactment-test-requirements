# -*- coding: utf-8 -*-
"""
生成 task6（可逆性2）提示词
使用各模型自己的task1参数化结果作为基准值
"""

import json
import os
from collections import defaultdict

def calculate_model_averages():
    """计算各模型对每个动词的五维度平均值"""
    model_verb_scores = defaultdict(lambda: defaultdict(lambda: {"FORCE": [], "HAND": [], "ARM": [], "VD": [], "HD": []}))
    
    base_dir = "pilot_results"
    for model_dir in os.listdir(base_dir):
        task1_en_dir = os.path.join(base_dir, model_dir, "task1_en")
        if not os.path.isdir(task1_en_dir):
            continue
        
        for filename in os.listdir(task1_en_dir):
            if filename.startswith("summary_") or filename.startswith("experiment_"):
                continue
            filepath = os.path.join(task1_en_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                model = data.get("model", model_dir)
                verb = data.get("verb", "")
                parsed = data.get("parsed_result", {})
                
                if parsed and verb:
                    for dim in ["FORCE", "HAND", "ARM", "VD", "HD"]:
                        if dim in parsed:
                            model_verb_scores[model][verb][dim].append(parsed[dim])
            except:
                pass
    
    # 计算平均值
    model_averages = {}
    for model in model_verb_scores:
        model_averages[model] = {}
        for verb in model_verb_scores[model]:
            scores = model_verb_scores[model][verb]
            avg = {}
            for dim in ["FORCE", "HAND", "ARM", "VD", "HD"]:
                vals = scores[dim]
                avg[dim] = round(sum(vals)/len(vals), 2) if vals else 0
            model_averages[model][verb] = avg
    
    return model_averages


def generate_task6_prompt_en(verb, model_avg):
    """生成英文task6提示词"""
    prompt = f"""Please make a simple judgment based on the native English speaker's understanding of this action, assuming you are a right-handed native English speaker.

Imagine you are in an empty room.

In front of you is a table.

On the table surface is placed a palm-sized object.

An action X can be represented using five structured behavioral dimensions:

FORCE: Describes the magnitude of force applied to an object during the action.
5 = very strong
4 = strong
3 = moderate
2 = weak
1 = very weak

ARM: Initial state of the arm before executing the action.
1 = arm extended
0 = arm bent

HAND: Initial vertical height of the hand before action execution.
0 = ground level
1–9 = relative height between ground level and the actor's body height
10 = at body height
11 = one unit above body height
12 = two units above body height

VD: Primary vertical motion direction of the hand during the action.
1 = downward
0 = upward

HD: Primary horizontal motion direction of the hand during the action.
1 = forward
0 = sideways

Your previous interpretation of action X was:

FORCE mean: {model_avg['FORCE']}
HAND mean: {model_avg['HAND']}
ARM mean: {model_avg['ARM']}
VD mean: {model_avg['VD']}
HD mean: {model_avg['HD']}

Please select the most appropriate verb that best matches the described action from the following candidate set:

throw
toss
fling
cast
chuck
hurl

Output exactly one verb only.
The verb must be one of the six candidates listed above.
Do not provide any explanation or additional text.
/no_think"""
    return prompt


def generate_task6_prompt_zh(verb, model_avg):
    """生成中文task6提示词"""
    prompt = f"""请按照中文母语者对该动作的相关理解进行简单判断，假设你是一个右利手的中文母语者。

想象你处于一个空旷房间中。

你的面前有一张桌子。

桌面上放置着一个手掌大小的物体。

已知，动作X可以按照以下五个维度标准被记录：

FORCE：描述动作对于物体施加力的大小
5 = 非常强
4 = 强
3 = 中等
2 = 弱
1 = 非常弱

ARM：动作开始前手臂的初始状态
1 = 手臂伸直
0 = 手臂弯曲

HAND：动作开始前手部停留的初始高度
0 = 地面高度
1–9 = 地面高度与执行者身高之间的相对高度
10 = 执行者身高
11 = 高于执行者身高一个单位
12 = 高于执行者身高两个单位

VD：执行动作时手部的主要垂直运动方向
1 = 向下
0 = 向上

HD：执行动作时手部的主要水平运动方向
1 = 向前
0 = 向侧方

你之前对动作X的综合平均演绎为：

FORCE平均得分：{model_avg['FORCE']}
HAND平均得分：{model_avg['HAND']}
ARM平均得分：{model_avg['ARM']}
VD平均得分：{model_avg['VD']}
HD平均得分：{model_avg['HD']}

请从以下候选动词中选择最符合该动作的一个：

扔
抛
甩
投
摔
丢

仅输出一个动词，这个动词必须是以上六个候选动词中的一个。
不要输出任何其他内容，不要解释你的选择。
/no_think"""
    return prompt


def main():
    print("计算各模型动词参数化平均值...")
    model_averages = calculate_model_averages()
    
    # 为每个模型生成task6提示词
    for model_name, verb_avgs in model_averages.items():
        task6_prompts = {"zh": {}, "en": {}}
        
        en_verbs = ["throw", "toss", "fling", "cast", "chuck", "hurl"]
        zh_verbs = ["扔", "丢", "抛", "投", "摔", "甩"]
        
        for en_verb, zh_verb in zip(en_verbs, zh_verbs):
            if en_verb in verb_avgs:
                avg = verb_avgs[en_verb]
                task6_prompts["en"][en_verb] = generate_task6_prompt_en(en_verb, avg)
                task6_prompts["zh"][zh_verb] = generate_task6_prompt_zh(zh_verb, avg)
        
        # 保存到文件
        output_file = f"task6_prompts_{model_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(task6_prompts, f, ensure_ascii=False, indent=2)
        
        print(f"已生成: {output_file}")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
