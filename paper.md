# 认知语义学视角下具身智能基座模型的“具身性”潜能初探

**熊羿成** 上海外国语大学

**李檬希** 西南大学

**关键词：** 大语言模型 视觉语言模型 具身智能 认知语言学 人工智能测评

**认知语义学视角下具身智能基座模型的“具身性”潜能初探**

熊羿成 上海外国语大学

李檬希 西南大学

关键词：大语言模型 视觉语言模型 具身智能 认知语言学 人工智能测评


具身智能（Embodied AI）正处于从语义规划向端到端动作生成的范式跃迁中，然而，现有视觉-语言-动作（VLA）模型虽在轨迹模仿上表现卓越，却因缺失本体感受与物理反馈而面临严重的认知断层。本文基于认知语言学理论，提出了一种旨在探测大模型**“潜具身性”（Latent Embodiment）的新型评测框架。本研究利用身体动作动词（PA-verbs）的语义分解模型，将动词解构为力度、肢体姿态、空间高度及运动轨迹五个物理维度，并在多种主流大语言模型（LLM）与多模态模型（VLM）上复现了心理语言学中的“表演范式”（Enactment Paradigm）实验。通过对 Qwen、GLM等模型的跨语言（中英双语）、跨模态及跨视角对比实验，本研究发现：第一，VLA 基座模型展现出了一定程度的潜具身认知，能够自发将动词映射为物理参数，但该能力高度依赖于视觉显著性（Visual Saliency），在处理精细化动作语义时表现出显著的物理直觉匮乏。第二，视觉模态的引入对动作认知的提升具有非均匀性，在优化宏观空间感知的过程中可能引入统计噪声，削弱模型对高频泛化动词的辨析力。

## 一、引言

具身智能系统被认为是人类通往通用人工智能（AGI）的最前沿路径之一。根据 Pfeifer 与 Bongard（2006）的定义，具身智能是智能体通过物理实体与其所处真实环境进行持续交互而涌现出的智能。其中，视觉-语言-动作模型（Vision-Language-Action models，VLA）是近年来最具代表性的研究方向，被认为为是通向通用机器人的有效、稳定策略（Li et al.,2026）。VLA模型这一概念最早由在RT-2的研究中提出，指的是以VLM或LLM为模型基座，利用人类遥操作产生运动参数进行模仿学习的训练路径。此后，又出现了早融合，扩散动作流等技术演进（Black et al.,2024），还有部分模型引入触觉参数（Feng et al.,2025,Yang, Feng et al.,2024），但其训练核心机制保持不变。

尽管模仿学习路径在动作执行成功率和准确率上取得了进步，但是其是否真正具备物理理解和具身认知饱受学界争议。有批评认为此种人类运动轨迹的统计拟合，缺乏本体感受、力学反馈、动态推理等功能，并非对物理运动的底层理解（Ross et al., 2011; Malik,2023），仅仅是“统计鹦鹉”。同时，许多VLA模型的评估依赖大语言模型作为裁判，有陷入递归污染的风险（Shumailov et al., 2024; Panickssery et al., 2024），此时测评所得到的成果仅能反应语义连贯性，而非理解真实性。（Bender，Koller，2020）大量的测评研究也表明，当前的具身模型在物理认知、常识推理等具身理解领域存在较明显不足，并有学者指出VLA模型在顶层抽象规划和低级运动控制的表现之间存在断层，认为当前的训练过分强调了语言理解，但是语言理解和物理执行之间存在鸿沟。

本研究同样认同当前针对的具身模型的语言理解训练可能无法帮其获得具身性；但是这并不能说明语言理解本身无法帮助模型理解物理执行，而是说明当前语言训练任务并没有考虑到语言本身的具身性特征。

通过对先前的18篇VLA测评研究进行了总结（表1），研究者发现对语言理解进行测评的理解仅有5篇，且5篇都是针对物体识别、空间理解、任务分解等抽象规划任务的测量，缺乏对语言形式、语义等语言学变量的观测。换言之，在VLA现在的测量标准中，语言仅仅是下达命令的固定符号和VLA模型内部通信与输出规划的一种方式而已。这样的语言关既不符合认知科学和语言学的实证结论，也不符合人机交互的现实场景。

认知语言学强调，人类的认知本身并非抽象的符号运算，而扎根于人类的运动感官系统和本体感觉（Lakoff & Johnson, 1980; Varela, Thompson & Rosch, 1991）。根据语言学和神经科学的实证结论，具身性对语言表征存在影响，语言表征内编码了具身性。反而，从训练材料的角度而言，VLA模型的加强训练数据中不包括本体感觉，因而其无法在加强训练中产生原生的具身性。换言之，VLA模型的具身特征不会出现在“A”中；“V”和“L”的基座模型则有可能提取从海量的人类语言和视觉材料中提取出具身性的代理表征，并涌现出理解具身性的能力，就像大语言模型涌现出语言学推理能力一般。作者在此将其定义为“潜具身性”或“具身性潜能”。

为了进一步检验语言具身性和具身智能基座模型的可能联系，本研究特别选择了动词语义理解这一角度开展实验。具体而言，研究人员借用认知语义学对身体动作动词（PAVerbs）的分析框架和“表演范式”（Enactment paradigm）将人类对动作动词具身化的“心理表征”参数化为符号表征（Talmy，Thompson，Gao），并使用提示词面向“大模型被试”复现并拓展了现有的行为实验（Gao&Wang，2012，2016）并与人类基准数据进行对比。同时，本研究控制了基座模型类型，语言环境和输出要求等变量，以讨论各个变量和技术路线对大模型具身认知的潜在影响。此外，本研究通过双向的模型设计深入考察了大语言模型动词具身理解的“可逆化性”（Mahajan et al., 2025等人），探索其理解深度。

对动词-动作语义理解中的“具身性潜能”验证格外重要，因为这关乎人类和模型在动作语义的理解上是否存在认知鸿沟，也弥补了现行人机交互和安全性研究的空白。

根据Long等人（2025，p5）提出的五级的机器人智能等级框架，“是否具备人类语言的基本理解能力”是区分基础服务机器人与通用服务机器人的重要判定标准；而因此，如果具身智能模型无法理解动词的具身性，其也就难以真正理解人类语言，这可能会严重影响VLA模型人机交互场景下的表现，从而影响其工作效率和安全性问题。

已有研究指出，交互式任务中对自然语言指令的错误理解是主要的物理安全隐患（Xing et.al 2025）；在部分对安全性和人机交互有着高要求的场合，这样的问题会进一步被放大。例如老年人陪护场景下，“搀”和“扶”同近义词动词区分错误可能导致安全后果，但这两个动作的轨迹表征基本一致。当前具身性的匮乏，本质上是细粒度动作语义的缺失，并会导致人机交互的失败。

针对具身智能系统的安全性问题，学界现主要关注动作执行与安全性预测（Li et al., 2023）、安全原则对齐（（Gulcehre et al., 2023；Ouyang et al., 2022；Touvron et al., 2023；Dai et al., 2023；Ji et al., 2024a；Zhou et al., 2025a；Ji et al., 2024b；Meng et al., 2025；Zhou et al., 2025b；Chen et al., 2025）、抗攻击能力等角度（Inan et al., 2023；Chi et al., 2024）)，尚未关注到人机交互过程中的断层隐患。与此同时，时下对具身模型的训练与评测研究依赖于封闭的训练集和固定的指令，主要关注模型执行端的成功率、准确度和推理泛化能力，在人机交互方面留有空白。

综上所述，VLA模型基座的动作动词语义理解与其是否具备“潜具身性”是一个在技术前沿、技术落地和安全伦理上都亟待探讨的问题。具体而言，本研究希望探究的研究问题如下：

**RQ1:当前基于海量数据训练的VLA基座模型是否能预测，能在多大范围上预测人类在表演范式下针对身体动作动词表现出的结果？**

**RQ2:LLM与VLM在不同任务下的表现有何差异？**

**RQ3：VLA基座模型对身体动作动词的理解是否对输出格式和语言敏感？**

## 二、文献综述

### 2.1 认知语言学与具身认知

### 2.1.1 身体动作动词（PA verbs）与近同义词区分

身体动作动词（Physical action verbs，下称PA动词）是一类及物动词，专门描述人类施事通过有方向的身体运动对物理对象或环境施加力量的事件。与描述心理状态、抽象关系或位移结果的动词不同。PA动词的语义核心在于动作本身的运动学内容：力量大、运动速度、轨迹方、接触方式以及身体部位的配置状态。这些物理参数并非语境推断的附加成分，而是构成PA动词词汇意义的内在特征（Gao, 2001a; Gibbs, 2003）。

Gao（2001a）以汉语身体动作动词为对象，在Jackendoff（2002, 2007）词汇概念结构框架的基础上提出了PA 动词的语义分解体系，将其意义拆解为施事与受事属性、路径、方式、力量和意图结构等基本语义成分。这一框架揭示了PA动词语义的多维性：同一动作类别中不同动词之间的区分，往往不是通过单一参数，而是通过多个物理维度的组合来实现的。

本研究格外关注PA动词的近义词区分问题，因为在具身智能与人机交互领域，近义词区分直接关系到语言指令的执行精度。汉语投掷类动词群（扔、丢、抛、投、摔、甩）是典型的此类案例。（Gao, 2001a; Gao, Wang & Nicoladis, 2016; Wang & Gao, 2016）这些动词均可粗略对应英文的throw，但在力度、手部初始位置与姿态、以及运动方向上存在系统性区分——这些区分在语料库与词典条目中几乎不被显性命名，却在母语者的动作演示中高度一致且可量化地呈现。

Gao（2001a）所建立的这一语义分解体系不仅是一项针对汉语动词的实证研究，更构成了本研究的理论基础框架：它提供了一套将PA verbs词义操作化为可量化物理维度集合的分析语言，使动词语义的跨说话者比较、跨语言比较乃至跨系统比较（包括人类与计算模型之间的比较）在方法论上成为可能。后续各节所涉及的表演范式设计、编码体系建构以及基准测试框架，均直接建立于这一分解框架之上。

### 2.2 表演范式：链接语义、身体感知与动作描述

表演范式（enactment paradigm）由Gao与Wang（2012）在汉语母语者研究中首次系统性引入PA verbs语义研究领域，随后经Gao等人（2016）发展为完整的双实验验证体系。其基本程序是：将目标动作为任务要求给母语者，要求其以标准化物体演示该动词所指称的动作，并以多角度视频录制其演示过程；随后由经训练的编码员依据预设的物理维度体系对动作特征进行系统编码。在投掷类动词研究中，所采用的编码维度包括：力量（FORCE，五分制量表）、手部初始高度（HAND，以身高归一化）、手臂初始姿态（ARM，伸直或弯曲）、手部运动的纵向方向（VD，向上或向下）以及横向方向（HD，向前或侧向）。各维度的评分者间信度在独立研究中均达到Cohen's kappa = 0.71至0.74的水平（Gao et al., 2016; Wang & Gao, 2016）。

表演范式的理论合理性植根于认知神经科学关于动作动词语义加工的丰富证据。模态特异性语义理论（modality-specific semantics）认为，词语意义的表征并非储存于独立于感觉运动系统的抽象符号系统中，而是分布于与该词语所涉及的感知与运动通道相对应的神经基底之上（Barsalou, 1999）。Bergen等人（2010）的研究证明，理解动作动词会激活与执行相应动作相关的效应器特异性感觉运动皮层区域；Willems等人（2010）进一步证明，惯用手的差异会导致动作动词理解时前运动皮层激活的侧化模式发生相应变化，提供了动作动词语义加工具有身体特异性的强有力证据；Tranel等人（2003）则表明，大脑对动作概念的提取依赖于感觉和运动特征模式的联合激活，与动作执行共享额叶岛盖等关键脑区。

上述神经科学证据为表演范式提供了直接的理论支撑：若动作动词的理解在神经层面激活了执行该动作所需的感觉运动表征，那么母语者在接收到动词刺激后所产生的运动输出，就是其词汇表征内容的行为外显化——被演示的动作，将被词汇编码但在话语中通常处于沉默的物理参数，以可观察的方式呈现出来。

### 2.3 大型语言模型与具身智能

### 2.3.1 具身智能相关模型

2022年的SayCan (Ahn et al., 2022) 是最早通过分层规划架构实现LLM指挥机器人执行长任务落地的模型。通过将机器人的传感器数据和原始图像转化为token，在大型语言模型PaLM-540B基础上训练出的PaLM-E被视为最早的“具身多模态语言模型(Driess et al.,2023)”。Brohan等人（2023）在RT-2的研究中创新的提出了VLA模型范式，通过将机器人运动轨迹数据与视觉语言数据进行联合训练，实现了对高层视觉语言推理与低级运动控制的端到端整合，表现出超越训练分布的泛化能力。

近年来，领域内涌现出多条技术演进路径。在环境构建方面，Holodeck (Yang et al., 2024) 实现了基于自然语言的三维具身训练环境生成；在感知控制方面，MP5 (Qin et al., 2024) 构建了语言驱动的开放式具身系统。针对模型架构，腾讯发布的 HY-Embodied (2026) 采用混合变换器（Hybrid Transformer）架构，通过非共享参数设计解决了大规模视觉训练带来的语言能力衰减问题并细化了物理空间理解。在动作生成机制上，0 (Black et al., 2024) 引入流匹配（Flow Matching）技术建模连续动作分布，显著提升了复杂任务的执行精度。此外，AnyTouch (Feng et al., 2025) 与 UniTouch (Yang et al., 2024) 通过统一视觉-触觉表征学习，赋予了模型对材质硬度、粗糙度等物理属性的认知能力。

为了增强模型对像素级原始特征的感应，一部分原生态多模态图像跳过了CLIP等与训练编码器，将图片在输入端就“Token化”，直接进入 Transformer，代表模型包括GPT-40,Chameleon，Emu3等。尽管早融合架构显著提升了系统的感知精度。早融合架构带来的认知精度提升，虽然仍然无法使得模型获取物理经验，但可能能够帮助模型涌现出“潜具身化”的认知能力。

截止本文，现有的VLA模型仍然有在执行需要高度灵巧或精确度任务时表现较差，却无法应对陌生的物理环境。同时，其物理推理能力仍然欠缺，对动作后果。最后，当前的所有训练方式都局限于轨迹训练和静态触觉训练，当前的VLA模型仍然不具备顺应力感应物理感知能力和本体感觉。也因此，本文所需要测量的“潜具身性”，是讨论当前模型训练路径能否最终通向真正的动作语义理解的重要环节。

### 2.3.2 现行VLM与LLM语言能力测评批判

	随着深度学习范式的演进，学界对大模型底层逻辑的反思从未停止。Panickssery 等人 (2024) 指出，在模型测评中使用“LLM 作为裁判（LLM-as-a-Judge）”存在严重的递归污染风险，可能导致评测结果仅反映模型内部的逻辑连贯性，而非其输出的真实性。Bender & Koller (2020) 亦强调，若模型仅处理形式而缺乏与意义的经验结合，则难以产生真正的物理理解。Zhu 等人 (2020) 进一步批判了“大数据、小问题”的规模化范式，认为其使模型倾向于捕捉显性统计特征，而无法获取与人类同源的物理常识。上述质疑在具身智能领域依然适用：现有的测评基准同样具有类似的问题。

作者整理总结了2024-2025年间15篇具身智能相关测评研究（见表一），

**[表1]**

| 测试名称 | 意图理解 | 感知推理 | 动作语义分析 | 形式逻辑 | 视觉感知 | 任务拆分与重述 | 跨具身硬件 | 评价指标 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RoboVQA (2024) | ✓ | × | × | √ | × | √ | √ | BLEU |
| EgoPlanBench (2024a) | × | √ | × | × | ✓ | √ | × | 多选题 |
| MMRo (2024c) | × | √ | × | × | ✓ | √ | × | LLM打分 |
| EAI (2024d) | × | √ | × | × | ✓ | √ | × | 成功率 |
| OpenEQA (2024) | × | √ | × | × | ✓ | √ | × | LLM打分 |
| EgoThink (2024b) | × | √ | × | × | ✓ | √ | × | LLM打分 |
| VideoEgoThink (2024a) | × | √ | × | × | ✓ | √ | × | LLM打分 |
| EmbodiedEval (2025) | × | √ | × | × | ✓ | √ | × | 多选 |
| EmbodiedBench (2025) | × | √ | × | × | ✓ | √ | √ | 正确率 |
| VLABench (2024a) | ✓ | √ | × | × | ✓ | √ | √ | 正确率 |
| Robobench（2025） | ✓ | √ | × | ✓ | ✓ | ✓ | ✓ | 模拟器 |
| MMSI-Bench（2025） | × | √ | × | ✓ | √ | √ | × | 模拟器 |
| ROBOMIND（2024） | × | √ | × | ✓ | ✓ | √ | ✓ | 正确率 |
| EmbSpatial-Bench（2024） | × | √ | × | ✓ |  | √ | × | 多选题 |

并对其考察的语言理解相关能力进行了整理，发现16个测评中，仅有三个测评研究考量了大语言模型对于指令的理解能力，其余模型都是将行动与目标作为明确标出的信息输入进提示词，现有测评主要关注任务规划、感官推理，逻辑判断，但是没有任何测评研究过具身智能模型对语言学的动作语义分析能力。即，我们仅能观测模型能否理解基本行动目标以及其能否在内部信息沟通过程中保证语言表征一致，却不知道模型能否将这种表征与动作进行准确的匹配，无法测量模型对动作的概念理解，而概念理解又根本的参与构建了具身性.

	在仅有的几个考虑指令理解的测评中，RoboVQA（2023）最早关注到了高层推理和语义理解在具身只能评测中的重要性。通过将视觉问答形式引入机器人操作，它不仅仅考察机器人执行的正确率，同时要求机器人理解并解释复杂场景；复旦大学团队张世铎、邱锡鹏等人2024年面向自然语言中的隐含意图这一场景设计了VLABench，考察具身智能系统任务理解和规划的能力。2025年的RoboBench（Luo et al.,2025）进一步对VLA模型高层推理能力进行了探索。RoboBench将VLA模型中的MLLM看作具身系统的“系统2”，并提出了指令理解、感知推理、泛化规划、出力（Affordance）预测和失败分析五个测量宏观规划能力的维度。这些工作初步考量了大语言模型面向语言指令的宏观规划能力，但是动作细粒度语义理解仍然有待探索。

### 2.3.3 测评范式：LLM作为被试与可逆性探索

基于上述讨论，本文的核心目的可以凝练成：测试在人类-具身智能的交互过程中，基座模型与人类对动词的理解是否具有具身层面的同质性。这一目标设计一致性测量和认知能力评估两个层面，因此本文引入了一致性测评和可逆性测评两个测评范式（可逆性需要扩写）。

本文引入了“一致性测评”范式来进行测量。一致性测评是LLM测评中的重要组成部分，在道德、价值观等领域得到了验证。其中，Scherrer等人(2023)将LLM视为问卷调查的受访者，通过实证调查测评模型中编码的道德信念，并使用了基于熵的测量方法，量化了模型做出选择的概率、不确定性以及一致性； Rozen(2024)使用施瓦兹肖像价值问卷PVQ-RR对LLM在不同提示词下的价值观结构进行了测试；Ye等人（2024）借用了生成式测量心理学的动态测评方法，测量了大语言模型的价值观。这些一致性测评研究说明，提示词环境下，LLM可以被视作被试对象，且其在这些领域的输出结果与人类结果具备相似性。

受到LLM测评研究和认知语义学的启发，本研究提出一种新型的“动作语义对齐测评”方案。我们不再单纯关注动作执行的端到端成功率，而是利用 Gao (2016) 表演范式（Enactment Paradigm）下的行为实验数据作为人类基准，将大模型置于模拟的实验环境下，要求其对“丢”类近义动词进行多维度的物理参数还原。通过量化模型预测参数与人类实证数据之间的统计距离（如 MSE），我们可以客观评估当前 VLA 模型在缺失本体感受的情况下，其对动作语义理解的深度与广度。这一路径不仅有助于识别模型在人机协作（如“搀”与“扶”）中的潜在认知风险，更通过跨语言、跨模型的对比，揭示了具身智能迈向“知行合一”的本质障碍。

## 三、研究方法

### 3.1行为实验基准：

本研究的实验仿照Gao（2016）的实验开展。

Gao的原实验对中文、英文、德语三个语言下的六个身体动作动词开展测试，这些动词均属于“投掷类”语义场。实验选取了中、英、德母语被试各30人，并控制了被试的出生地、性别、年龄等特征。被试按照母语进行分组，并被要求在房间中表演出实验者向其展示的动作。

表演者的行为后被分为五个维度进行参数量化：力量、手部高度、手臂姿势、水平方向和垂直轨迹。实验结果证明，五个参数为度能够清晰地区分出五个不同的动词，且三个语言之间的动词不存在严格的对译。同时，不同文化的被试的动作演示存在不同的建模框架与规律性的差异，例如英语母语者对力度、手部方向、手臂姿势敏感，而汉语母语者则对五个维度全部敏感。

为了开展跨语言的对比评测，本研究沿用Gao (2016) 的研究，选取了相同的：

本文后续分析所依照的人类基准真值数据如下：

**[表2]**

| 表2 丢类动词的描述性特征（Gao,2016） | 表2 丢类动词的描述性特征（Gao,2016） | 表2 丢类动词的描述性特征（Gao,2016） | 表2 丢类动词的描述性特征（Gao,2016） | 表2 丢类动词的描述性特征（Gao,2016） | 表2 丢类动词的描述性特征（Gao,2016） |
| --- | --- | --- | --- | --- | --- |
|  | 力度 (FORCE) | 手部高度 (HAND) | 臂姿 (ARM) | 水平方向 (HD) | 垂直轨迹 (VD) |
|  | 均值 (SD) | 均值 (SD) | 弯曲 : 伸直 | 向前 : 向侧 | 向上 : 向下 |
| 汉语“投掷”类动词 | 汉语“投掷”类动词 | 汉语“投掷”类动词 | 汉语“投掷”类动词 | 汉语“投掷”类动词 | 汉语“投掷”类动词 |
| 扔 (rēng) | 3.05 (0.51) | 5.43 (1.82) | 11 : 18 | 21 : 8 | 21 : 8 |
| 丢 (diū) | 2.33 (0.83) | 4.63 (1.08) | 6 : 24 | 16 : 12 | 23 : 5 |
| 抛 (pāo) | 3.16 (0.45) | 6.65 (2.52) | 14 : 15 | 29 : 1 | 30 : 0 |
| 投 (tóu) | 3.21 (0.52) | 9.13 (0.79) | 30 : 0 | 30 : 0 | 29 : 1 |
| 摔 (shuāi) | 4.55 (0.56) | 8.51 (1.11) | 29 : 0 | 25 : 4 | 0 : 29 |
| 甩 (shuǎi) | 3.74 (0.68) | 6.27 (1.49) | 21 : 8 | 9 : 20 | 14 : 15 |
| 英语“投掷”类动词 | 英语“投掷”类动词 | 英语“投掷”类动词 | 英语“投掷”类动词 | 英语“投掷”类动词 | 英语“投掷”类动词 |
| throw | 3.62 (0.64) | 8.58 (1.91) | 24 : 5 | 29 : 0 | 26 : 3 |
| fling | 3.44 (0.70) | 6.41 (1.62) | 17 : 11 | 25 : 4 | 27 : 2 |
| chuck | 3.91 (0.68) | 7.06 (2.18) | 19 : 10 | 27 : 2 | 21 : 8 |
| cast | 3.01 (0.47) | 6.18 (2.30) | 15 : 14 | 27 : 2 | 28 : 1 |
| hurl | 4.39 (0.74) | 8.00 (2.07) | 21 : 8 | 28 : 1 | 24 : 5 |
| toss | 3.01 (0.22) | 4.50 (1.63) | 3 : 26 | 29 : 0 | 29 : 0 |

### 3.2 仿真开展

### 3.2.1 模拟被试：

	本研究的实验中，选取两类开源大语言模型作为“被试”：

视觉-语言-动作（VLA）模型：Qwen2-VL-7B-Instruct。该模型在亿级图文数据上预训练，具备跨模态动作语义理解能力。

纯文本大语言模型（LLM）：Qwen2-7B-Instruct，作为VLA模型的对照，用于评估视觉信息对动作语义理解的具体贡献。

两类模型均通过 Hugging Face Transformers 库加载，采用 4-bit 量化（load_in_4bit=True）以适配本地计算资源（AMD Ryzen 7840HS CPU + 16GB 共享内存）。模型推理温度固定为 0.5，每个实验条件重复采样 30 次，取均值作为稳定估计。

### 3.2.2 实验设计

本模型的实验设计为2（视角：直接 vs 间接）× 2（模型：VLA vs LLM）× 2（输出格式：参数格式 vs 言语格式）× 12（6个英语动词+6个汉语动词） 的四因素设计，其中各次API独立调用间不存在缓存与顺序效应。

实验的自变量包括：实验视角，模型类型、输出格式、语言材料；其中实验视角指的是要求模型作为被试描述自己的动作，或要求模型作为主试描述虚拟被试的动作。因变量包括模型输出的五个运动维度参数与人类真值之间的均方误差（MSE）。MSE 计算前研究人员会将模型输出的离散/连续值线性映射与人类数据一同映射到0-1量纲之间。。

研究人员在每个变量水平下重复采样 30 次，取 30 次 MSE 的均值作为该动词在该条件下的观测值。

### 3.2.2 言语表征-参数映射

	本研究区分了言语和参数两种输入（输出）格式，以测量模型不同的反应。

### 3.3 指标构建和数据分析

	本研究使用均方误差和衡量模型的动作语义理解准确度。均方误差和月底，说明模型认识越准确

参照 Gao (2016) 的语义分解模型，在实验场景1（直接/间接表演范式）中，本研究将每一个“丢”类动作映射为一个五维特征向量：

力度 (): 采用五点李克特量表，数值区间为，其中 5 代表“极强”，1 代表“极轻”。

手臂姿态 (): 对应人类实验中的“伸直”与“弯曲”频率。本研究要求模型输出其预测“手臂为伸直状态”的主观概率，并将其与人类标注数据的经验频率分布进行对齐比较。

手部高度 (): 采用相对高度标注法。以参与者脚部地面高度为 0，头顶高度为 10，将动作起始时的手部位置映射至的连续区间。

垂直运动方向 (): 二元编码，数值 1 代表“向下”，0 代表“向上”。

水平运动方向 (): 二元编码，数值 1 代表“向前”，0 代表“向侧”。

通过上述参数化处理，模型生成的非结构化自然语言描述可以被转化为结构化的向量。

### 3.4.2 数据重缩放

为了消除不同维度量纲差异的影响，研究人员在获取了数据后针对不同的量纲进行了线性重缩放将所有原始观测值映射至统一的单位区间。

对于第个动词的第个维度数值，其归一化后的标准值计算公式为：
令为测试动词总数（）,为特征维度总数（）。则模型与人类基准之间的对齐误差和定义如下：


其中：为模型针对第个动词在第个维度生成的归一化预测均值；为人类基准数据在对应维度上的归一化期望值。

## 四、结论

研究人员对收集到的数据进行了统计检验和数据可视化，观察到如下结论：

### 4.1 GLM-4V偏好参数格式输出

研究人员统计了四个模型在参数和言语格式下的均方误差和，发现GLM-4V模型在参数格式的输入情况下，均方误差和显著低于言语格式组，说明GLM-4V显著偏好参数格式的输出。在另外的三组模型中则没有观察到类似的显著现象。

参数格式更广泛的参与了模型的训练，也符合VLM基座模型特定。视觉模型对于参数格式的偏好或许说明其能够较为准确的捕捉语义参数特征，并将其转化为运动学数据。这支持了GLM-4V拥有“具身认知潜能”的假设，因为其动作理解能力并不依赖于动作表征的一致性。但是这种现象并没有出现在Qwen模型中，因此这或许是GLM-4V模型的自身特征，而非训练方法的普遍效果。

### 4.2：LLM在动词语义理解的任务下对语言不敏感

为了探究语言资源分布是否会导致模型产生动作语义的认知偏差，研究人员针对各模型在中文与英文任务下的均方误差（MSE）进行了配对样本 t 检验 。

这一发现揭示了大模型具身认知机制的两个本质特征，值得在后续研究中进一步探讨：不同于人类被试在 Gao (2016) 的实证研究中展现出的文化偏向性（即母语会显著改变人类对“丢”类动词物理参数的界定），LLM 与 VLM 的表现对语言的变化并不敏感。这可能表明模型内部对于动作语义的编码并非扎根于特定的语言文化经验，而是源于大规模跨语言对齐预训练后，在潜在空间内涌现出的一套通用的、去背景化的物理常识模版。

同时，通过对不同模型组合的语言多样性进行可视化对比，我们可以进一步观察语言对模型的影响。下图中图 (a) 的柱状图显示中文 MSE（0.789）略高于英文（0.596），但结合图 (b) 的小提琴图可以看出，这种微小差异源于中文语境下个别动词（如“投”）极大的离群误差，而非整体认知的落后。

图 (c) 和 (d) 的配对 t 检验结果明确显示，所有测试模型（GLM-4, Qwen2 及其多模态版本）在中文和英文环境下的 MSE 差异均不具备统计学显著性（）。

这意味着，无论输入是汉语还是英语，模型提取出的物理参数向量是高度趋同的。

### 4.3 视觉模态引入提升的非均匀性

研究人员统计了Qwen2 系列模型在多模态对齐前后针对特定动词的均方误差和差值，发现视觉模态的引入对动作语义认知的提升具有明显的“非均匀性”和“任务依赖性”。 

在“投（中）”、“throw（英）”和“hurl（英）”等动词上，Qwen2-VL 的 MSE和 显著低于其纯文本基座 Qwen2（差值最高达 -1.702）。这说明视觉预训练能显著优化对这些动作的空间认知。

然而，在“丢（中）”、“扔（中）”和“fling（英）”等动词上，视觉模态的引入反而导致了 MSE 的上升（正差值）。这暗示在处理低物理量特征或日常高频泛化词时，多模态训练可能引入了视觉噪声，导致其在参数精细化预测上不如经过大规模文本对齐的 LLM 稳定。

这种认知提升的非均匀性说明，视觉语言模型对大语言模型的提升并非全面的。视觉语言模型如果能够比大语言模型表现得更好，并非由于其表现出了某种普遍的、常识性的物理认识，而是因为它对一些特征更加熟悉。、

## 五、结语

综合分析发现，VLA 基座模型的认知模式呈现出‘宏观鲁棒、微观失真’的特征。一方面，统计检验（图 4）证实了模型在处理不同语言时具有极高的稳定性，并未表现出人类般的语言文化偏向，这反映了其‘潜具身性’的统计本质；另一方面，动词级的深度剖析（图 2c）揭示了模态对齐在特定动作语义上的失效风险。这种视觉增强并不总是正向的，尤其在精细动作语义与轨迹模态的映射中，模型表现出明显的逻辑波动。这为未来针对特定动作（如‘搀’与‘扶’）进行内生安全性对齐提出了迫切的技术需求。

未来，研究可以考虑对更多模型进行拓展测评，已提出更具有指导意义的测评指标。同时研究结果建议大模型开发者对模型进行语言特征的针对性训练，以提升其文化敏感性和动作语义理解的深度。

**参考文献**

Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., ... & Zeng, A. (2022). Do as I can, not as I say: Grounding language in robotic affordances. arXiv:2204.01691.

Atkins, B. T. S., & Levin, B. (1995). Building on a corpus: A linguistic and lexicographical look at some near-synonyms. International Journal of Lexicography, 8(2), 85-114.

Barsalou, L. W. (1999). Perceptual symbol systems. Behavioral and Brain Sciences, 22(4), 577-609.

Bergen, B., Lau, T., Narayan, S., Stojanovic, D., & Wheeler, K. (2010). Body part representations in verbal semantics. Memory & Cognition, 38(7), 969-981.

Bisk, Y., Zellers, R., Le Bras, R., Gao, J., & Choi, Y. (2020). Experience grounds language. In Proceedings of EMNLP 2020 (pp. 8718-8735). ACL.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., ... & Zeng, A. (2022). RT-1: Robotics transformer for real-world control at scale. arXiv:2212.06817.

Cruse, D. A. (1986). Lexical semantics. Cambridge University Press.

de la Riva Lopez, E. M., Francis, W. S., & Garcia, J. (2012). Repetition priming within and between languages in verb generation. Memory, 20(4), 358-373.

DiMarco, C., Hirst, G., & Stede, M. (1993). The semantic and stylistic differentiation of synonyms and near-synonyms. In Proceedings of the AAAI Spring Symposium (pp. 114-121).

Divjak, D., & Gries, S. T. (2006). Ways of trying in Russian: Clustering behavioral profiles. Corpus Linguistics and Linguistic Theory, 2(1), 23-60.

Divjak, D., & Gries, S. T. (2008). Clusters in the mind? Converging evidence from near synonymy in Russian. The Mental Lexicon, 3(2), 188-213.

Driess, D., Xia, F., Sajjadi, M. S. M., Lynch, C., Chowdhery, A., Ichter, B., ... & Florence, P. (2023). PaLM-E: An embodied multimodal language model. arXiv:2303.03378.

Gao, H. (2001a). The physical foundation of the patterning of physical action verbs: A study of Chinese verbs. Lund University Press.

Gao, H. H., Wang, H., & Nicoladis, E. (2016). The delineation of throw verbs in Mandarin Chinese: Behavioural and perceptual approaches. Journal of Cognitive Science, 17(1), 95-131.

Gibbs, R. W. (2003). Embodiment and cognitive science. Cambridge University Press.

Hoang, H., Mori, Y., Nicoladis, E., Gao, H. H., & Du, Y. (2024). HL Mandarin speakers toss the same way as fluent Mandarin speakers. Heritage Language Journal.

Jackendoff, R. (2002). Foundations of language: Brain, meaning, grammar, evolution. Oxford University Press.

Jackendoff, R. (2007). Language, consciousness, culture: Essays on mental structure. MIT Press.

Kennedy, A., & Hirst, G. (2012). Measuring semantic relatedness across languages. In Proceedings of the xLiTe Workshop at NIPS 2012.

Lakoff, G., & Johnson, M. (1980). Metaphors we live by. University of Chicago Press.

Liu, D. (2010). Is it a chief, main, major, primary, or principal concern? International Journal of Corpus Linguistics, 15(1), 56-87.

Liu, D. (2013). Salience and construal in the use of synonymy. Cognitive Linguistics, 24(1), 67-113.

Majid, A., Boster, J. S., & Bowerman, M. (2008). The cross-linguistic categorization of everyday events. Cognition, 109(2), 235-250.

Majid, A., Gullberg, M., Staden, M. V., & Bowerman, M. (2007). How similar are semantic categories in closely related languages? Cognitive Linguistics, 18(2), 179-194.

Malt, B. C., Gennari, S., Imai, M., Ameel, E., Tsuda, N., & Majid, A. (2008). Talking about walking: Biomechanics and the language of locomotion. Psychological Science, 19(3), 232-240.

Qin, Y., Zhou, E., Liu, Q., Yin, Z., Sheng, L., Zhang, R., ... & Shao, J. (2024). MP5: A multi-modal open-ended embodied system in Minecraft via active perception. In Proceedings of CVPR 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. In Proceedings of ICML 2021 (pp. 8748-8763).

Talmy, L. (1985). Lexicalization patterns. In T. Shopen (Ed.), Language typology and syntactic description (Vol. 3, pp. 57-149). Cambridge University Press.

Talmy, L. (2000). Toward a cognitive semantics (Vols. 1-2). MIT Press.

Tranel, D., Kemmerer, D., Adolphs, R., Damasio, H., & Damasio, A. R. (2003). Neural correlates of conceptual knowledge for actions. Cognitive Neuropsychology, 20(3-6), 409-432.

Varela, F. J., Thompson, E., & Rosch, E. (1991). The embodied mind: Cognitive science and human experience. MIT Press.

Wang, H., & Gao, H. H. (2016). Cross-linguistic categorization of throwing events: A behavioral approach. Cognitive Linguistic Studies, 3(2), 259-276.

Wang, S., & Hirst, G. (2012). Exploring patterns in dictionary definitions for synonym extraction. Natural Language Engineering, 18(3), 313-342.

Willems, R. M., Hagoort, P., & Casasanto, D. (2010). Body-specific representations of action verbs. Psychological Science, 21(1), 67-74.

Yang, Y., Sun, F. Y., Weihs, L., VanderBilt, E., Herrasti, A., Han, W., ... & Clark, C. (2024). Holodeck: Language guided generation of 3D embodied AI environments. In Proceedings of CVPR 2024.

Zellers, R., Bisk, Y., Schwartz, R., & Choi, Y. (2019). SWAG: A large-scale adversarial dataset for grounded commonsense inference. In Proceedings of EMNLP 2018 (pp. 93-104).

Zitkovich, B., Yu, T., Xu, S., Xu, P., Xiao, T., Xia, F., ... & Brohan, A. (2023). RT-2: Vision-language-action models transfer web knowledge to robotic control. In Proceedings of CoRL 2023, PMLR 229, 2165-2183.

Long, X., Zhao, Q., Zhang, K., Zhang, Z., Wang, D., Liu, Y., Shu, Z., Lu, Y., Wang, S., Wei, X., Li, W., Yin, W., Yao, Y., Pan, J., Shen, Q., Yang, R., Cao, X., & Dai, Q. (2025). *A Survey: Learning Embodied Intelligence from Physical Simulators and World Models* (arXiv:2507.00917). arXiv. （page6-7）

BBC News. (2017, December 15).Mirai botnet: Three admit creating and running attack tool. 

BBC News. (2015, July 1). *Robot 'kills' worker at Volkswagen plant in Germany*. 

Pfeifer, R., & Bongard, J. C. (2006). *How the body shapes the way we think: a new view of intelligence*. MIT press.

Duan, J., et al. (2022). *A Survey of Embodied AI: From Design to Intelligent Behavior*. IEEE Transactions on Artificial Intelligence.

Li, X., Li, P., Qian, L., Liu, M., Wang, D., Liu, J., Kang, B., Ma, X., Wang, X., Guo, D., Kong, T., Zhang, H., & Liu, H. (2026). What Matters in Building Vision-Language-Action Models for Generalist Robots (arXiv:2412.14058). arXiv. 

Scherrer, N., Shi, C., Feder, A., & Blei, D. M. (2023). *Evaluating the Moral Beliefs Encoded in LLMs* (arXiv:2307.14324). arXiv. 

Sorensen, T., Jiang, L., Hwang, J., Levine, S., Pyatkin, V., West, P., Dziri, N., Lu, X., Rao, K., Bhagavatula, C., Sap, M., Tasioulas, J., & Choi, Y. (2024). Value Kaleidoscope: Engaging AI with Pluralistic Human Values, Rights, and Duties. *Proceedings of the AAAI Conference on Artificial Intelligence*, *38*(18), 19937～19947. 

Feng, R., Hu, J., Xia, W., Gao, T., Shen, A., Sun, Y., Fang, B., & Hu, D. (2025). *AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors* (arXiv:2502.12191). arXiv. 

Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Shi, L. X., … Zhilinsky, U. (2024). *: A Vision-Language-Action Flow Model for General Robot Control* (arXiv:2410.24164; 版 1). arXiv. 

Li, J., Sun, S., Yuan, W., Fan, R.-Z., Zhao, H., & Liu, P. (2023). **Generative Judge for Evaluating Alignment**. *arXiv preprint arXiv:2310.05470*.

Rozen, N., Bezalel, L., Elidan, G., Globerson, A., & Daniel, E. (2024). **Do LLMs have consistent values?** *arXiv preprint arXiv:2407.12878v3*.

Scherrer, N., Shi, C., Feder, A., & Blei, D. M. (2023). **Evaluating the Moral Beliefs Encoded in LLMs**. In *Advances in Neural Information Processing Systems, 36* (NeurIPS 2023).

Sorensen, T., Jiang, L., Hwang, J. D., Levine, S., Pyatkin, V., West, P., Dziri, N., Lu, X., Rao, K., Bhagavatula, C., Sap, M., Tasioulas, J., & Choi, Y. (2024). **Value Kaleidoscope: Engaging AI with Pluralistic Human Values, Rights, and Duties**. In *Proceedings of the AAAI Conference on Artificial Intelligence, 38*.

Ye, H., Xie, Y., Ren, Y., Fang, H., Zhang, X., & Song, G. (2024). **Measuring Human and AI Values Based on Generative Psychometrics with Large Language Models**. *arXiv preprint*.

Zheng, J., Wang, H., Zhang, A., Nguyen, T. D., Sun, J., & Chua, T.-S. (2024). **ALI-Agent: Assessing LLMs' Alignment with Human Values via Agent-based Evaluation**. In *Advances in Neural Information Processing Systems, 37* (NeurIPS 2024).
