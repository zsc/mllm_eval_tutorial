# 第 3 章：评测平台工程化：统一接口、批量运行、可视化与 CI

## 1. 开篇与学习目标

在单模态 NLP 时代，跑几个 benchmark 可能只需要一个 Jupyter Notebook。但在 MLLM（多模态大模型）时代，输入涉及 4K 视频流、长音频、高分辨率文档图像，输出涉及多轮对话、工具调用（Function Call）和多媒体生成。数据量的爆炸和交互的复杂性，使得**评测本身如果不工程化，将成为模型迭代最大的瓶颈。**

一个优秀的评测平台不仅仅是“跑分工具”，它是**模型能力的体检中心**和**产品发布的守门员**。本章的核心目标是构建一个**模块化、可扩展、抗脆弱**的生产级评测基础设施。

**本章学习目标**：
1.  **架构设计模式**：掌握 Configuration-as-Code 和 Pipeline 模式，实现数据、模型、评测逻辑的彻底解耦。
2.  **高吞吐数据管线**：深入理解多模态数据的 ETL（提取、转换、加载）流程，处理视频解码、音频流式加载与缓存策略。
3.  **运行器与容错**：设计支持断点续跑、并发控制、分布式调度的健壮 Runner。
4.  **CI/CD 流水线**：建立从 PR 冒烟到版本门禁的分级测试体系，设定 Latency 和 Cost 预算。
5.  **驾舱一体化仿真**：在离线环境中构建“虚拟车辆”，Mock 硬件信号与车控 API，实现端到端闭环评测。

---

## 2. 评测平台架构体系

一个成熟的 MLLM 评测平台（Evaluation Harness）应遵循 **“配置驱动 (Config-Driven)”** 和 **“组件插件化 (Plugin-Based)”** 的原则。

### 2.1 顶层设计与 ASCII 视图

平台的核心思想是：**将变化最快的部分（模型接口、数据集）与相对稳定的部分（调度逻辑、打分算法）分离。**

```text
[用户输入] (Config YAML)
     │
     ▼
[注册中心 (Registry)] <--- 动态加载 Dataset / Model / Scorer 类
     │
     ▼
[运行器 (Runner)] -------------------------------------------+
│  1. Data Loader (ETL: 解码/采样/缓存)                       |
│  2. Prompt Builder (模版组装/Few-shot注入)                  |
│  3. Model Adapter (异构接口统一/重试/限流)                   |
│  4. Response Dumper (实时落盘 jsonl)                        |
+------------------------------------------------------------+
     │
     ▼
[后处理流水线 (Post-processing Pipeline)]
│  1. Extractor (正则/解析器: 从输出提取答案)
│  2. Judge/Scorer (计算指标: WER/Rouge/GPT-4-Score)
│  3. Analyzer (聚合分析: 失败聚类/对比Diff)
+------------------------------------------------------------+
     │
     ▼
[可视化报告 (Dashboard)] (HTML/WandB)
```

### 2.2 仓库结构约定 (Repository Structure)

推荐采用类似 OpenCompass 或 Detectron2 的配置继承机制，以支持大量的消融实验。

```text
mllm-eval-platform/
├── configs/                  # [配置中心]
│   ├── _base_/               # 基础配置 (继承用)
│   │   ├── datasets/         # 数据集定义 (path, reader_cfg)
│   │   └── models/           # 模型参数 (temperature, max_tokens)
│   ├── regression/           # 回归任务集 (e.g., nightly_v1.py)
│   └── ablation/             # 消融实验集 (e.g., context_len_test.py)
├── core/
│   ├── adapters/             # [模型适配] HTTP/gRPC/Local Wrapper
│   ├── data/                 # [数据管线] Transform, CollateFn
│   ├── evaluators/           # [打分器] Metric calculators
│   └── hooks/                # [钩子] Callbacks (logging, visualization)
├── tools/
│   ├── analysis/             # 错误分析工具, 训练数据查重
│   └── mock/                 # [车端模拟] Vehicle Signal Mocker
├── work_dirs/                # [产出物]
│   └── 20250101_model_v2/
│       ├── config_dump.yaml  # 复现快照
│       ├── predictions.jsonl # 中间结果 (含 prompt, raw output)
│       ├── results.json      # 最终分数
│       └── failure_cases/    # 渲染后的坏例页面
└── tests/                    # 单元测试
```

---

## 3. 关键组件工程化详解

### 3.1 模型适配层 (Model Adapter)：统一异构接口

MLLM 的后端千奇百怪（HF Pipeline, vLLM, TensorRT-LLM, 私有云 API）。Adapter 必须屏蔽这些差异，向 Runner 暴露**统一的多模态消息协议**。

**Rule of Thumb**: 内部协议应向 OpenAI Chat 格式靠拢，但必须扩展对本地媒体文件的支持。

*   **输入标准化**：
    ```python
    # 内部统一表示 (Intermediate Representation)
    message = {
        "role": "user",
        "content": [
            {"type": "image", "source": "/local/path/to/img.jpg"}, # 或 base64, 或 s3_url
            {"type": "audio", "source": "/local/path/to/speech.wav"},
            {"type": "text", "text": "这段语音里的用户想去哪里？"}
        ]
    }
    ```
*   **参数透传**：Adapter 应支持 `generation_kwargs` 的透传，以便控制 `temperature`, `top_p`, `repetition_penalty`。
*   **异常处理**：必须在 Adapter 层处理 `ModelOverloaded`, `RateLimitError`，实现指数退避重试（Exponential Backoff）。

### 3.2 数据管线 (Data Pipeline)：从 I/O 到 Tensor

这是 MLLM 评测中最重、最易出错的环节。

*   **视频解码策略 (Video Decoding)**：
    *   *问题*：在线解码 4K 视频极其消耗 CPU，导致 GPU 等待（Starvation）。
    *   *优化*：
        1.  **离线预处理**：对于固定的数据集，预先抽取关键帧（FPS=1 或 场景切变点）存为 JPG。
        2.  **缓存机制**：使用 LRU Cache 缓存最近解码的帧或 Audio Mel-Spectrogram。
        3.  **多进程 Loader**：PyTorch `DataLoader` 的 `num_workers > 0` 是必须的。
*   **确定性 (Determinism)**：
    *   评测中**严禁**使用随机增强（Random Flip, Color Jitter）。
    *   Resize 和 Crop 必须中心对齐（Center Crop）。
    *   音频重采样（Resample）必须指定固定的算法和 backend (e.g., `sox` vs `ffmpeg`)，微小的差异可能导致 ASR WER 波动。
*   **动态 Batching**：
    *   纯文本容易 batch，但不同长度的视频/音频混合 batch 极难。
    *   *策略*：对于多模态任务，推荐 **Batch Size = 1** 并发执行（通过多线程/多进程请求 API），或者按模态+长度分桶（Bucketing）后再 Batch。

### 3.3 运行器 (Runner) 与 状态管理

*   **断点续跑 (Resume Capability)**：
    *   **原则**：评测过程随时可能因为 OOM 或网络波动崩溃。
    *   **实现**：Runner 启动时，先扫描 output 目录下的 `results.jsonl`，读取已存在的 `sample_id` 加载到 Set 中。数据迭代器自动 `continue` 跳过这些 ID。
    *   **原子写入**：确保每一条结果写入是原子的（flush），避免 json 截断。
*   **并发控制 (Concurrency Throttling)**：
    *   配合 Token Bucket 算法控制每分钟请求数 (RPM) 和 Token 数 (TPM)，防止被云端 API 封禁或把自建推理服务打挂。

### 3.4 打分器 (Scorer)：解耦与多级评测

打分不应与推理耦合。推荐采用 **Inference -> Dump -> Judge** 的两阶段模式。

1.  **客观提取器 (Extractor)**：
    *   针对选择题，使用增强型正则（Regex）提取选项。处理 "The answer is A" 或 "A matches the image" 等变体。
    *   针对 JSON 输出，使用宽松的 JSON Parser（如 `json5` 或大模型修复 parser）。
2.  **相似度计算器 (Metric Calculator)**：
    *   文本：BLEU, ROUGE-L, METEOR.
    *   代码：Pass@k (需沙箱执行).
    *   向量距离：Embedding Cosine Similarity (语义相似度).
3.  **LLM-as-a-Judge**：
    *   使用 GPT-4 或专门微调的 Critic 模型对开放性问题打分（1-10分）。
    *   **Gotcha**: Judge 模型自身存在 bias（偏好长回复）。需要校准（Calibration）或使用 Pairwise 比较（Win/Tie/Loss）。

### 3.5 报告与可视化系统

一份 100MB 的日志文件没人看，需要可视化的仪表盘。

*   **动态仪表盘**：集成 Streamlit 或 WandB。
*   **Diff View (关键功能)**：
    *   左侧：Baseline 模型输出。
    *   右侧：当前模型输出。
    *   高亮：语义差异部分。
    *   *用途*：用于 RAG 评测，快速判断新模型是否引入了幻觉。
*   **失败聚类 (Failure Clustering)**：
    *   自动将 Error Case 按 tag 归类：`[OCR_Miss]`, `[Logic_Error]`, `[Safety_Trigger]`.

---

## 4. CI/CD 集成策略：将评测融入研发心跳

不要等到发版前才评测。

### 4.1 分级测试金字塔

| 级别 | 触发时机 | 数据规模 | 核心指标 | 耗时预算 |
| :--- | :--- | :--- | :--- | :--- |
| **L0: Smoke (冒烟)** | 每次 PR 提交 | 极小 (每类任务 5-10 例) | 格式正确性, 无 Crash | < 5 min |
| **L1: Nightly (回归)** | 每日凌晨 | 中等 (核心集 10% 采样) | WER, Accuracy, RAG召回率 | < 2 hrs |
| **L2: Weekly/Release** | 发版/周维度 | 全量 (所有 Benchmark) | 全项指标 + 竞品对比 | > 24 hrs |

### 4.2 质量门禁 (Quality Gates)

在 CI Pipeline 中设置硬性拦截规则：
1.  **功能回退拦截**：核心任务（如导航指令解析）准确率下降幅度不得超过 1%。
2.  **时延拦截**：TTFT (Time To First Token) 95分位值不得增加超过 50ms。
3.  **安全拦截**：Prompt Injection 攻击成功率必须为 0%。

---

## 5. 训练数据问题反查工作流

当评测分数异常高（疑似泄漏）或由于“不知道”而回答错误时，需要反查训练数据。

**工程实现**：
1.  **索引构建**：使用向量数据库（Milvus/FAISS）对所有训练数据（图/文）建立索引。
2.  **即时检索**：评测报告页面增加 "Check Training Data" 按钮。
    *   点击后，Embedding 当前评测样本，在向量库中检索 Top-K 相似训练样本。
3.  **污染判定**：计算 n-gram 重叠率或图像指纹相似度。如果重叠率 > 0.8，标记为 "Contaminated"，该分数在最终报告中应剔除或降权。

---

## 6. 车舱落地：驾舱一体工程化专项

车载环境的特殊性在于：**强实时性、硬件依赖、端云协同**。通用评测框架无法直接覆盖，需要定制。

### 6.1 虚拟车身环境 (Mock Vehicle Environment)

在离线评测 Runner 中，必须注入一个 **Context Manager** 来模拟车辆状态。

*   **信号仿真 (CAN Simulation)**：
    *   构造一个 `VehicleState` 对象，包含：`Speed`, `Gear`, `WindowPosition`, `AC_Temp`, `Passengers [Seat_ID, Status]`.
    *   **动态变化**：支持剧本（Scenario）定义。例如：第 1 轮对话车速 0，第 3 轮对话车速 80（模拟起步），考察模型是否因为车速变化而拒绝视频播放请求。
*   **API Mocking**：
    *   拦截模型输出的 `open_window(seat="driver")` 工具调用。
    *   返回模拟的执行结果：`{"status": "success", "new_state": "open"}` 或 `{"error": "hardware_failure"}`。
    *   **验证点**：模型收到 error 后，是否能生成合理的安抚话术，而不是复读指令。

### 6.2 端侧 (Edge) vs 云侧 (Cloud) 双栈评测

车机通常采用“端侧小模型（快速响应/隐私）+ 云侧大模型（复杂知识）”的架构。

*   **端侧评测台架**：
    *   利用 Android Debug Bridge (ADB) 或 SSH 连接开发板（Orin-X / 8295）。
    *   推送量化模型（W4A16, INT8）和测试集到端侧。
    *   执行推理并拉取日志。
*   **一致性校验 (Consistency Check)**：
    *   对比端侧量化模型与云侧 FP16 模型的输出分布（KL 散度）。
    *   监控端侧特有指标：**RAM 占用峰值**、**NPU 利用率**、**功耗**。

### 6.3 硬件在环 (HIL) 辅助

对于音频和视觉，纯软件 Mock 不够。
*   **音频注入**：不直接喂 wav 文件，而是通过声卡回环或专门的音频注入设备，模拟车机麦克风阵列的信号路径（考察 AEC 回声消除和降噪算法对 ASR 的影响）。
*   **时延分段统计**：
    *   在评测框架中埋点，精确统计：
        *   `T_ASR`: 语音转文字耗时
        *   `T_NLU`: 意图理解耗时
        *   `T_LLM_FirstToken`: 思考首字耗时
        *   `T_TTS_Ready`: 语音合成首包耗时
    *   **Rule of Thumb**: 驾舱交互要求 `T_ASR_End` 到 `T_TTS_Start` (响应延迟) < 800ms ~ 1.2s。

---

## 7. 本章小结

*   **架构解耦**：Adapter 负责兼容，Runner 负责调度，Scorer 负责裁判，三者分离是平台可维护的关键。
*   **数据决定上限**：多模态评测中，视频解码、音频采样的确定性和缓存策略决定了评测的稳定性和速度。
*   **持续集成**：评测不是一次性的活动，而是研发流水线中的“心跳”，必须分级进行。
*   **车规级要求**：驾舱一体评测需要引入“虚拟车身”和“硬件约束”，关注端云一致性与全链路时延。

---

## 8. 练习题

### 基础题 (50%)
1.  **架构理解**：画出从 Dataset 加载到生成 Final Report 的数据流图，标出哪里需要缓存，哪里可以并发。
2.  **配置设计**：设计一个 YAML 配置文件片段，描述一个名为 "traffic_sign_recognition" 的评测任务，包含 dataset path, metric (accuracy), 和一个特定的 prompt template。
3.  **指标计算**：编写伪代码，实现一个名为 `ResponseDumper` 的类，具备 `dump(sample_id, input, output)` 方法，要求能够处理程序崩溃后的数据完整性（提示：flush 和 append）。

### 挑战题 (50%)
1.  **流式评测设计**：对于 ASR 任务，设计一个 Runner，模拟实时音频流输入（chunk by chunk），并计算“用户说完话到文字上屏”的延迟。如何确保评测的可复现性？
2.  **训练数据排查**：假设评测发现模型在“打开天窗”这个指令上表现极好（100%），但在“打开遮阳帘”上表现极差（0%）。设计一个自动化的反查流程，利用向量检索技术分析训练数据分布的差异。
3.  **车端 Mock 剧本**：编写一个 JSON 格式的测试剧本，模拟用户在驾驶过程中（Speed > 0），尝试进行“看视频”操作。剧本应包含车辆状态变化、用户的多轮对话输入、以及预期的模型行为（拒绝并建议听音频）。

<details>
<summary>点击查看提示 (Hint)</summary>

*   **基础题3 Hint**: 打开文件时使用 mode='a' (append)。每次 write 后调用 `file.flush()` 或 `os.fsync()` 强制刷盘。
*   **挑战题1 Hint**: 需要一个 Generator 来按固定时间间隔 yield 音频块。可复现性需要固定 chunk size 和发送间隔，消除系统调度带来的抖动干扰（使用逻辑时间而非挂钟时间）。
*   **挑战题2 Hint**: 对“打开天窗”和“打开遮阳帘”的指令分别做 Embedding，在训练库中搜索近邻。如果前者能搜到大量 exact match，后者搜不到，说明覆盖率不均。
*   **挑战题3 Hint**: 剧本结构应包含 `initial_state: {speed: 60}`, `turns: [{"user": "我想看狂飙", "expect_action": null, "expect_reply_contains": "驾驶中无法观看"}]`。
</details>

---

## 9. 常见陷阱与错误 (Gotchas)

1.  **多进程死锁 (Deadlock in Multiprocessing)**:
    *   *现象*：评测跑到一半卡住，GPU 显存占用 0，无日志输出。
    *   *原因*：PyTorch 的 `DataLoader` 多进程与 CUDA 初始化冲突，或者 OpenCV 在 fork 模式下的多线程问题。
    *   *对策*：设置 `mp.set_start_method('spawn')`，或者在 Runner 中只使用多线程 (Threading) 请求 API 服务，而不自己在 Runner 进程内跑推理。
2.  **API 欠费或限流导致的“假性零分”**:
    *   *现象*：Benchmark 分数暴跌。
    *   *原因*：云端 API 返回 429 Too Many Requests，Extractor 提取不到答案，判为错。
    *   *对策*：必须在 Adapter 层捕获 HTTP Error，区分“模型答错”和“服务拒绝”。后者应抛出异常暂停评测或无限重试。
3.  **Git LFS 导致的“空文件”**:
    *   *现象*：视频理解任务全错，日志显示“文件损坏”。
    *   *原因*：拉取仓库时没安装 `git-lfs`，图片/视频文件实际上只是几 KB 的指针文本。
    *   *对策*：在 Pipeline 初始化阶段检查媒体文件的 Magic Number 或最小文件大小。
4.  **Prompts 对齐陷阱**:
    *   *现象*：对比两个模型时，A 模型用了 CoT Prompt，B 模型用了 Direct Prompt，导致结论不公平。
    *   *对策*：评测框架应强制将 System Prompt 和 User Template 抽离为独立配置，确保对比实验控制变量。
5.  **车机 Mock 状态未重置**:
    *   *现象*：Case 1 把车速设为 120，Case 2 预期是静止场景，但 Runner 没重置状态，导致 Case 2 触发驾驶安全限制。
    *   *对策*：引入 `teardown` 钩子，每跑完一个 Sample 或 Session，强制重置 Vehicle Mock Context 到默认状态。
