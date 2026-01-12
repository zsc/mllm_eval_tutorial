# 第 15 章：Agent 能力评测（ReAct、工具调用、长任务、记忆）

## 15.1 开篇：从“言语者”到“行动者”

在 MLLM 的进化树中，如果说 RAG 是为了解决“知之为知之”，那么 **Agent（智能体）** 则是为了解决“行胜于言”。Agent 标志着大模型从被动的文本生成器（Chatbot）转变为能够主动感知环境、规划路径、调用工具并改变世界状态的行动者（Copilot/Action-Bot）。

评测 Agent 与评测对话模型存在本质维度的升维：
1.  **非确定性环境**：对话通常是静态的，而 Agent 的每一步操作都会改变环境状态（如删除了文件、扣除了余额），导致后续的输入发生变化。
2.  **误差累积效应**：在一个包含 10 步操作的长任务中，第 1 步的微小偏差（如选错了日期）可能导致第 10 步的结果完全错误。
3.  **闭环验证难题**：评测不仅需要验证模型“说了什么”，更要验证模型“做了什么”以及“结果对不对”。

本章将建立一套完整的 Agent 评测体系，涵盖从简单的工具调用（Function Calling）到复杂的 ReAct 推理循环，再到长程记忆保持的系统性评估。

---

## 15.2 Agent 核心架构与评测切面

目前主流 Agent 多遵循 **ReAct (Reasoning + Acting)** 或 **Plan-and-Solve** 范式。我们需要对 Agent 运行时的每个环节进行切片评测。

### 15.2.1 ReAct 循环的解剖

Agent 的运行是一个由 **[Thought] -> [Action] -> [Observation]** 构成的无限循环，直到任务结束。

**ASCII 示意图：ReAct 循环中的评测埋点**

```text
User Goal: "帮我把 download 文件夹里所有大于 10MB 的 PDF 移动到 archive 目录。"

      +-------------------------------------------------------------+
      |                        MLLM Agent                           |
      |                                                             |
(T1)  | [Thought]:  需要先列出 download 目录文件，筛选大小和后缀。  | <--- 评测切面 A: 规划能力
      |             (Plan: List -> Filter -> Move)                  |      (是否正确分解了子任务？)
      |                                                             |
(A1)  | [Action]:   os.list_dir(path="./download")                  | <--- 评测切面 B: 工具选择与参数
      +-----------------------------+-------------------------------+      (Schema 是否符合？路径对吗？)
                                    | Call Tool
                                    v
      +-------------------------------------------------------------+
      |                 Environment (Mock/Sandbox)                  |
      |                                                             |
(O1)  | [Observation]: ["a.pdf (2MB)", "b.pdf (15MB)", "img.png"]   | <--- 评测切面 C: 环境仿真度
      +-----------------------------+-------------------------------+      (Mock 数据是否足以触发下一步？)
                                    | Return Result
                                    v
      +-------------------------------------------------------------+
      |                        MLLM Agent                           |
      |                                                             |
(T2)  | [Thought]:  发现 b.pdf 是目标文件。a.pdf太小，img不是pdf。  | <--- 评测切面 D: 状态追踪与推理
      |             现在移动 b.pdf。                                |      (是否正确理解了 Observation？)
      |                                                             |
(A2)  | [Action]:   os.move(src="./download/b.pdf", dst="./archive")|
      +-------------------------------------------------------------+
                                    |
                  (循环直至输出 Finish 或达到 Max Steps)
```

### 15.2.2 三大核心能力维度

1.  **工具使用 (Tool Use / Function Calling)**
    *   **检索 (Retrieval)**：在成百上千个工具（API 库）中，能否召回正确的那个？（例如：不要在需要 `math.sqrt` 时调用 `weather.get`）。
    *   **参数填充 (Slot Filling)**：能否从复杂的上下文或模糊的用户指令中提取出精确参数？（例如：将“下周三”转换为 `2025-10-15`）。
    *   **容错 (Exception Handling)**：当工具返回报错（如 `ConnectionError` 或 `InvalidParam`）时，Agent 是崩溃、胡言乱语，还是尝试自我修正参数重试？

2.  **规划与逻辑 (Planning & Reasoning)**
    *   **任务分解**：将复杂目标拆解为线性或并行的子步骤。
    *   **依赖管理**：识别步骤间的依赖关系（必须先 `get_user_id` 才能 `query_balance`）。
    *   **反思 (Reflection)**：在多次尝试失败后，能否改变策略？

3.  **长程记忆 (Long-term Memory)**
    *   **状态保持**：在第 20 轮交互时，是否还记得第 1 轮设定的 `verbose=True` 全局约束。
    *   **跨会话记忆**：能否利用昨天的对话历史来辅助今天的决策（如用户偏好）。

---

## 15.3 评测环境工程：Mock 与 Sandbox

**Rule of Thumb**：**永远不要在不可控的真实环境中评测 Agent 指标。** 真实环境的网络波动、API 变动、数据即时性会导致评测结果无法跨版本对比（Non-deterministic）。

我们需要构建分级的评测环境：

### 15.3.1 Level 1: 静态 Mock (Stateless Mock)
适用于**工具调用准确率**的单元测试。
*   **原理**：预定义好 `(Input Prompt, Expected Tool Call)` 对。
*   **实现**：不执行真正的工具，只检查 LLM 输出的文本（Action 字符串）。
*   **优点**：速度极快，成本低，适合 CI 冒烟测试。
*   **局限**：无法测试多步依赖和错误恢复。

### 15.3.2 Level 2: 状态机 Mock (Stateful Mock)
适用于**多轮 ReAct 流程**评测。
*   **原理**：维护一个虚拟的状态字典（State Dict）。
*   **示例**：
    *   初始状态：`{"files": ["report.pdf"], "trash": []}`
    *   Action: `delete("report.pdf")`
    *   Mock Server 逻辑：更新状态为 `{"files": [], "trash": ["report.pdf"]}`，并返回 `Success`。
*   **优点**：支持多步逻辑验证（如“先查后删”），且完全确定性。
*   **场景**：数据库操作、文件系统操作、购物车流程。

### 15.3.3 Level 3: 沙箱容器 (Sandboxed Container)
适用于**代码生成与执行**（Code Agent）或**复杂 GUI 操作**。
*   **原理**：为每个评测任务启动一个 Docker 容器或虚拟机。
*   **实现**：
    *   使用 OpenDevin 或 E2B 等框架。
    *   环境快照（Snapshot）：每次任务开始前重置到纯净镜像。
    *   网络隔离：限制只能访问 Mock 的 API Server，防止模型通过公网搜索答案（Leaking）。
*   **成本**：极高，通常用于 Nightly 或 Weekly 深度评测。

---

## 15.4 指标体系详解

### 15.4.1 结果质量指标 (Outcome Metrics)

1.  **任务成功率 (Success Rate, SR)**
    *   **定义**：任务结束时，环境状态是否符合预期？
    *   **判定方法**：
        *   *确定性判定*：检查数据库字段、文件是否存在、API 返回值。
        *   *LLM-based Judge*：对于开放性任务（如“写一个关于某事的总结”），将轨迹（Trajectory）和结果喂给 GPT-4 进行打分。
2.  **Pass@K**
    *   给模型 K 次独立尝试的机会（每次从头开始，Temperatue > 0），只要有一次成功即算通过。用于衡量模型的潜能上限。

### 15.4.2 过程质量指标 (Process Metrics)

1.  **轨迹效率 (Trajectory Efficiency)**
    *   $$ \text{Efficiency} = \frac{\text{Steps}_{\text{optimal}}}{\text{Steps}_{\text{actual}}} $$
    *   如果标准做法是 3 步，模型走了 10 步才完成，说明效率极低，虽然 SR=100%，但在车载等场景不可用。
2.  **幻觉工具率 (Tool Hallucination Rate)**
    *   调用了不存在的函数，或捏造了不存在的参数名的比例。
3.  **格式依从度 (Schema Compliance)**
    *   输出的 JSON/XML 能够被标准 Parser 解析成功的比例。

### 15.4.3 记忆与长上下文指标

1.  **信息检索准确率 (Retrieval Accuracy in Context)**
    *   Needle-in-a-Haystack 的变体：在 100 轮对话历史中插入一条 Action 指令（“顺便把日志级别设为 Debug”），看最终执行时是否生效。
2.  **状态漂移 (State Drift)**
    *   在长任务中，模型是否会忘记之前的约束？（例如：用户要求“全程只用英文”，第 15 步后模型突然切回中文）。

---

## 15.5 现有开源基准与选型建议

| 基准名称 | 适用场景 | 特点 | 推荐指数 |
| :--- | :--- | :--- | :--- |
| **AgentBench** | 综合能力 | 包含 OS、DB、KG、卡牌游戏等 8 个环境，覆盖面广 | ⭐⭐⭐⭐⭐ |
| **ToolBench** | 工具调用 | 侧重指令微调与 API 泛化，包含大量真实 API 的 Mock | ⭐⭐⭐⭐ |
| **GAIA** | 困难推理 | 任务看似简单但需要复杂多步推理，目前模型普遍低分，适合测上限 | ⭐⭐⭐⭐⭐ |
| **SWE-bench** | 代码工程 | 解决真实的 GitHub Issue，难度极高，适合 Coding Agent | ⭐⭐⭐ |
| **AppAgent** | 移动端操作 | 结合多模态（看图操作手机），适合车机/手机助手评测 | ⭐⭐⭐⭐ |

---

## 15.6 本章小结

1.  **评测核心**：Agent 评测关注的是“状态改变的正确性”，而不仅仅是文本输出的语义相似度。
2.  **架构循环**：必须对 **Thought (规划)**、**Action (执行)**、**Observation (理解)** 三个环节分别埋点，才能定位是“想错了”还是“手滑了”。
3.  **环境分级**：从静态 Mock 到动态 Sandbox，环境越真实，评测成本越高，确定性越低。建议 CI 阶段用 Stateless Mock，版本验收用 Stateful Sandbox。
4.  **安全底线**：Agent 具备破坏力。评测集中必须包含“诱导删除系统文件”、“诱导转账”等防御性测试用例。

---

## 15.7 练习题

### 基础题
1.  **环境构建**：你需要评测一个“订机票 Agent”。请设计一个 **Stateful Mock** 环境。
    *   *Hint*：你需要维护一个包含“航班余票”、“用户余额”、“订单列表”的虚拟数据库。当 `book_ticket` 被调用时，余票减 1，余额减少，订单增加。
2.  **指标计算**：模型 A 完成任务用了 5 步，成功率 90%；模型 B 用了 3 步，成功率 85%。在**车载语音助手**场景下，你倾向于选择哪个模型？为什么？
    *   *Hint*：考虑用户对延迟的容忍度（Time-to-Action）以及语音交互的冗长感。
3.  **Schema 验证**：给定工具 `search(query: str, limit: int = 10)`。模型输出 `search(keywords="apple", max_num="five")`。请指出其中的 3 个错误。
    *   *Hint*：参数名错误、参数值类型错误、多余参数/缺失参数。

### 挑战题
4.  **死循环检测算法**：设计一个算法，能够在评测运行时自动中断陷入死循环的 Agent，并给出“Loop Detected”的错误码。
    *   *Hint*：简单的字符串匹配不够。考虑 Action + Arguments 的哈希值序列，寻找重复子串（如 A->B->A->B）。
5.  **反思能力评测**：设计一个测试用例，强制 Agent 第一次尝试失败，考察其自我修正能力。
    *   *Hint*：Mock 环境在第一次调用正确参数时故意返回“System Busy”或“Unknown Error”，看 Agent 是复读还是重试/查文档。
6.  **思考题**：在 RAG + Agent 混合场景中（先查手册再操作），如何通过指标区分是“知识检索错误”还是“操作逻辑错误”？
    *   *Hint*：需要中间指标 Context Recall（检索到的内容是否包含答案）作为分界线。

<details>
<summary><strong>点击查看参考答案</strong></summary>

*   **题 1**：Mock Class 需包含 `__init__` 初始化状态，`book()` 方法需包含 `if balance < price: return Error` 等逻辑。
*   **题 2**：车载场景通常倾向于 **模型 B**（效率优先），前提是 85% 的成功率在可接受范围内，或者有良好的失败兜底（Ask for clarification）。多 2 步的交互在语音场景下会增加 10-20 秒的时间，体验极差。
*   **题 3**：1. 参数名 `keywords` 错误（应为 `query`）；2. 参数名 `max_num` 错误（应为 `limit`）；3. 参数值 `"five"` 是字符串（应为 `int` 如 `5`）。
*   **题 4**：维护一个滑动窗口或 Hash Set。`history = []`. 每步 `curr_hash = hash(tool_name + sorted_args)`. 如果 `curr_hash` 连续 N 次出现在 `history` 的尾部，或呈现周期性，则判定 Loop。
*   **题 5**：场景：查询天气。Step 1: Agent 调用 `get_weather(city="Beijing")`。Mock 返回：`Error: City name must be in Pinyin with strictly lowercase`。期望 Step 2: Agent 输出 `get_weather(city="beijing")`。
*   **题 6**：计算 $P(Success | Context\_Correct)$ 和 $P(Success | Context\_Wrong)$。如果前者很高但后者很低，说明 Agent 能力没问题，是 RAG 拖后腿。如果两者都低，说明 Agent 执行能力差。
</details>

---

## 15.8 常见陷阱与错误 (Gotchas)

1.  **Mock 的数据泄漏 (Data Contamination)**
    *   *陷阱*：Mock 的天气接口总是返回“25度”。模型经过微调后，记住了“天气=25度”，不再调用工具而是直接回答。
    *   *对策*：Mock 数据应在运行时随机生成（如随机温度），强制模型必须执行 `Observation` 读取步骤。

2.  **解析器的“过度溺爱” (Over-lenient Parsing)**
    *   *陷阱*：评测脚本里的 Regex 写得太强，帮模型自动修复了缺少的引号、逗号。
    *   *后果*：评测分很高，上线接真实 API 时全挂。
    *   *对策*：评测阶段应使用与生产环境一致的 Strict JSON Parser。

3.  **忽略了“什么都不做”的正确性**
    *   *陷阱*：有些任务需要 Agent 判定“无法完成”并拒答。如果评测只包含可完成的任务，模型会变成“乱操作狂”。
    *   *对策*：测试集中必须包含 10%-20% 的不可完成任务（Unsolvable Tasks），预期结果是 Agent 输出“我无法完成，因为...”。

---

## 15.9 车舱落地：驾舱一体（对话→工具→UI→导航的闭环）

车载 Agent 是多模态、实时、安全敏感的综合体。

### 15.9.1 端到端链路评测
在车机中，Agent 往往不仅调用 API，还联动 UI。
*   **场景**：“帮我找一家附近评分最高的川菜馆，并发给微信上的老婆。”
*   **工具链**：`POI Search (Map)` -> `Filter/Sort` -> `WeChat API`。
*   **评测点**：
    *   **Slot Carry-over**：地图搜到的餐厅名字/地址，是否精准透传给了微信发送接口？（常见错误：发过去的信息是“这家店”而不是具体的店名）。
    *   **多模态反馈**：Agent 操作成功后，是否在屏幕上弹出了 Toast 或 Card？评测需要结合 **GUI 截图理解**（第 10 章）来验证 UI 反馈的正确性。

### 15.9.2 影子模式 (Shadow Mode) 评测
由于无法在真实驾驶中让测试版模型随意操作车辆，推荐使用“影子模式”。
*   **实施**：在路测车上，记录驾驶员的真实语音指令和随后的真实操作（如手点屏幕导航）。
*   **回放**：在云端/离线环境中，将同样的语音输入给 Agent。
*   **对比**：比较 Agent 生成的 Action 序列与驾驶员真实操作（Ground Truth）的重合度。
    *   *注意*：Agent 可能有比人更好的解法，因此不匹配不一定代表错，需人工仲裁或规则校验。

### 15.9.3 硬件在环 (Hardware-In-the-Loop, HIL)
车载 Agent 运行在算力受限的 SoC（如高通 8295/8255）上。
*   **资源抢占评测**：当 Agent 进行复杂的 ReAct 推理时，是否导致导航掉帧？是否导致 DMS 监控延迟？
*   **指标**：
    *   **TTFT (Time to First Token)**：首字延迟。
    *   **Total Latency**：完成任务的总耗时。
    *   **CPU/NPU Usage**：推理过程中的峰值功耗。

### 15.9.4 安全护栏 (Safety Guardrails)
*   **权限隔离**：评测 Agent 是否能在驾驶状态下**拒绝**高风险指令（如“播放视频”、“打开引擎盖”）。
*   **二次确认**：对于敏感操作（如“呼叫 110”、“导航去 2000公里外”），Agent 必须触发 Confirm UI 或语音确认。评测标准是：*没有 Confirm 步骤直接调 API 判为 FAIL*。
