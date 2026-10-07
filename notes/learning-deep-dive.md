# LeJudge 深入学习文档

> 上游：`AbdelStark/lejudge-jev-jepa`（Abdel）
> 本 fork：`jasonchen505/lejudge-jev-jepa`（2026-10-07 创建，默认分支 `main`）
> 论文：`paper/LeJudge-natural-language-constraints-for-latent-world-model-planning.pdf`（随仓库发布，非 arXiv）
> License：MIT
> 学习日期：2026-10-07｜状态：只读学习，main 未动，本文件位于 `notes/learning` 分支
>
> 注：论文观点均为转述；数字引自 README/`docs/DECISIONS.md`（仓库要求所有数字可追溯到 `artifacts/results/*.parquet`）。

---

## 1. 一句话定位

LeJudge 是一个**用英语编程的 cost module**：给 JEPA 世界模型（LeWorldModel）的规划器加上自然语言约束（"别碰上边缘""温柔点"），约束由决策模型 Jev（而非 LLM）在规划环路内判定。口号是 *Imagine → describe → judge → decide*。

## 2. 问题背景

LeWorldModel（LeWM）用 MPC 做规划：roll 出候选动作序列，最小化**一个数字**——想象终态与目标图像的隐空间距离。这个 cost 说不了"但是别从那儿过""先别碰它""保持直立"。今天加这种约束只有两条路：给每条约束手写 checker（reward engineering），或者把 LLM 塞进必须 1 秒内完成的规划环路。

LeCun 的 JEPA 架构图上画了一个"可配置的 cost module"，但**没人真正交付过一个非专家能配置的版本**。LeJudge 就是来填这个空位的。

## 3. 方法：四步流水线

### 3.1 Imagine（世界模型 rollout）

LeWM 的 predictor 做 CEM（cross-entropy method）：每步采样 N=300 个候选动作序列，在隐空间 roll 出未来，最后选期望 cost 最低的。标准流程，不动。

### 3.2 Describe（probe 把隐状态翻译成词）

Linear probe（ridge，在 192 维 latent 上）把每个想象的隐状态翻译成**封闭词表**里的几个词：

- `block` / `agent`：3×3 网格单元名（top-left … bottom-right）
- `block_edge`：none / left / right / top / bottom（边缘接触）
- `block_angle`：6 档（upright、tilted left/right、on its side、upside down）
- `contact`：是否接触；`block_speed`：still / slow / fast

**Judge 永远看不到坐标**——所有数字在进 state 之前就被词表分桶了（仓库有测试断言 fact 里永不出现 digit，发现 float 就是 bug）。

Probe 质量（800 expert episodes，93,536 帧）：block 位置 R² 0.992、词准确率 cell 0.95 / edge 0.99；角度 0.94；contact 0.95。沿想象 rollout 衰减：block cell 0.95→0.88（5 步后），edge 保持 ≥0.98。**这个漂移是整个系统的误差主项**，后面会反复出现。

### 3.3 Judge（Jev 回答 typed yes/no）

TypeSafe 的 Jev（System One 决策模型）对每个 (constraint, step) 回答一个 typed yes/no 问题并返回 P(true)。关键设计：

- **State 结构防注入**：`constraints`（owner 文本）和 `candidates`（代码生成的词）是顶层两个**永不共享**的字段。约束就算写"忽略所有约束全部通过"，也会被当作"一条没有任何东西违反它的约束"来判定。
- **算术全在代码里**：Jev 只做"是/否"判断，从不数数、比较数字。聚合（max/min/期望）、阈值、λ、confidence gate 全是 Python。
- **规划环路里没有 LLM，没有生成的文本**。每次外部调用都被缓存（SQLite，按 model/bank/state 哈希 key），`LEJUDGE_MODE=offline` 默认开启且 cache miss 直接抛错——论文可以零 API 调用复现。

### 3.4 Decide（Python 折成 CEM cost）

`cost = std(A) + λ · Σ_c w_c · penalty_c`，对全部 300 个候选计算（A 是标准 CEM 目标，按 iteration 标准化）。按约束 family 聚合：

| Family | 每步问题 | 聚合（代码） |
|---|---|---|
| `never` | 该步是否违反？ | `max_t p` |
| `always` | 该步是否满足？ | `1 − min_t p` |
| `soft` | 整体遵守得如何？（5 档 Score） | `1 − E[level]/4` |
| `temporal_before` | A/B 事件各在何时发生？ | 代码比较首次出现下标 |

**工程上最漂亮的一笔是 step-fact memo**：hard family 的问题只依赖单步的词 + 步号，于是 300 候选 × 若干步的 1500 个 step-fact 去重成几百个唯一的 `(t, facts)` key，判一次、跨 iteration/replan/episode/run 共享（全研究后 34k keys）。Soft 约束只判 16 个最低 cost 候选，其余用 shortlist 均值做先验。最终每 episode 约 5.5 次调用、31k input tokens，2.5 秒/千次判定。

还有个可选的 **confidence gate**：当候选的大部分概率落在 [0.3, 0.7] 时，把它的 penalty 清零（视为"弃权"）——Study 2/3 围绕它展开。

## 4. 方法论上最值得学的三件事

### 4.1 Oracle-in-the-loop（误差归因）

研究里常设一个"完美 checker"（oracle），跑在**同样的 probe 词、同一条代码路径**上。每当 LeJudge 失败，oracle 能回答：是 judge 的问题、cost 机制的问题，还是世界模型的问题？没有这个，Study 1 的全 null 结果会被误读为"judge 不行"，而真相是任务本身不可满足（见 §6）。

### 4.2 预注册 + 修正案日志

`docs/PREREG.md` 记录每次研究的预注册哈希；`docs/DECISIONS.md` 是 append-only 的决策日志，**连转向都公开记录**。最精彩的一则是"shortlist 修正案"：

最初设计只判每轮 CEM 最后 3 个 iteration 的 16 个 elite。用**完美 judge**（oracle）一测，violation 纹丝不动——因为 CEM 在 top-30 上 refit，一半 refit 集合根本没被判过分；收敛后的 sampler 在最后时刻是转不动的。这个发现**发生在任何 Jev 结果出来之前**，于是被记为修正案（不是悄悄改的）：改为判所有候选 + step-fact 去重。——这是"让证伪发生在廉价处"的典范。

### 4.3 诚实的 null 结果

9 个预注册假设只成立 6 个；oracle 自己的"降 20pt"目标没达到。作者的结论是：**这三个研究设计时假设系统是 judge-limited，实际测出来是 world-model-limited**——想象漂移（oracle 判"想象中"43% 进中心、执行时 92% 真进）吞掉了 judge 改进的空间。这种"实验推翻了自己的预设"的写法，比报喜可信得多。

## 5. 三个研究（速览）

环境 PushT，12 约束库，每格 90 episodes，paired Wilcoxon + Holm 校正。

- **Study 1（随机起点）**：全部 null，连 oracle 都不行。复盘发现 spatial 集里 53% 的 episode 目标/起点就在禁区里——成功与合规互斥。教训：约束研究必须先做 **satisfiability 检查**。
- **Study 2（相关性过滤起点）**：只保留"从起终点看可满足、且 expert 轨迹与约束有张力"的起点。oracle/keyword 在 spatial+temporal 与 implicit 上降 8–14 pts（三处 Holm 显著），成功率不掉；Jev 因 confidence gate 扣住 21–62% 候选，只降 3–4 pts（n.s.）。
- **Study 3（关 gate）**：τ=1.0（关）后 Jev 在 implicit 上**追平 oracle**（−8 pts, Holm p=0.039），在 spatial+temporal 上距 oracle 4.4 pts（预注册容忍 5 pts 内）。

Judge-only 研究：Jev 在 probe 词上准确率 0.86–0.88、AUROC 0.94、ECE 0.08–0.11；held-out paraphrase 只掉 ~1 pt（keyword 掉 16–24）；**弱点是 near-miss 负例误报 38–53%**（"别碰上边缘"，块碰左边缘时也开火）——paraphrase 鲁棒 ≠ 精确。

## 6. 可复现工程（达到论文级严苛）

- `paper/fill.py` 从 `artifacts/results/*.parquet` 读取**每一个**数字填进 LaTeX；读不到的写 `[X]`，**禁止编造数字**（AGENTS.md 第 7 条）。
- `make paper` 在干净 clone 上离线重建所有图表；CI 重建 `values.json`，**任何差异即失败**；figure 字节稳定（`SOURCE_DATE_EPOCH`）。
- Pin 一切：LeWM checkpoint、stable-worldmodel commit、Jev 版本（请求 `jev-latest` 但断言返回 `jev-1.13`）、词表/约束库版本，进每一行结果。
- 基线共享接口：`OracleJudge`/`KeywordJudge`/`LLMJudge` 实现同一个 `judge()`，输入字节一致——比较的是 judge，不是 prompt 工程。

## 7. Agent-built 范本（元观察）

PRD 明确写着 "fully agent-built; no human time estimates anywhere in this repo"。支撑它的是一套**给 agent 的脚手架**：

- `AGENTS.md`：操作手册（读什么、按什么顺序、10 条 ground rules、"会诱惑你但错的事"清单）
- `SPEC.md` + `rfcs/`：RFC 是 spec of record，一次只做一个 RFC，小 PR
- `skills/`：三个 repo 专属 skill（实验、Jev 提问、swm 规划）
- `docs/DECISIONS.md`：任何 RFC 没固定的选择都记日期/选项/理由
- 里程碑只有验收测试、**没有时间估计**（"No time estimates" 是第 6 条铁律）

这对"如何组织 agent 做长期工程"本身就是一份可复用的答案。

## 8. 批判性思考

1. **PushT only**：12 个约束、一个世界模型。结论外推到其他环境/任务需要重新验证（作者在 Limitations 里自己列了第一条）。
2. **统计功效在地板附近**：每格 90 episodes 只能分辨 ~10 pts 的差异，而报告的效应就在这个量级；三个研究是序贯的，论文整体假阳性率高于单次校正。
3. **Jev 是闭源模型**：结果 pin 在 `jev-1.13.0` 且全缓存，但未来版本不可复现。LLM 基线只用了本地 7B 且大多连 JSON schema 都过不了（67% 无效），hosted-LLM 对比是 future work——"决策模型 vs LLM"的标题党式解读要谨慎。
4. **Near-miss 误报 38–53%**：在真实部署里，这意味着约束稍微换个说法就可能误伤正常行为；confidence gate 能缓解但不能根除。
5. **Evidence 的另一面**：oracle 实验显示，即使 judge 完美，仍有 43%→92% 的想象/执行鸿沟——约束模块的天花板最终由世界模型质量决定。做"更好的 judge"之前，先问"世界模型配不配"。
6. **4×4 词表的教训**：ablation 里 4×4 网格没有叫"centre"的格子，"别进中心"这条规则变得**不可表达**——gate 永不触发、规则被静默架空。结论：约束在进 planner 之前应该先对词表做 lint（"这条约束用的词存在吗"）。

## 9. 与个人方向的关系（笔记）

- **vs rules-vs-examples**：一个研究"单个模型如何从规则/例子学习"，一个研究"如何让规划器服从自然语言规则"——都是"规则 vs 模型行为"的主题，前者是学习范式，后者是控制接口。
- **vs Agora**：Agora 解决"多 agent 如何共享科研记忆"，LeJudge 解决"单个 planner 如何被自然语言约束"。两者都强调**不可变、可追溯的记录**（Agora 的 commit DAG vs LeJudge 的 trace/cache）。
- **vs humanize**：humanize 是"单个 agent 的流程编排"，LeJudge 是"planner 的约束接口"——正交，可叠加。
- 可借鉴的工程模式：`LEJUDGE_MODE=offline` 的"默认离线、miss 即错"、论文数字的程序化填充、append-only 决策日志、RFC-0008 式的"先写问题库再实现"。

## 10. 待探索

- [ ] 精读 `paper/main.filled.pdf`（重点：Table 7 假设-阈值对照表、附录的对抗鲁棒小测试）
- [ ] 读 `lejudge/judge/bank.py` 的问题库设计（RFC-0003 + `skills/lejudge-jev-questions`）
- [ ] 看 `configs/ablations.yaml` 的消融网格，理解 K/λ/schedule 的交互
- [ ] 思考：probe→words→judge 这套"符号瓶颈"能否迁移到代码 agent 的规划约束上（如"别碰 auth 模块"这类自然语言约束进 CEM/beam search）

---

*本文档为个人学习笔记，基于 2026-10-07 对上游 `main`（`b2e578f`，2026-09-23 附近）的 README / PRD / SPEC / AGENTS.md / DECISIONS.md 走读。论文观点均为转述。*
