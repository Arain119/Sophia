# 预训练评测体系

本项目训练的是未做指令微调的 base decoder。评测首先回答“下一个 token 的概率建模是否持续变好”，其次才观察 raw continuation 是否开始形成稳定的语言、代码和知识结构。不能把少量 greedy 续写当成 chat 模型能力，也不在中途用未经推导的阈值决定是否更换训练轨迹。

## 评测层级

| 层级 | 资产与规模 | 目的 | 运行时机 |
| --- | --- | --- | --- |
| 稳态监控 | 训练 val loss、grad norm、吞吐、QK-Clip telemetry | 发现数值/吞吐/注意力异常 | 每个 update |
| checkpoint 诊断 | `generation_quality.jsonl` 的固定分层子集（64/100/200）、raw greedy、同一 seed | 比较语言、数学、代码、中文和重复趋势 | 每 1000 step 或需要时 |
| 中期能力 | 完整 2000 条 raw generation、受限数学/代码执行、C-Eval/CMMLU 子集、val/test PPL | 判断能力是否与 loss 同步、定位域缺口 | 中期 checkpoint |
| release 证据 | 全量 2000 generation、全量受限执行、固定 C-Eval/CMMLU、每来源 val/test PPL、去污染和稳定性绑定 | 形成可复核的最终证据包 | 20B checkpoint |

## 指标解释

- **Loss/PPL 是主指标。** 使用 tokenizer 后的 held-out token，按来源分别报告 loss/PPL，避免混合来源的平均值掩盖单一域退化。
- **Raw generation 只测 base 行为。** 固定 `do_sample=0`、seed 和 `max_new_tokens`；报告答案匹配、4-gram 重复、prompt echo、空输出、首 token EOS 和语言匹配，并按语言与 capability 分组。
- **数学和代码必须可执行。** 数学采用 exact match；代码只在 AST 白名单沙箱中执行固定输入，禁止把“看起来像代码”计为正确。
- **中文必须是真 UTF-8。** 模型输入 suite、prompt 和 tokenizer decode 全部按 UTF-8 读写；任何 `�`、典型 mojibake（如 `璇风`）或无法 round-trip 的输入样本先剔除。当前正式 `generation_quality.jsonl` 和 C-Eval/CMMLU 输入已通过 UTF-8 读取，中文样本不是乱码。去污染 signature 是不送入模型的源文本指纹，保留上游数据中的少量替换占位符不会影响生成评测。
- **长上下文单独报告。** 4K 是训练合同；可在 1K/2K/4K 的同一 held-out 文本上报告 loss 和跨段 continuation，但不把未训练的外推能力混入正式分数。

## 轨迹比较

`ml/tooling/scripts/eval_generation_quality.py` 的 `--max_cases` 只用于诊断抽样：它按 `language × capability` 分层、使用固定 hash seed、从不重复 case。`0` 仍表示全量 2000 条，release gate 不接受子集。

多个 checkpoint 的报告可由 `ml/tooling/scripts/summarize_pretrain_eval_trajectory.py` 合并。该工具保留每份报告的路径、SHA-256、suite/manifest 指纹，并明确标记缺失报告；它只展示趋势，不选择 checkpoint、不引入能力阈值。

## 去污染与可比性

评测资产在 release protocol 中以 SHA-256 固定，generation signatures 和 benchmark signatures 不进入训练数据。比较不同 step 时必须使用同一 suite、同一 tokenizer、同一 decode 参数和同一数据 manifest。改变其中任意一项就应生成新报告，不与旧轨迹直接拼接。

## 当前判断规则

训练中：loss 持续下降且 finite、PPL 没有域级反常、生成健康度不恶化，就继续正式轨迹；少量 greedy 样本变好或变坏都只是诊断信号。训练完成后再用完整 release evidence 判断 base 模型是否值得进入后续 SFT/蒸馏，而不是在 20B 尚未完成时提前中断。
