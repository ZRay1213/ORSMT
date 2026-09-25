# Translator 与 NL→数学模型检查

已实现的入口是 `ORSMTWorkflow`。自然语言检查采用 LLM＋SMT：LLM 先独立提取需求并逐条引用原文，再以另一次调用复核完整性与解释；系统冻结该解释，SMT 分别检查数学模型、程序模型是否符合它。LLM 可以解释反例，但不能修改 SMT 判定。`CERTIFIED` 的范围始终是**经复核的形式化解释**；它不等于自然语言理解已获数学证明。同一模型的两次调用可能存在相关错误，需另做人工标注评估。

## 实际输入输出

| 步骤 | 输入 | 输出与准入条件 |
|---|---|---|
| 1. 需求提取 | 原始 `question: str`、实例 `data: str`；不提供候选模型、代码或标准答案 | `formulation`、`requirements[]`、`ambiguities[]`、`missing_information[]`、`complete: bool`。变量、约束、目标均必须有原文引文；系统根据唯一精确引文计算 Unicode 字符位置 |
| 2. 独立复核调用 | 同一原文、提取结果、结果哈希 | `approved`、`complete`、`issues`、`rejected_requirement_ids`、`uncovered_source_spans`、`reviewed_sha256`；冻结成 `orsmt-source-product-v1`，保存请求、响应、模型身份、tokens、耗时和哈希 |
| 3. 数学模型转写 | dict/JSON、LP/MPS、LaTeX、文本，或已经构造的原生模型对象 | `Translation(model=ModelIR, format, mechanism, original, details)`；机械提取与 LLM 转写分别标记；原文和模型有独立哈希 |
| 4. 程序模型捕获 | 现有 Gurobi 执行捕获，或返回原生对象的 `build_model()` 程序 | `ModelIR`、提取元数据；`capture_builder` 在限时子进程执行宿主批准的程序，不构成恶意代码安全沙箱 |
| 5. 需求一致性检查 | 冻结的 source product、候选 `ModelIR`、可选显式变量对应 | `orsmt-source-check-v1`：双向可行域、目标偏好、目标数值，原文条款定位、反例、结构差异、taxonomy。部分需求只能得到局部结果，不能整体 `CERTIFIED` |
| 6. 模型—代码—结果检查 | 分别提取的 written/program、可选 `ResultRecord` | 既有 `diagnosis`、`feedback`，加 `source_alignment`；模型代码共同错误也可以被源需求检查发现；冗余仍逐条统计，并与语义错误区分 |
| 7. 反馈解释 | 原需求＋SMT 证据 | 独立的 `llm_explanation`；保存原判定和证据哈希，要求引用真实原文 |
| 8. 修复与复检 | 完整通过复核的 source product、written/code、宿主执行器 | 确定性生成受支持的线性模型代码，重新捕获、求解并检查；`repair_from_source` 再对最终 formulation 检查；修复接受范围明确标为 `reviewed_formal_interpretation_only` |

哈希用于可追溯性和意外修改检测，不是来源真实性的数字签名。显式变量对应由宿主提供；同名默认对应是明确假设。修复入口要求源模型和实际模型已采用一致变量名称；映射后的检查可用，但不自动推断变量业务含义。

## 支持范围

- **Pyomo**：已构造的 ConcreteModel/Block，线性表达式，连续/整数/二元区间域、固定变量、活动约束、范围约束、单目标与目标常数。非线性、GDP、逻辑、SOS、非区间变量域等明确拒绝。
- **PuLP**：LpProblem 变量域、线性约束、目标方向及常数；SOS 明确拒绝。验证版本范围为 3.x。
- **OR-Tools Linear Solver**：从 MPModelProto 提取变量、线性范围约束、目标偏移；一般约束与二次目标明确拒绝。
- **OR-Tools CP-SAT**：有限整数域（包括不连续取值区间）、线性约束、布尔或/与/基数/XOR、enforcement、assumptions、all-different、max/abs。通过有限边界推导 big-M，辅助变量做存在量词投影；不把辅助变量当作业务变量。当前不支持 scheduling/interval、table、circuit、整数乘除等，返回拒收，不能报通过。转换默认最多 100,000 行。
- **LaTeX/自由数学文本**：`MathematicalTranscriber` 直接调用显式配置的 OpenAI-compatible 服务。保留模型原有错误；不补猜变量域或缺失数据；有歧义、截断、身份不匹配或不支持的结构均拒收。
- **LP/MPS/Gurobi、JSON/IR**：保留原有支持。JSON 使用 Decimal 再转精确分数，重复 key 和非有限数字拒收。

原生 API 的浮点系数按其十进制表示解释，不声称恢复浮点化之前的原始有理数。CP-SAT 降低后的约束数量与原生约束数量不同；对降低后行的冗余统计不能直接当作原生业务条款的冗余率。

## 调用示例

```python
from orsmt.language_frontend import ChatJSONClient, SourceGrounder
from orsmt.source_check import SourceChecker
from orsmt.translator import ModelTranslator
from orsmt.workflow import ORSMTWorkflow

client = ChatJSONClient(endpoint, served_model_name, max_tokens=3072)
reviewer = ChatJSONClient(review_endpoint, review_model_name, max_tokens=3072)
translator = ModelTranslator.from_client(client)
checker = SourceChecker(SourceGrounder(client, reviewer), translator=translator)
workflow = ORSMTWorkflow(translator=translator, source_checker=checker)

# 先冻结源需求；此阶段不向模型发送候选模型或标准答案。
source = workflow.prepare_source(question, data=rendered_instance_data)
report = workflow.check(case_id=case_id, written=latex_text, written_format='latex',
                        program=pyomo_model, source_product=source)
# 完整源解释通过复核后才可选择以它为条件参考进行修复。
repair = workflow.repair_from_source(source_product=source, case_id=case_id,
                                    written=latex_text, written_format='latex',
                                    code=code, execute=host_native_executor)
```

`ChatJSONClient` 不暗中切换模型或重试。端点需要 HTTPS 或本机 HTTP；API key 可显式传入，不写进调用记录。每次逻辑调用保留 request/response、实际模型身份及 token 用量；来源判定和数学文本转写的调用成本分别记录。

## 可复现实验入口

`ops/run_source_frontend.py` 接受以下输入 JSON：

```json
{
  "id": "benchmark:sample_id",
  "question": "原始自然语言需求",
  "data": "实例数据，可省略",
  "formulation": "数学模型文本，或 JSON 模型对象",
  "formulation_format": "text",
  "code": "待执行的原始 Gurobi 程序"
}
```

运行时显式给 `--endpoint`、`--model`、可选独立 `--review-endpoint`/`--review-model`，以及 `--input`、`--out`。`--explain` 启用反馈解释，`--repair` 启用修复。`--reuse-product` 可复用已冻结的源解释，避免反复提取造成参考漂移。`--mapping` 文件形如 `{"written":{"源变量":"数学模型变量"},"program":{"源变量":"代码变量"}}`；有映射的实验当前只做检查，不自动修复。

输出包括 `input.json`、`source_product.json`、`written_translation.json`、`program_capture.json`、`checks.json`、可选 `explanation.json`/`repair.json`、`llm_calls.jsonl`、`summary.json`、`MANIFEST.json`。失败也保留已产生的记录；没有数据的指标不填成零成功率。

测试集覆盖原生模型到手写可信 IR 的双向等价、CP-SAT 所有有限组合的原生求解器对照、NL 解释拒绝/部分覆盖/哈希修改、共同错误和修复复检，以及真实 HTTP 协议的截断、模型身份、重复 key 测试。HTTP 协议测试使用 fixture 服务，不算真实 LLM 实验；真实调用记录单独保存。完整 benchmark 的准确率提升、定位质量、修复成功率与性能扩展实验尚需在固定样本、预算和人工来源标注下测量，不能由集成例子代替。
