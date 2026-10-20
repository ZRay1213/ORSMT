# Reproduction and interfaces

### Run the tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

### Create a repair plan

The following command freezes a deterministic six-model repair plan. It does not execute generated programs, load a GPU model, or call an LLM.

```bash
python ops/run_routed_milp_repair.py \
  --out /tmp/orsmt-plan \
  --mode routed \
  --plan \
  --reviser-model Qwen2.5-14B-Instruct
```

The output reports **3,034 problems**, **six models**, **18,204 assigned outputs** and `gpu_loaded: false`. The plan is saved as `/tmp/orsmt-plan/PLAN.json`. Use a fresh output directory when changing the configuration.

### Optional native adapters and Gurobi replay

```bash
python -m pip install -e '.[native,gurobi]'
python examples/brownie_trace.py --native-gurobi
```

Gurobi-backed execution requires a valid Gurobi installation and license. Actual routed repair also needs an explicitly configured model endpoint; inspect `python ops/run_routed_milp_repair.py --help` for runner options. [`requirements-repro.txt`](../requirements-repro.txt) records experiment-host versions, while [`pyproject.toml`](../pyproject.toml) provides compatible version ranges for installation.

[`ops/runtime/capture_gurobi.py`](../ops/runtime/capture_gurobi.py) runs generated programs in a child process and captures the model at `optimize()`. Process isolation is not a security sandbox: run untrusted generated code inside an external container or equivalent sandbox.

## 🧩 Interfaces and documentation

| Interface | Purpose |
| --- | --- |
| [`ORSMTWorkflow`](../src/orsmt/workflow.py) | Artifact translation, SMT diagnostics and conditional repair |
| [`Verifier`](../src/orsmt/verify.py) | Feasible regions, objective relations and returned reports |
| [`diagnose_repair`](../src/orsmt/repair_workflow.py) | Separate discrepancy detection from the choice of repair target |
| [`repair_case`](../src/orsmt/repair_engine.py) | Deterministic or source-guided repair followed by verification |
| [`emit_gurobi`](../src/orsmt/milp_codegen.py) | Generate a complete fixed-instance Gurobi program from a selected model |

Further guides, in Chinese: [pipeline and diagnostics](../docs/pipeline_and_taxonomy_zh.md) · [repair workflow](../docs/repair_workflow_zh.md) · [translation and source frontend](../docs/translator_source_frontend_zh.md).

The frozen scope covers linear programs and pure or mixed-integer linear programs with a single linear objective. Scope selection uses release metadata, reference model structure or reviewed source evidence. Candidate correctness, execution success, checkpoint size and verification outcomes do not select problem IDs.

<details>
<summary><strong>Repository layout</strong></summary>

```text
src/orsmt/                         Typed representation, translation, SMT checks, repair and metrics
tests/                             Unit and integration tests
ops/                               Core audit and repair entry points
examples/                          Runnable examples
docs/                              Pipeline, repair and frontend guides
assets/                            README workflow illustration
data/milp_benchmark_v3/            Frozen 3,034-problem scope and provenance
data/paper_core/candidates/        Six candidate ledgers
data/paper_core/verification/      Six frozen model–code verification ledgers
scripts/verify_release.py          Offline integrity and denominator checks
configs/paper_core.json            Six-model reproduction configuration
```

</details>


## Frozen pre-repair release inventory

The frozen core audit covers **3,034 MILP problems from 12 benchmark sources**, evaluated with six Qwen2.5-Instruct checkpoints. Every model has the same 3,034 assigned problems; every assigned output stays in the denominator.

| Qwen2.5-Instruct checkpoint | Certified equivalent | Refuted | Remaining | Assigned outputs |
| --- | ---: | ---: | ---: | ---: |
| 1.5B | 760 | 974 | 1,300 | 3,034 |
| 3B | 426 | 414 | 2,194 | 3,034 |
| 7B | 1,349 | 239 | 1,446 | 3,034 |
| 14B | 1,346 | 103 | 1,585 | 3,034 |
| 32B | 1,932 | 86 | 1,016 | 3,034 |
| 72B | 2,054 | 89 | 891 | 3,034 |
| **Total** | **7,867** | **1,905** | **8,432** | **18,204** |

These are the **frozen model–code audit outcomes before repair**. “Remaining” includes unresolved or unsupported pairs. The [release manifest](../data/paper_core/manifest.json) records the expected counts and SHA-256 hashes; the [data guide](../data/README.md) explains provenance and the ledger layout.

The repository contains the core implementation and this six-model data release. External repair baselines, host-system comparisons, the 0.5B checkpoint and historical experiment directories are outside this release.
