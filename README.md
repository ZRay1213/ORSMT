<div align="center">

# ORSMT

### Semantic Verification and Repair in LLM-Based Optimization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Problems: 3,034](https://img.shields.io/badge/Problems-3%2C034-146B67)](data/README.md)
[![Assigned outputs: 18,204](https://img.shields.io/badge/Assigned_outputs-18%2C204-146B67)](data/paper_core/manifest.json)

[Quick start](#-quick-start) · [How it works](#-how-it-works) · [Released results](#-released-results) · [Reproduction](#-reproduce-the-core-audit) · [Citation](#-citation)

</div>

A written optimization model and its solver program can agree with each other and still omit the same requirement. They can also return the same objective value while describing different feasible decisions. **ORSMT** checks the semantic relations among source obligations, written mathematics, the model instantiated by solver code, and the returned report. It produces certificates or concrete counterexamples, routes repair, and checks repaired artifacts before adoption.

**Paper:** *ORSMT: Unleashing the Potential of Satisfiability Modulo Theories for Semantic Verification and Repair in LLM-Based Optimization*<br>
**Authors:** Rui Zhang¹, Chenyu Zhou², Jianghao Lin², Yaoxin Wu³, Dongdong Ge², Wen Song¹<br>
¹ Shandong University · ² Shanghai Jiao Tong University · ³ Eindhoven University of Technology

## 🔎 A short example

> **Request:** Make eight items, with brownies accounting for at least 40% of the total.
>
> **Written model and solver model:** Both encode `5 × brownies − 2 × other ≥ 0`.
>
> **Returned solution:** Three brownies and five other items.

The written and solver models agree, and the returned point satisfies both. But three brownies out of eight is only **37.5%**. The source requirement is `3 × brownies − 2 × other ≥ 0`, which this point violates.

The runnable [brownie example](examples/brownie_trace.py) checks each relation separately:

| Check | Outcome | Evidence |
| --- | --- | --- |
| Written model preserves the 40% source obligation | `REFUTED` | `brownies = 3, other = 5` violates the requirement |
| Written model and runtime model agree | All implementation checks `CERTIFIED` | Both encode the same weaker constraint |
| Returned point, value and optimality agree with the runtime model | All reporting checks `CERTIFIED` | The report is valid for that runtime model |
| End-to-end assessment | `REFUTED` | Agreement between artifacts does not establish source fidelity |

## 🧠 How it works

[![ORSMT workflow: independently obtained source obligations, written mathematics, solver code and reports enter a common exact representation. SMT checks certify relations or return counterexamples. Reconstruction or source-guided revision is followed by verification before adoption.](assets/orsmt-overview.png)](assets/orsmt-overview.png)

1. **Translate independent artifacts.** Structured model declarations, supported native models, and LP/MPS inputs enter a typed representation of decisions, domains, constraints and objectives. Mathematical prose can use an explicit LLM translation callback. Host execution captures the model actually instantiated by solver code.
2. **Check semantic relations.** SMT queries compare feasible regions, objective preferences and values, and returned reports. Source obligations are checked when a formal source representation is available.
3. **Diagnose and repair.** Counterexamples identify implicated constraints or domains. A selected artifact supports reconstruction, or a source-guided proposal revises the formulation.
4. **Recheck before adoption.** Reconstructed or revised artifacts pass the required relation checks. Unsupported translation, ambiguous correspondence, timeouts and solver unknowns remain explicitly unresolved.

An LLM can propose a translation or repair. **Verification verdicts come from the formal checks.**

## 🚀 Quick start

### Install

Use **Python 3.10+**. The commands below use Bash on Linux, macOS, or Git Bash on Windows.

```bash
git clone https://github.com/ZRay1213/ORSMT.git
cd ORSMT
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The core requires Z3. The example below needs **no API key, GPU or Gurobi license**.

### Run one case

```bash
python examples/brownie_trace.py > brownie_report.json
```

Open `brownie_report.json`. Its `records` contain individual verdicts, witnesses, implicated constraints and replay evidence. The top-level `profile.end_to_end` is `REFUTED`; the record with `claim_id` equal to `share_at_least_40_percent` contains the distinguishing point shown above.

### Verify the released data

```bash
python scripts/verify_release.py
```

Expected summary: `status: PASS`, **3,034 problems**, **six models**, and **18,204 assigned outputs**. This check validates file hashes, problem IDs and aggregate counts without rerunning model generation or calling an LLM.

## 📊 Released results

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

These are the **frozen model–code audit outcomes before repair**. “Remaining” includes unresolved or unsupported pairs. The [release manifest](data/paper_core/manifest.json) records the expected counts and SHA-256 hashes; the [data guide](data/README.md) explains provenance and the ledger layout.

The repository contains the core implementation and this six-model data release. External repair baselines, host-system comparisons, the 0.5B checkpoint and historical experiment directories are outside this release.

## 🧪 Reproduce the core audit

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

Gurobi-backed execution requires a valid Gurobi installation and license. Actual routed repair also needs an explicitly configured model endpoint; inspect `python ops/run_routed_milp_repair.py --help` for runner options. [`requirements-repro.txt`](requirements-repro.txt) records experiment-host versions, while [`pyproject.toml`](pyproject.toml) provides compatible version ranges for installation.

[`ops/runtime/capture_gurobi.py`](ops/runtime/capture_gurobi.py) runs generated programs in a child process and captures the model at `optimize()`. Process isolation is not a security sandbox: run untrusted generated code inside an external container or equivalent sandbox.

## 🧩 Interfaces and documentation

| Interface | Purpose |
| --- | --- |
| [`ORSMTWorkflow`](src/orsmt/workflow.py) | Artifact translation, SMT diagnostics and conditional repair |
| [`Verifier`](src/orsmt/verify.py) | Feasible regions, objective relations and returned reports |
| [`diagnose_repair`](src/orsmt/repair_workflow.py) | Separate discrepancy detection from the choice of repair target |
| [`repair_case`](src/orsmt/repair_engine.py) | Deterministic or source-guided repair followed by verification |
| [`emit_gurobi`](src/orsmt/milp_codegen.py) | Generate a complete fixed-instance Gurobi program from a selected model |

Further guides, in Chinese: [pipeline and diagnostics](docs/pipeline_and_taxonomy_zh.md) · [repair workflow](docs/repair_workflow_zh.md) · [translation and source frontend](docs/translator_source_frontend_zh.md).

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

## 📖 Citation

For this code and data release:

```bibtex
@misc{zhang2026orsmt,
  title={ORSMT: Unleashing the Potential of Satisfiability Modulo Theories for Semantic Verification and Repair in LLM-Based Optimization},
  author={Rui Zhang and Chenyu Zhou and Jianghao Lin and Yaoxin Wu and Dongdong Ge and Wen Song},
  year={2026},
  howpublished={Code and data repository},
  url={https://github.com/ZRay1213/ORSMT}
}
```

## ✉️ Contact

Rui Zhang: [202300190245@mail.sdu.edu.cn](mailto:202300190245@mail.sdu.edu.cn).
