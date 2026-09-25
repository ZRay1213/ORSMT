# ORSMT

ORSMT verifies and repairs the semantic relations among a written optimization model,
the model instantiated by solver code, and the returned solution report. It translates
supported artifacts into an exact typed representation, uses SMT queries to certify or
refute their relations, and rechecks repaired artifacts before adoption.

This repository contains the core implementation and the frozen six-model data used for
the main model--code audit. It intentionally excludes the 0.5B checkpoint, external repair
baselines, host-system comparisons, and historical experiment directories.

## Included population

- 3,034 MILP problems from 12 benchmark sources.
- Six Qwen2.5-Instruct checkpoints: 1.5B, 3B, 7B, 14B, 32B, and 72B.
- 18,204 assigned outputs in total.
- 7,867 certified-equivalent model--code pairs.
- 1,905 refuted model--code pairs.
- 8,432 unresolved or unsupported pairs retained in the denominator.

Every candidate and verification ledger contains exactly the same 3,034 problem IDs.
`data/paper_core/manifest.json` records the expected counts and SHA-256 hashes.

## Repository layout

```text
src/orsmt/                         Core typed IR, translation, SMT checks, repair, and metrics
tests/                             Unit and integration tests
ops/                               Core audit and repair entry points
examples/                          Small runnable examples
docs/                              Pipeline, repair, and frontend documentation
data/milp_benchmark_v3/            Frozen 3,034-problem scope and provenance
data/paper_core/candidates/        Six candidate ledgers
data/paper_core/verification/      Six frozen model--code verification ledgers
scripts/verify_release.py          Offline integrity and denominator check
configs/paper_core.json            Six-model reproduction configuration
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For native modeling-library adapters and Gurobi replay:

```bash
python -m pip install -e '.[native,gurobi]'
```

`requirements-repro.txt` records the package versions on the experiment host. The
ordinary package metadata keeps compatible version ranges so the core can also run in
newer environments.

Gurobi-backed execution additionally requires a valid local Gurobi installation and
license. SMT-only tests use Z3 and do not require Gurobi.

## Reproduce the core audit checks

Verify the frozen data, hashes, IDs, and six-model totals:

```bash
python scripts/verify_release.py
```

Run the complete test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Run a small SMT example:

```bash
PYTHONPATH=src python examples/brownie_trace.py
```

Create a deterministic six-model repair plan without executing programs or calling an
LLM:

```bash
PYTHONPATH=src python ops/run_routed_milp_repair.py \
  --out /tmp/orsmt-plan \
  --mode routed \
  --plan \
  --reviser-model Qwen2.5-14B-Instruct
```

The plan must report 3,034 problems, six models, and 18,204 assigned outputs.

## Core interfaces

- `orsmt.workflow.ORSMTWorkflow` translates artifacts, obtains SMT diagnostics, and
  connects conditional repair.
- `orsmt.verify.Verifier` checks feasible-region inclusion, objective preference and
  value relations, and returned reports.
- `orsmt.repair_workflow.diagnose_repair` separates a detected discrepancy from the
  decision about which artifact should change.
- `orsmt.repair_engine.repair_case` executes deterministic or source-guided repair and
  re-verifies the resulting artifacts.
- `orsmt.milp_codegen.emit_gurobi` generates a complete fixed-instance Gurobi program
  from the selected mathematical model.

An LLM may propose a translation or repair, but it does not issue a verification verdict.
Unsupported translation, ambiguous correspondence, timeouts, and solver unknowns remain
explicitly unresolved.

## Runtime capture

`ops/runtime/capture_gurobi.py` provides the host capture callback used by the public
runner. It executes generated programs in a child process and captures the Gurobi model
at `optimize()`. Process isolation is not a security sandbox; untrusted generated code
must be run inside an external container or equivalent sandbox.

## Scope

The frozen scope covers linear programs and pure or mixed-integer linear programs with a
single linear objective. Scope selection is based on release metadata, reference model
structure, or reviewed source evidence. Candidate correctness, execution success,
checkpoint size, and verification outcomes are not used to select problem IDs.
