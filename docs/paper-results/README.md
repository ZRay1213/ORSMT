# Homepage experimental results and examples

These figures present results from the ORSMT manuscript used to prepare the public
arXiv version on 2026-10-20. The source table snapshots are included here so readers
can inspect the underlying values and distinguish the evaluation protocols.
`values.json` is generated directly from these snapshots.

## Full-population pipeline comparison

Source: [manuscript full-population table](source_tables/comparison_compact.tex),
label `tab:nativepipelines`.

| Pipeline | Correct answer (%) | Model–code match (%) | Correct and matched (%) |
| --- | ---: | ---: | ---: |
| Original | 27.1 | 43.2 | 25.2 |
| ReLoop | 31.7 | 45.5 | 26.1 |
| Opt-Verifier | 38.9 | 25.3 | 20.9 |
| ORSMT | 40.2 | 88.2 | 40.2 |

This is the full pipeline comparison with protected selection over six generators
and 3,034 assigned problems per generator. Model–code match and correct-and-matched
rates use all 18,204 assigned outputs. Correct-answer rates use the 18,192 outputs
with references. The homepage plots only the two metrics sharing the full denominator.
The full table retains the original correct outputs and newly correct outputs.

The [per-benchmark reconstruction table](source_tables/matrix.tex) reports a
separate before/after experiment **before protected selection**. Its per-generator
rates should not be substituted for the protected-pipeline rates above.

## Controlled repair comparison

Source: [manuscript controlled ablation table](source_tables/ablation_compact.tex),
label `tab:newrepairquality`.

| Feedback / repair policy | Correct and matched (%) |
| --- | ---: |
| ReLoop | 44.8 |
| Opt-Verifier | 49.4 |
| ORSMT direct revision | 46.4 |
| ORSMT with plan review | 47.5 |
| ORSMT with row review | 49.6 |
| ORSMT certified voting | 50.9 |

The cohort is fixed to the 1,905 originally refuted model–code outputs. Every
feedback method uses the same Qwen2.5-14B reviser and one temperature-zero revision
with a 32,768-token context. The population differs from the full-pipeline table.
The figures show aggregate rates, without inferred error bars or significance
markers. The [manuscript's clustered comparison](source_tables/paired_comparison.tex) supports a positive difference
against ReLoop; its interval against Opt-Verifier includes zero.

## Host integration

Source: [manuscript host integration table](source_tables/plugin_compact.tex),
label `tab:plugincomparison`.

| System | Pipeline | Correct answer (%) | Artifact match (%) | Correct and matched (%) |
| --- | --- | ---: | ---: | ---: |
| OptiMUS-0.3 | Original | 21.3 | 36.7 | 20.8 |
| OptiMUS-0.3 | Original + ORSMT | 39.5 | 63.9 | 39.5 |
| OptiMind | Original | 62.1 | 74.3 | 55.3 |
| OptiMind | Original + ORSMT | 62.8 | 96.8 | 62.7 |

Both systems use the same 3,034 assignments. OptiMUS uses the exact model–code
relation. OptiMind uses an exact scalar extractor or reference-blind two-of-three
auditors and rechecks reconstructed artifacts. The homepage keeps the hosts in
separate panels and does not imply identical verification coverage.

## Visual examples

### NL4Opt-67: written proportion constraint

The [released trace](../../data/worked_trace/nl4opt67/summary.json) records a real
Qwen2.5-32B candidate. The written formulation uses `y - x >= 0.4`; the captured
program correctly uses `3y - 2x >= 0`. Both require `50x + 70y >= 1000`, integers
`x, y >= 0`, and `x <= 15`, and minimize `30x + 40y`.

The witness `(15, 10)` is feasible in the program and reviewed reference, but
infeasible in the written model. The recorded returned optimum `(2, 13)` has value
580 and satisfies both original artifacts. Source review identifies the written
model as the repair target. Its corrected row restores model agreement, with no
code change. This stored revision is manually reviewed, with zero LLM calls.

The figure shows feasible integer points only within `0 <= y <= 20`; the upper
plot window is not a model constraint. Shading denotes the continuous relaxation.
The formal checks concern the supplied formal reference; the trace does not claim
mechanical verification of unrestricted natural language.

### ComplexOR-3: integer versus binary domain

Source: [manuscript worked cases](source_tables/worked_cases.tex), label
`fig:workedcases`, right panel. This is a 1.5B output. With `max x + y`,
`x + y <= 2`, and nonnegative integer variables, the written model admits `(2, 0)`.
The program makes `x` binary and excludes this point. Both still admit the optimal
point `(0, 2)`, with value 2. Restoring the integer domain repairs model agreement.
The repeated constraint is shown once in the geometry; duplicate-row removal
preserves this feasible region. Source-level assignment requirements are separate.

## Recreate the illustrations

Plotting dependencies are optional and do not change the core package installation.

```bash
python -m pip install matplotlib numpy
python scripts/render_readme_results.py
```

This generates five PNG figures, editable SVG versions and `values.json`, without
running model generation, repair baselines or host experiments. The public frozen
ledgers contain the pre-repair core audit. See the [release guide](../reproduction.md)
for supported execution and the release inventory.
