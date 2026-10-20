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

The cohort is fixed to the 1,905 originally refuted model–code outputs. The
policies use the same Qwen2.5-14B reviser, but they have different candidate and
review call costs. Certified voting selects among five candidates. Each single
candidate revision call uses temperature zero with a 32,768-token context. Program,
model, requirement and judgment allowances are 3,072, 4,096, 8,192 and 800 tokens.

The separate one-round source-revision/reconstruction comparison measures restored
model–code matches, not the 50.9% correct-and-matched voting result. Its 84.4%
restoration rate must not be described as the outcome of five-candidate voting or
as evidence that all policies have equal call budgets.

The figures show aggregate rates without inferred error bars or significance
markers. The [clustered comparison](source_tables/paired_comparison.tex) supports a
positive difference against ReLoop; the interval against Opt-Verifier includes zero.

## Reconstruction and source revision

[Original Overleaf PDF](../../assets/overleaf/repair-progression.pdf), referenced
in `appendix_reorganized.tex` as `fig:repairoutcomes`.

| Stage | Model–code match (%) | Correct and matched (%) |
| --- | ---: | ---: |
| Original | 43.2 | 25.2 |
| After reconstruction | 86.4 | 32.4 |
| After source revision | 86.4 | 36.7 |

These values are measured **before protected selection**, over the full assigned
population. They must not be substituted for the protected full-pipeline results
of 88.2% and 40.2%. The PNG is rendered directly from the original PDF; its bar
heights, count labels and layout are preserved.

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

The figure reads all constraints, bounds and domains from the stored original,
captured and revised model objects, evaluates them with exact rational arithmetic,
and shows feasible integer points within `0 <= y <= 20`. This upper plot window is
not a model constraint. No continuous feasible region is used as a visual proxy
for integer decisions.
The formal checks concern the supplied formal reference; the trace does not claim
mechanical verification of unrestricted natural language.

### ComplexOR-3: integer versus binary domain

Source: [full manuscript case](source_tables/appendix_cases_detailed.tex),
`app:complexexample` and `fig:domaincase`. The figure uses the original Overleaf
PDF `generated_consistency/case_domain.pdf`. This is a 1.5B output. With `max x + y`,
`x + y <= 2`, and nonnegative integer variables, the written model admits `(2, 0)`.
The program makes `x` binary and excludes this point. Both still admit the optimal
point `(0, 2)`, with value 2. Restoring the integer domain repairs model agreement.
The repeated constraint is shown once in the geometry; duplicate-row removal
preserves this feasible region. Source-level assignment requirements are separate: the original assignment
problem has reference answer 3, while the domain-repaired generated model still
has optimum 2. The full source repair requires indexed assignment variables and
constraints, as stated in the manuscript.

## Recreate the illustrations

Plotting dependencies are optional and do not change the core package installation.

```bash
python -m pip install matplotlib numpy
python scripts/render_readme_results.py
```

This generates three comparison plots and the released shipping case, each as
PNG and SVG, plus `values.json`. It does not regenerate the original Overleaf
artwork. To compile the referenced method source and export the original PDFs:

```bash
python scripts/export_overleaf_figures.py
```

That command requires pdfLaTeX and Poppler. The raw source, original PDFs,
Overleaf revision and source hashes are recorded in
[the export record](../../assets/overleaf/README.md). These rendering commands do
not run model generation, repair baselines or host experiments.
