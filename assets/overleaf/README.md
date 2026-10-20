# Original Overleaf artwork and faithful exports

Source revision: `0e57b3e`, fetched and checked on 2026-10-20. The active manuscript
references `figures/orsmt_method_conference_v11.tex` in `method_repair.tex`.
Older `completion_flow_conference_v8.pdf` and `v9.pdf` files are not substituted
for that current source.

| Homepage asset | Original Overleaf file | Handling |
| --- | --- | --- |
| `method.pdf` / `method.png` | `figures/orsmt_method_conference_v11.tex` | Compile the unmodified TikZ source with its four original icons and the manuscript's Times/Helvetica configuration |
| `repair-progression.pdf` / `.png` | `generated_consistency/repair_effects_3034.pdf` | PDF copied byte for byte; PNG rendered directly from it |
| `domain-case.pdf` / `.png` | `generated_consistency/case_domain.pdf` | PDF copied byte for byte; PNG rendered directly from it |

`source/figures/` contains the exact referenced TikZ source and icons.
`source/method.tex` is the standalone export wrapper. `fix-cm` permits the source's
explicit mathematical font sizes without LaTeX size substitution. It changes
neither figure labels nor formulas. A three-point border protects the artwork
from clipping. Fonts in the PDF are embedded. PNGs are rendered at 350 dpi.

Run from the repository root:

```bash
python scripts/export_overleaf_figures.py
```

This requires pdfLaTeX and Poppler. The two original plot PDFs are the vector
masters; this command renders them without redrawing data or labels.
`provenance.json` records source hashes and the manuscript figure references.

## Interpretation checks

- The original progression figure is **before protected selection**. Its final
  rates are 86.4% model–code match and 36.7% correct and matched. The protected
  full-pipeline table separately reports 88.2% and 40.2%.
- The domain figure certifies agreement of the generated pair after domain repair.
  Its optimum remains 2. The original assignment task's reference answer is 3,
  so domain repair does not restore source fidelity.
- Five-candidate certified voting, review ablations and the one-round source
  revision experiment use different policies and call costs.
- The method figure's source checks apply to admitted source contracts. The
  adoption gate applies the checks required by the active policy. Absolute
  objective values are checked separately from semantic preference ordering.
