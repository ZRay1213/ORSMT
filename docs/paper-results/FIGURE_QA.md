# Homepage figure reconciliation

Checked against Overleaf revision `0e57b3e` on 2026-10-20.

## Issues corrected

1. The former standalone method export used default fonts. The active manuscript
   uses Times and Helvetica. The current export compiles the referenced v11 TikZ
   source unchanged, preserves its mathematical notation and uses the manuscript
   font configuration. `fix-cm` permits its declared math sizes. The rightmost
   report label fits inside its box, with a margin verified in the PDF text bounds.
2. The original reconstruction ablation is now displayed directly from Overleaf's
   PDF. Its pre-selection 86.4% and 36.7% outcomes are separated from the protected
   pipeline's 88.2% and 40.2% outcomes.
3. The policy comparison no longer describes five-candidate voting and all review
   policies as a shared single-round call budget. Its labels, caption and protocol
   distinguish the five-candidate policy from the one-round reconstruction study.
4. The domain case now uses the original Overleaf PDF. Its description includes
   duplicate-row removal and the distinction between repaired generated optimum 2
   and source reference answer 3.
5. The shipping illustration now evaluates original, captured and revised model
   constraints with exact rational arithmetic. It shows shared and differing
   integer decisions directly, without shaded continuous relaxations. It reads
   the separating point and returned solution from the stored trace. Annotations
   lie in empty regions, with a shared legend outside the axes.
6. Stale method and custom domain images were removed so the README has one
   referenced version for each original figure.

## Checks performed

- Directly copied source files and original PDFs are byte-identical to Overleaf;
  source paths and SHA-256 hashes are in `assets/overleaf/provenance.json`.
- The current method figure, original domain PDF, original progression PDF,
  controlled policy comparison and revised shipping visualization were inspected.
- Method PDF fonts are embedded; the explicit font sizes compile without glyph or
  size-substitution warnings after `fix-cm` is loaded.
- Derived SVGs retain `<text>` elements. Main and integration comparison geometry
  and values are unchanged from the previously reviewed versions.
- All README and documentation relative links resolve.
- The shipping counterexample is admitted by the original program and revised
  model, excluded by the original formulation, and the recorded optimum is
  admitted by all three. The displayed revised/program feasible sets agree.

The original manuscript's mathematical definitions and numerical results are
preserved. The source-contract and active-policy prerequisites are explained
beside the method figure. No additional experimental runs or significance claims
were introduced.
