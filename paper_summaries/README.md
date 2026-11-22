# Paper summary scripts

Two standalone scripts that recompute supplementary quantities from the
manuscript's own tables (copies of the table sources are in `figures/`).
They read only these tables; no experiment records, GPU or network are needed.

| Script | Reads | Reproduces |
|---|---|---|
| `analysis/seed_summary_bounds.py` | `figures/tab_component_ablation.tex` (main-text Table 4) | the finite-sample bounds of Supplementary Table 12 (Appendix F) |
| `analysis/contact_count_ranges.py` | `figures/tab_per_planner.tex` (Supplementary Table 10) | the feasible overlap / discordance ranges quoted in Appendix D |

Run from this directory:

    python3 analysis/seed_summary_bounds.py
    python3 analysis/contact_count_ranges.py

Both print JSON. Table 12 rounds each bound downward at its displayed precision.
