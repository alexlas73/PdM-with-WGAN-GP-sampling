# Results archive

Licensed under CC BY 4.0 (see `../LICENSE-RESULTS`).

* `predictions/` - the single archived set of predictions from which every reported result is computed:
  `predictions.zip` with `pred_sXX_<strategy>.csv` for 30 splits x 8 strategies (validation and test probabilities for every record),
  `settings.csv` (settings selected on validation and wall-clock time per split and strategy), `config.json`
  (the fixed design), `label_audit.csv`, and `versions.json` (environment of the run). All 240 runs completed
  without error.
* `tables/` - output of `python -m ftaa.analysis --run results/predictions`: one CSV per table, the per-split
  metrics, `summary.json`, and `revision_results_tables.xlsx` (all tables, one sheet each). See
  `../docs/PAPER_MAP.md` for which file is which table.
* `figures/` - Figures 5-8 (`Fig_*`, PNG 300 dpi and PDF, produced by `ftaa.analysis`) and Figures 9-10
  (`Fig9_*`, `Fig10_*`, produced by `ftaa.tsne` for split 1).
