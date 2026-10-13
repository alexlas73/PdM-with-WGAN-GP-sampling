# Pre-specified generator settings (Table 8)

Licensed under CC BY 4.0 (see `../../LICENSE-RESULTS`).

Added in version 2.1.0 at the editor's request (second revision round). The generator settings in the main
archive (`../predictions/`) were selected once by the Optuna search on the original partition, which overlaps
the test partitions of other splits. To measure the effect of that overlap, both generative strategies were
rerun on the same 30 splits with settings fixed in advance and independent of AI4I 2020: the tabgan 2.2.2
library defaults (maximum 500 epochs, early-stopping patience 25, batch size 500 capped at the subset size,
filter learning rate 0.02; `TABGAN_DEFAULTS` in `ftaa/config.py`). Everything else was unchanged.

* `predictions.zip` - `pred_sXX_Single_WGAN-GP.csv` and `pred_sXX_Multi_WGAN-GP.csv` for 30 splits (60 runs).
  The other six strategies are not rerun: they do not depend on generator settings.
* `settings.csv`, `config.json`, `label_audit.csv` - as in the main archive, for these 60 runs.
* `tables/` - output of the command below (all eight strategies, with the generative strategies replaced),
  and `table8_generator_tuning.csv` (Table 8 of the paper: tuned vs pre-specified settings).

Reproduce the tables (1 min, CPU):

```bash
python -m ftaa.analysis --run results/predictions --replace results/round2_defaults --out analysis_round2
```

Rerun these 60 runs (about 10 minutes on a T4 GPU):

```bash
python -m ftaa.experiment --data data/predictive_maintenance.csv --out runs/round2 \
    --methods "Single WGAN-GP" "Multi WGAN-GP" --generator-settings defaults
```

Result: mean PR-AUC 0.768 (failure-type-aware) and 0.714 (single generator), against 0.803 and 0.770 with the
tuned settings and 0.818 for the cost-sensitive baseline. The tuning therefore favored the generative strategies.
