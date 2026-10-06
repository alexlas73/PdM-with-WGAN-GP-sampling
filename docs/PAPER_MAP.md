# Paper -> code and files

| Paper element | Produced by | File(s) |
|---|---|---|
| Label Audit (Evaluation Methods) | `ftaa.data.label_audit`, `ftaa.check_data` | `results/predictions/label_audit.csv` |
| Repeated splits, leakage-safe protocol | `ftaa.data.strata`, `ftaa.data.split_indices`, `ftaa.experiment.main` | - |
| Data preparation (Celsius, temperature difference, ordinal Type, scaling) | `ftaa.data.load`, `ftaa.data.make_prep` | - |
| Table 2 (strategies and settings searched) | `ftaa/config.py`, `ftaa.experiment.run_method` | - |
| Generative Architecture | `ftaa/samplers.py` | - |
| Generator Hyperparameter Selection, Figure 1 | `ftaa.hpo` | - |
| Classifier and Resampler Tuning | `ftaa.experiment.tune_and_predict` | `results/predictions/settings.csv` |
| Computing Environment | `ftaa.versions` | `results/predictions/versions.json` |
| Table 3 | `ftaa.analysis` | `results/tables/performance.csv` |
| Figure 5 | `ftaa.analysis` | `results/figures/Fig_PR_AUC_by_split.{png,pdf}` |
| Friedman test, Figure 6 | `ftaa.analysis` | `results/tables/summary.json` (`friedman` -> `PR-AUC`); `results/figures/Fig_CD_PR_AUC.{png,pdf}` |
| Table 4, Figure 7 | `ftaa.analysis` | `results/tables/paired_differences.csv` (`Measure == 'PR-AUC'`); `results/figures/Fig_paired_diff_PR_AUC.{png,pdf}` |
| Table 5, Figure 8 | `ftaa.analysis` | `results/tables/per_mechanism.csv` (`Operating point == 'rec90'`); `results/figures/Fig_per_mechanism_recall_rec90.{png,pdf}` |
| Per-mechanism results at the F1-optimal and 0.5 thresholds; per-mechanism significance | `ftaa.analysis` | `per_mechanism.csv` (`F1opt`, `t0.5`); `per_mechanism_diffs.csv` |
| Table 6 (matched operating points, workload per 1,000) | `ftaa.analysis` | `results/tables/operating_points.csv` (`rec90`, `fpr1`; also `rec80`, `fpr2`, `F1opt`, `t0.5`) |
| Table 7 (computational cost, selected settings) | `ftaa.analysis` | `results/tables/compute.csv`, `results/tables/selected_settings.csv` |
| Figures 9 and 10 (t-SNE, split 1) | `ftaa.tsne` | `results/figures/Fig9_tsne_resampling_split1.png`, `Fig10_tsne_generative_split1.png` |

Operating-point codes: `t0.5` fixed threshold 0.5; `F1opt` F1-optimal threshold on validation; `rec80` / `rec90`
validation threshold for overall recall 0.80 / 0.90; `fpr1` / `fpr2` validation threshold for a false-positive rate
of 1 % / 2 %. All validation-chosen thresholds are applied unchanged to the test partition.
