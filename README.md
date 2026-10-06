# Failure-Type-Aware Generative Augmentation: Evaluation Code, Predictions, and Results

This repository holds the code, the complete archived predictions, and every result table and figure for the paper

> **Codifying Failure Knowledge for Industrial Decision Support: An Evaluation of Failure-Type-Aware Generative Augmentation.**
> *Interdisciplinary Journal of Information, Knowledge, and Management* (2026).
> Alexis Lazanas.
>
> Version 2.0.0 of this repository. The code of the earlier single-partition analysis of the same design
> (version 1) remains available under the release tag `v1-earlier-analysis`.

The study asks whether partitioning generative data augmentation by **recorded failure mechanism** (one WGAN-GP
generator per mechanism) improves failure detection on the AI4I 2020 predictive-maintenance benchmark, overall
and for each mechanism. It compares eight strategies under a leakage-safe protocol with identical classifier tuning,
across 30 repeated stratified data splits.

| Strategy | What it does | Mean test PR-AUC (SD), 30 splits |
|---|---|---|
| Baseline | Random Forest with balanced class weights | **0.818** (0.032) |
| Borderline-SMOTE | interpolation from boundary failures | 0.806 (0.027) |
| Multi WGAN-GP | **failure-type-aware**: one generator per mechanism | 0.803 (0.034) |
| SMOTE | interpolation between neighbouring failures | 0.797 (0.032) |
| ADASYN | adaptive interpolation | 0.791 (0.030) |
| SMOTENC | SMOTE with the product type treated as nominal | 0.786 (0.034) |
| Single WGAN-GP | one generator on all failures | 0.770 (0.048) |
| RUS | random undersampling | 0.695 (0.052) |

In short: the tuned cost-sensitive baseline was not outperformed; the failure-type-aware strategy did not differ
significantly from it; augmentation offered no benefit over adjusting the baseline's decision threshold at matched
operating points; and no strategy detected the rarest mechanism (tool wear) well. See the paper for the full results.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Repository layout](#2-repository-layout)
3. [The pipeline](#3-the-pipeline)
4. [Data](#4-data)
5. [Installation](#5-installation)
6. [Reproducing the paper](#6-reproducing-the-paper)
7. [File formats](#7-file-formats)
8. [For developers](#8-for-developers)
9. [Determinism and known caveats](#9-determinism-and-known-caveats)
10. [Citation and licenses](#10-citation-and-licenses)

---

## 1. Quick start

**Verify every number in the paper (one minute, CPU only, no dataset download needed):**

```bash
git clone https://github.com/alexlas73/PdM-with-WGAN-GP-sampling.git && cd PdM-with-WGAN-GP-sampling
pip install -r requirements-analysis.txt
python -m ftaa.analysis --run results/predictions --out analysis_check
python -m pytest -q          # includes checks of the paper's Tables 3-7 against the archived predictions
```

**Rerun the whole experiment (about 7 hours on a T4 GPU):** open
[`notebooks/run_in_colab.ipynb`](notebooks/run_in_colab.ipynb) in Google Colab and follow it, or see
[section 6](#6-reproducing-the-paper).

## 2. Repository layout

```
ftaa/                      the pipeline (a small Python package; run modules with python -m ftaa.<module>)
  config.py                every fixed setting of the study (grids, generator settings, strategy names)
  data.py                  loading, feature preparation, label audit, stratification, repeated splits
  samplers.py              the two WGAN-GP samplers (single generator; one generator per mechanism)
  experiment.py            Stage 2: 30 splits x 8 strategies -> archived predictions
  hpo.py                   Stage 1: Optuna search that fixed the generator settings
  analysis.py              Stage 3: all tables, statistics and figures from the predictions
  tsne.py                  Figures 9-10: t-SNE illustration of one split
  check_data.py            verifies the downloaded dataset
  compat.py                shims for tabgan 2.2.2 with current NumPy / scikit-learn
  versions.py              records library versions
results/
  predictions/             THE ARCHIVE: predictions.zip (240 prediction files) + settings.csv, config.json, label_audit.csv, versions.json
  tables/                  every table of the Results, as CSV and one Excel workbook, plus summary.json
  figures/                 Figures 5-10 (PNG 300 dpi; PDF where available)
notebooks/run_in_colab.ipynb   step-by-step Colab notebook (levels A, B, C)
tests/                     pytest suite (data protocol, samplers, paper numbers, end-to-end smoke test)
docs/
  REPRODUCE.md             detailed reproduction instructions
  PAPER_MAP.md             paper section / table / figure -> code and file
  DESIGN_NOTES.md          implementation details, deviations, and caveats
data/README.md             where to download the dataset and how to check it
```

## 3. The pipeline

```
 Stage 1 (run once)            Stage 2 (the experiment)                     Stage 3 (the analysis)
 ftaa.hpo                      ftaa.experiment                              ftaa.analysis
 original split, seed 42       30 stratified 70/15/15 splits (seed = split) results/predictions
 100-trial Optuna/TPE search   x 8 strategies                               -> Tables 3-7, Figures 5-8,
 per generative strategy       RF grid + SMOTE k tuned on validation         statistics (Friedman/Nemenyi,
          |                    -> val + test probabilities per record        corrected paired CIs,
          v                    -> results/predictions/                       Wilcoxon-Holm, d_z)
 settings fixed in config.py ------------^                                   ftaa.tsne -> Figures 9-10
```

**Leakage-safe protocol.** Within every split, preprocessing is fitted on the training partition only; resampling
and generation touch the training partition only; classifier settings, neighbourhood sizes and decision thresholds
are chosen on the validation partition; the test partition is used once, for the final evaluation. The failure
mechanism label is never a classifier input: it is used only to stratify the splits, to partition the failures for
the failure-type-aware generators, and to compute per-mechanism recall.

**Strategies** (all except the baseline balance the training partition to 1:1):

| Strategy | Implementation | Searched on every split |
|---|---|---|
| Baseline | `RandomForestClassifier(class_weight='balanced')` | RF grid |
| RUS | `imblearn RandomUnderSampler` | RF grid |
| SMOTE / SMOTENC / Borderline-SMOTE | `imblearn`, `k_neighbors` | k in {3, 5, 7} x RF grid |
| ADASYN | `imblearn`, `n_neighbors` | k in {3, 5, 7} x RF grid |
| Single WGAN-GP | `ftaa.samplers.OneTabGANSampler` (tabgan 2.2.2, CTGAN-based) | RF grid; generator fixed |
| Multi WGAN-GP | `ftaa.samplers.SpecializedTabGANSampler` | RF grid; generators fixed |

RF grid: 200 trees, `max_features='sqrt'`, `max_depth` in {10, 20} x `min_samples_leaf` in {1, 3}, selected on
validation PR-AUC. Generator settings (from Stage 1): Multi patience 600, batch 10, filter learning rate 0.0124;
Single patience 390, batch 10, filter learning rate 0.0101; up to 8,000 epochs with early stopping; LightGBM
post-generation filter (300 trees, depth 2).

## 4. Data

The study uses the **Kaggle single-label version** of AI4I 2020 (Bansal, 2021), derived from the UCI release
(Matzka, 2020). The dataset is **not redistributed here**; download `predictive_maintenance.csv` as described in
[`data/README.md`](data/README.md) and check it:

```bash
python -m ftaa.check_data data/predictive_maintenance.csv
```

The check confirms 10,000 rows and the label audit reported in the paper: 339 failures (112 heat dissipation,
95 power, 78 overstrain, 45 tool wear, 9 without a recorded mechanism) and 18 random failures marked as normal
operation. Stage 3 does not need the dataset: the archived predictions contain everything it uses.

## 5. Installation

| Purpose | Environment | Command |
|---|---|---|
| Verify the results (Stage 3, tests) | any OS, Python >= 3.10, CPU | `pip install -r requirements-analysis.txt` |
| Rerun the experiment (Stages 1-2, t-SNE) | Linux with an NVIDIA GPU, Python 3.13 | `pip install -r requirements.txt` |
| Google Colab | T4 GPU runtime | first cell of `notebooks/run_in_colab.ipynb` |

`requirements.txt` pins the exact versions that produced the archive (Python 3.13.15, scikit-learn 1.6.1,
imbalanced-learn 0.14.2, tabgan 2.2.2, PyTorch 2.11.0, Optuna 4.4.0, LightGBM 4.6.0, pandas 2.2.3, NumPy 2.1.3;
see `results/predictions/versions.json`). Optionally `pip install -e .` makes `ftaa` importable from anywhere.

## 6. Reproducing the paper

Full instructions, expected outputs, and timings are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md). In brief:

```bash
# Level A - tables, statistics and figures from the archive (1 min, CPU)
python -m ftaa.analysis --run results/predictions --out analysis_out

# Level B - the experiment (about 6.7 h on a T4; resumable), then Level A on the new archive
python -m ftaa.experiment --data data/predictive_maintenance.csv --out runs/rerun
python -m ftaa.analysis   --run runs/rerun --out runs/rerun/analysis

# Figures 9-10 (about 13 min on a T4)
python -m ftaa.tsne --data data/predictive_maintenance.csv --settings results/predictions/settings.csv --split 1 --out runs/figures

# Level C - the generator search (many hours per strategy)
python -m ftaa.hpo --data data/predictive_maintenance.csv --strategy multi  --out runs/hpo --storage sqlite:///runs/hpo/multi.db
python -m ftaa.hpo --data data/predictive_maintenance.csv --strategy single --out runs/hpo --storage sqlite:///runs/hpo/single.db
```

Which file produces which table or figure is listed in [`docs/PAPER_MAP.md`](docs/PAPER_MAP.md).

## 7. File formats

**`pred_sXX_<strategy>.csv`** (inside `results/predictions/predictions.zip`; a new run of `ftaa.experiment` writes
them as separate files, and `ftaa.analysis` reads either form) - one file per split (01-30) and strategy (spaces in
names become underscores), 3,000 rows each (1,500 validation + 1,500 test):

| column | meaning |
|---|---|
| `split` | split index 1-30 (also the random seed of that split) |
| `method` | strategy name as in `ftaa/config.py` |
| `set` | `val` or `test` |
| `row` | 0-based row index in `predictive_maintenance.csv` |
| `y` | binary target (1 = failure) |
| `failure_type` | the dataset's `Failure Type` label of that record (evaluation only) |
| `prob` | predicted failure probability, rounded to 6 decimals |

**`results/predictions/settings.csv`** - one row per (split, strategy): selected `k` (SMOTE family),
`max_depth`, `min_samples_leaf`, the validation PR-AUC of the selected setting, and `fit_seconds` (wall-clock time
of the whole strategy on that split, including tuning).

**`results/tables/`** - `performance.csv` (Table 3), `paired_differences.csv` (Table 4), `per_mechanism.csv`
(Table 5), `operating_points.csv` (Table 6), `compute.csv` and `selected_settings.csv` (Table 7),
`per_mechanism_diffs.csv`, the per-split metrics, `summary.json` (Friedman/Nemenyi statistics), and
`revision_results_tables.xlsx` (all tables, one sheet each).

## 8. For developers

**Code organisation.** Every fixed choice lives in `ftaa/config.py`; `ftaa/data.py` owns the data and the splits;
`ftaa/experiment.py` contains one function per step (`run_method` trains one strategy on one split) and writes one
file per (split, strategy), atomically, so runs are resumable and can be parallelised by split
(`--splits 1-10`, `--splits 11-20`, ... into the same `--out` folder; `settings.csv` is appended by each process).

**Adding a strategy.**
1. Add its name to `METHODS` in `ftaa/config.py`.
2. Add a branch in `ftaa.experiment.run_method` that resamples `(Z, ytr)` (or the raw `Xtr` for generators) and
   calls `tune_and_predict`, so that it receives the same RF search as every other strategy.
3. Run `python -m ftaa.experiment ... --methods "<new name>"` into a copy of the archive folder and then
   `python -m ftaa.analysis`; the analysis picks up every strategy in `METHODS`.

**Using another dataset.** Adapt `ftaa/data.load` (features), `NUMERICAL_FEATURES` / `CATEGORICAL_FEATURES` /
`FAILURE_COL` in `config.py`, and the expected counts used by `check_data`. The per-mechanism evaluation uses the
mechanism names in `MECH` in `ftaa/analysis.py`.

**Tests.** `python -m pytest -q` runs in a few minutes on a CPU without the real dataset (a synthetic file with the
same schema is generated) and without tabgan (the generator is replaced by a fake in `tests/test_samplers.py`).
`tests/test_analysis.py` checks the paper's reported numbers against the archive. Continuous integration runs the
suite on every push (`.github/workflows/tests.yml`).

## 9. Determinism and known caveats

* The splits, the non-generative strategies, the analysis, and the figures from the archive are deterministic.
* GPU training of the WGAN-GP generators is not bit-for-bit reproducible, so a rerun of Stage 2 reproduces the two
  generative strategies closely but not to the last digit. **The archived predictions are the reference** from
  which every reported number is computed.
* The generator settings were selected once, on the original partition of an earlier analysis, and then held fixed;
  that partition overlaps the test partitions of other splits, which gives the generative strategies a small
  optimistic bias (stated in the paper).
* Further implementation details (the scaler of the generative strategies is fitted on the augmented training
  partition; the unlabeled failures are used by the single generator only; Stage 1 specifics) are in
  [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md).

## 10. Citation and licenses

If you use this code or the archived results, please cite the paper and this repository (see `CITATION.cff`;
"Cite this repository" on GitHub).

* **Code** (`ftaa/`, `tests/`, `notebooks/`): MIT License, see [`LICENSE`](LICENSE).
* **Archived predictions, tables and figures** (`results/`): Creative Commons Attribution 4.0 International,
  see [`LICENSE-RESULTS`](LICENSE-RESULTS).
* **Dataset**: not included. AI4I 2020 is published by the UCI Machine Learning Repository under CC BY 4.0
  (Matzka, 2020); the single-label version is published on Kaggle (Bansal, 2021) under the terms shown there.

**References**
* Matzka, S. (2020). Explainable artificial intelligence for predictive maintenance applications. *Proceedings of the 2020 International Conference on Artificial Intelligence for Industries* (pp. 69-74). IEEE. https://doi.org/10.1109/AI4I49448.2020.00023
* AI4I (2020). AI4I 2020 predictive maintenance dataset. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
* Bansal, S. (2021). Machine predictive maintenance classification [Dataset]. Kaggle. https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
* Ashrapov, I. (2020). Tabular GANs for uneven distribution. arXiv:2010.00638. https://doi.org/10.48550/arXiv.2010.00638
* Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239-281. https://doi.org/10.1023/A:1024068626366
* Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*, 1-30.
