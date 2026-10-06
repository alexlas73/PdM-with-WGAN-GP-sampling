# Reproducing the paper

There are three levels. Each level is independent: Level A needs only this repository; Level B needs the dataset
and a GPU; Level C is needed only to re-derive the fixed generator settings.

| Level | Reproduces | Needs | Time |
|---|---|---|---|
| A | every number, table and figure of the Results (Tables 3-7, Figures 5-8, all statistics) from the archived predictions | Python, CPU | ~1 min |
| B | the archived predictions themselves (Stage 2), then Level A on them; Figures 9-10 | dataset, NVIDIA GPU | ~7 h + 13 min |
| C | the generator settings in `ftaa/config.py` (Stage 1) | dataset, NVIDIA GPU | many hours per strategy |

The commands below are run from the repository root. In Google Colab, use `notebooks/run_in_colab.ipynb`, which
contains the same steps as cells.

---

## Level A - verify the reported results

```bash
pip install -r requirements-analysis.txt
python -m ftaa.analysis --run results/predictions --out analysis_out
python -m pytest -q tests/test_analysis.py
```

Expected console output (abridged):

```
splits complete: 30/30; missing: []
        Strategy  PR-AUC mean  PR-AUC SD  Mean rank PR-AUC
        Baseline       0.8180     0.0322            1.9667
Borderline-SMOTE       0.8058     0.0270            3.3333
   Multi WGAN-GP       0.8027     0.0337            3.4000
...
        Multi WGAN-GP − Baseline    -0.0150 -0.0506   0.0206        False    10      20
...
Friedman PR-AUC chi2=115.36 p=7.1e-22 CD=1.917
```

`analysis_out/` then contains the same files as `results/tables/` and `results/figures/` (Figures 5-8). The CSV
files are identical to the archived ones; `tests/test_analysis.py` checks the values reported in the paper.

## Level B - rerun the experiment

### B1. Environment

A Linux machine (or Colab) with an NVIDIA GPU and Python 3.13:

```bash
pip install -r requirements.txt
python -m ftaa.versions          # prints the versions; compare with results/predictions/versions.json
```

In Colab, restart the session once after installing (Runtime -> Restart session).

### B2. Dataset

Download `predictive_maintenance.csv` (see `data/README.md`), place it in `data/`, and check it:

```bash
python -m ftaa.check_data data/predictive_maintenance.csv
```

It must end with `OK: matches the dataset used in the study`.

### B3. Quick test (about 1 minute)

```bash
python -m ftaa.experiment --data data/predictive_maintenance.csv --out runs/quick --splits 1 --methods Baseline RUS SMOTE
```

On split 1 the archived test PR-AUC values are Baseline 0.8295, RUS 0.6978, SMOTE 0.8157; the rerun must give the
same values, because these strategies are deterministic.

### B4. Full run

```bash
python -m ftaa.experiment --data data/predictive_maintenance.csv --out runs/rerun
```

* Progress is printed per (split, strategy). Expected time on a T4: about 6.7 hours in total (Multi WGAN-GP about
  9 min and Single WGAN-GP about 4 min per split; tabgan's progress bars count up to 8,000 epochs, but early
  stopping ends training much earlier, so their time estimates are too high).
* The run is resumable. If it stops (for example, a Colab disconnect), run the same command again: completed
  (split, strategy) files are skipped. In Colab, write `--out` to Google Drive so that it survives a disconnect.
* To split the work over several machines or sessions, give each a different range, e.g. `--splits 1-10`,
  `--splits 11-20`, `--splits 21-30`, and copy all output files into one folder before the analysis.
* Failures are written to `errors.log` in the output folder; the archived run had none.

### B5. Analysis of the rerun

```bash
python -m ftaa.analysis --run runs/rerun --out runs/rerun/analysis
```

Compare `runs/rerun/analysis/performance.csv` with `results/tables/performance.csv`. The six non-generative
strategies reproduce exactly. The two generative strategies reproduce closely but not exactly, because GPU training
of the generators is not bit-for-bit deterministic.

### B6. Figures 9 and 10

```bash
python -m ftaa.tsne --data data/predictive_maintenance.csv --settings results/predictions/settings.csv --split 1 --out runs/figures
```

About 13 minutes on a T4 (both generators are retrained for split 1). The interpolation panels reproduce exactly;
the generative panels can differ slightly for the reason above. The coordinates are saved as
`tsne_split1_coordinates.csv`.

## Level C - rerun the generator search

```bash
python -m ftaa.hpo --data data/predictive_maintenance.csv --strategy multi  --out runs/hpo --storage sqlite:///runs/hpo/multi.db
python -m ftaa.hpo --data data/predictive_maintenance.csv --strategy single --out runs/hpo --storage sqlite:///runs/hpo/single.db
```

Each search runs 100 trials; each trial trains the generator(s) once. With `--storage`, an interrupted search
resumes from the trials already completed. The output `<strategy>_best.json` holds the best settings, and
`<strategy>_trials.csv` all trials. The settings used in the paper are printed at the end for comparison. Because
generator training is not bit-for-bit deterministic, the search can select somewhat different settings; the paper
uses the settings fixed in `ftaa/config.py`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `AttributeError: module 'numpy' has no attribute 'Inf'` or `OneHotEncoder() got an unexpected keyword argument 'sparse'` | tabgan 2.2.2 with NumPy 2 / new scikit-learn. `ftaa` applies the fix (`ftaa/compat.py`) automatically; make sure you call the modules through `python -m ftaa...` and do not import tabgan before ftaa. |
| `check_data` reports different counts | You have the UCI multi-flag release or another derivative. Use the Kaggle single-label file (`data/README.md`). |
| Colab disconnects during Level B | Reconnect, rerun the install, Drive and data cells, then the same experiment cell; it resumes. |
| Very slow GAN training | No GPU in use. Check `python -m ftaa.versions` (`gpu` should name the device) or Colab's runtime type. |
| `ModuleNotFoundError: ftaa` | Run from the repository root, or `pip install -e .` once. |
