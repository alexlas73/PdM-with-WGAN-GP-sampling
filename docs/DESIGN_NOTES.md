# Design notes, implementation details, and caveats

This file records details that matter for exact reproduction or for extending the code, including places where
the implementation is more specific than the paper's prose.

## Data and splits

* **Outcome.** The binary `Target` is the outcome throughout. The 18 records labeled `Random Failures` have
  `Target = 0` and are therefore normal operation; the 9 records with `Target = 1` and `Failure Type = No Failure`
  are failures without a recorded mechanism. Both are kept in every partition.
* **Features.** Six numerical features (air and process temperature in degrees Celsius, rotational speed, torque,
  tool wear, and the derived process-minus-air temperature difference) and the product type (L < M < H, ordinal).
  `FailureType` is carried in the feature frame because the samplers need it, but `make_prep()` drops it, so no
  classifier ever sees it.
* **Stratification.** Splits are stratified on `Target|FailureType`. Strata smaller than six records would be merged
  into one `rare` stratum; with this dataset the smallest stratum (`1|No Failure`) has nine records, so no merge
  occurs.
* **Split seeds.** Split *s* uses `random_state = s` for both `train_test_split` calls (train vs. rest, then
  validation vs. test within the rest, stratified on the same labels), and the same *s* seeds the resamplers,
  the generators, the Random Forest, NumPy and PyTorch for every strategy on that split.

## Strategies

* **Preprocessing of the generative strategies.** The generators work on raw feature values. The scaler and
  encoder are then fitted on the augmented training partition (original plus synthetic records); for all other
  strategies they are fitted on the original training partition before resampling. In both cases only training
  data are used.
* **Invalid synthetic rows.** Synthetic rows whose numerical values cannot be parsed or whose `Type` is missing are
  dropped before the classifier is trained (`keep` mask in `run_method`).
* **Exact balance.** Both samplers generate exactly the number of synthetic records needed for a 1:1 ratio. The
  post-generation filter may return more or fewer records than requested; surplus records are sampled down, and a
  shortfall is filled by resampling the generated records with replacement.
* **Allocation (Multi WGAN-GP).** The required number is divided among the four mechanisms in proportion to their
  training failures, with largest-remainder rounding. The unlabeled failures train no generator (they cannot be
  assigned to one) but stay in the training data as real records. The single generator, which ignores
  `FailureType`, is trained on all training failures including the unlabeled ones.
* **Batch size.** The generator batch size is capped at the size of its training subset and rounded down to a
  multiple of 10 (the packed discriminator uses pac = 10), with a minimum of 10.
* **Tie-breaking in the grid.** `tune_and_predict` keeps the first setting with the highest validation PR-AUC, in
  the order of `RF_GRID`; for the SMOTE family, the first `k` with the best result is kept.

## Stage 1 (generator search) as it was run

The search in `ftaa/hpo.py` reproduces the original search, including choices that differ from Stage 2:

* It used the original partition of the earlier analysis (70/15/15, stratified on `Target` only, seed 42).
* The Random Forest inside the objective was tuned once by 3-fold cross-validation on the unaugmented training
  data and uses `class_weight='balanced'`, also on augmented data.
* The Multi objective allocates `int(total * share)` records per mechanism, with shares computed over all training
  failures (including unlabeled ones), so it generates slightly fewer records than an exact 1:1 ratio.
* A `MedianPruner(n_warmup_steps=10)` is configured, but each trial reports a single step, so no trial is pruned in
  practice.
* PyTorch was put in deterministic mode (`cudnn.deterministic`, `use_deterministic_algorithms(warn_only=True)`).

The selected settings were then fixed for Stage 2 (`ftaa/config.py`). Because the original partition overlaps the
test partitions of the repeated splits, this gives the two generative strategies a small optimistic bias.

## Stage 3 (analysis)

* **Thresholds** are chosen on each split's validation predictions and applied to its test predictions:
  F1-optimal = argmax of F1 along the validation precision-recall curve; recall target = the highest threshold that
  still reaches the target recall; false-positive-rate target = the lowest threshold whose validation FPR does not
  exceed the target. Achieved test values therefore differ slightly from the targets.
* **PR-AUC** is `average_precision_score`. Workload per 1,000 observations is computed per split and averaged.
* **Corrected confidence intervals** (Nadeau & Bengio, 2003): the variance of a mean paired difference over J = 30
  splits uses the factor (1/J + n_test/n_train) with n_test = 1,500 and n_train = 7,000; t distribution with
  J - 1 degrees of freedom. A difference is called significant only when this interval excludes zero.
  Per-mechanism recall intervals use the same correction and are clipped to [0, 1].
* **Friedman / Nemenyi** (Demšar, 2006): ranks within each split (1 = best); critical difference
  q(0.05; k, inf)/sqrt(2) * sqrt(k(k+1)/(6N)), giving CD = 1.92 for k = 8, N = 30.
* **Supplementary tests**: paired Wilcoxon signed-rank (`zero_method='wilcox'`) with Holm correction across the seven
  focal comparisons of each measure, and Cohen's d_z; these do not correct for the dependence between splits.

## Figures 9 and 10 (t-SNE)

* One joint embedding (perplexity 30, PCA initialization, seed = split) of: a random sample of 1,500 normal-operation
  training records together with those retained by random undersampling, all original training failures, and 600
  synthetic records drawn from each strategy.
* Because the plot's sample and the undersampler draw from the same random permutation (both use
  `RandomState(split)`), the 237 normal records retained by undersampling fall inside the 1,500-record sample, so
  1,500 normal records are embedded in total.
* t-SNE preserves neighbourhoods, not distances or densities, and normal operation is subsampled: the figures show
  which records lie near one another, not how common they are.

## Reproducibility limits

* The archived predictions in `results/predictions/` are the reference for every reported number.
* GPU training of the generators is not bit-for-bit deterministic across runs and hardware; reruns reproduce the
  two generative strategies closely but not exactly. The other six strategies are deterministic given the pinned
  library versions.
* Two compatibility shims (`ftaa/compat.py`) are needed to run tabgan 2.2.2 with NumPy 2 and current
  scikit-learn; they restore removed aliases and map a renamed keyword and do not change any computation.
