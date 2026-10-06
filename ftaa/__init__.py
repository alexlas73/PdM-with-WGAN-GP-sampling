"""ftaa: evaluation pipeline for failure-type-aware generative augmentation on AI4I 2020.

Modules
-------
config      fixed settings of the study (grids, generator settings, strategy names)
compat      compatibility shims needed to run tabgan 2.2.2 with current NumPy / scikit-learn
data        data loading, feature preparation, label audit, stratification, repeated splits
samplers    the two WGAN-GP samplers (single generator, one generator per failure mechanism)
experiment  stage 2: 30 repeated splits x 8 strategies -> archived predictions
hpo         stage 1: Optuna search for the generator settings (run once, original split)
analysis    stage 3: every table, statistic and figure of the paper from the archived predictions
tsne        illustrative t-SNE figures (Figures 9 and 10) for one split
check_data  verifies that the downloaded dataset is the one used in the study
versions    records the library versions of the current environment
"""
__version__ = "2.0.0"
