"""Fixed settings of the study. Changing anything here changes the experiment."""

# Strategies, in the order used throughout the code and the paper's tables.
METHODS = ['Baseline', 'RUS', 'SMOTE', 'SMOTENC', 'Borderline-SMOTE', 'ADASYN',
           'Single WGAN-GP', 'Multi WGAN-GP']
FOCAL = 'Multi WGAN-GP'          # the failure-type-aware strategy; paired differences are FOCAL minus other

# Repeated splits: 70 % train, 15 % validation, 15 % test; seed = split index (1..N_SPLITS).
N_SPLITS = 30
TEST_SIZE_FIRST = 0.30           # train vs (validation + test)
TEST_SIZE_SECOND = 0.50          # validation vs test, within the 30 %
MIN_STRATUM = 6                  # strata (Target|FailureType) smaller than this are merged into 'rare'

# Classifier: Random Forest, same search for every strategy on every split (selected on validation PR-AUC).
RF_N_ESTIMATORS = 200
RF_MAX_FEATURES = 'sqrt'
RF_GRID = [(d, l) for d in (10, 20) for l in (1, 3)]     # (max_depth, min_samples_leaf)

# SMOTE family: neighbourhood size searched jointly with the RF grid.
K_GRID = (3, 5, 7)

# Generator settings, selected once by the Optuna search in ftaa.hpo (original split, seed 42) and held fixed.
MULTI = {'patience': 600, 'batch_size': 10, 'adv_learning_rate': 0.012414180611485873}
SINGLE = {'patience': 390, 'batch_size': 10, 'adv_learning_rate': 0.010104754829563628}
GAN_EPOCHS = 8000                # upper bound; early stopping (patience) ends training earlier
TARGET_POS_RATIO = 0.5           # 1:1 failures to normal records after augmentation
FILTER_N_ESTIMATORS = 300        # LightGBM post-generation (adversarial) filter
FILTER_MAX_DEPTH = 2

# Data
DROP_COLUMNS = ['UDI', 'Product ID']
NUMERICAL_FEATURES = ['Air_temperature_C', 'Process_temperature_C', 'Rotationalspeedrpm', 'TorqueNm',
                      'Toolwearmin', 'Temperature_difference_C']
CATEGORICAL_FEATURES = ['Type']
TYPE_ORDER = ['L', 'M', 'H']
TARGET = 'Target'
FAILURE_COL = 'FailureType'
NOT_A_MECHANISM = {'No Failure', 'Random Failures'}   # FailureType values that do not get their own generator

# Expected label audit of the Kaggle single-label version (FailureType x Target counts).
EXPECTED_AUDIT = {
    ('Heat Dissipation Failure', 1): 112, ('Power Failure', 1): 95, ('Overstrain Failure', 1): 78,
    ('Tool Wear Failure', 1): 45, ('No Failure', 1): 9, ('No Failure', 0): 9643, ('Random Failures', 0): 18,
}
EXPECTED_ROWS = 10000


def gen_params(cfg, seed):
    """Generator parameters passed to tabgan.GANGenerator."""
    return {'epochs': GAN_EPOCHS, 'patience': cfg['patience'], 'batch_size': cfg['batch_size'], 'seed': seed}


def adv_params(cfg, seed):
    """Parameters of the LightGBM post-generation filter."""
    return {'n_estimators': FILTER_N_ESTIMATORS, 'max_depth': FILTER_MAX_DEPTH,
            'learning_rate': cfg['adv_learning_rate'], 'random_state': seed, 'verbose': -1}
