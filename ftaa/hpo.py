"""Stage 1: Optuna search for the generator settings (run once per generative strategy).

This reproduces the search that produced the fixed settings in ftaa/config.py (MULTI and SINGLE). It was run
once, on the original partition of the earlier analysis (stratified on Target only, seed 42), and the selected
settings were then held fixed for all 30 repeated splits of Stage 2. Because that partition overlaps the test
partitions of other splits, this gives the generative strategies a small optimistic bias (stated in the paper).

Search space (both strategies)
  patience            10..1000, step 10   early-stopping patience of generator training (max 8,000 epochs)
  batch_size          10..60, step 10     must be a multiple of 10 (packed discriminator, pac = 10)
  adv_learning_rate   0.01..0.04, log     learning rate of the LightGBM post-generation filter
Objective: validation PR-AUC of a Random Forest trained on the augmented training partition. The Random Forest
settings inside the objective are chosen once, before the search, by 3-fold CV on the unaugmented training data.
Sampler: TPE (multivariate, group, n_ei_candidates=200, 20 random start-up trials, seed 42); 100 trials.

Faithfulness notes (kept as in the original search, deliberately not "fixed"):
  * The Random Forest inside the objective uses class_weight='balanced', also on augmented data.
  * The Multi objective allocates synthetic records by int(total * share), with share = mechanism count /
    all training failures (including those without a mechanism), so it generates slightly fewer records than
    a 1:1 ratio; Stage 2 uses exact largest-remainder allocation instead.
  * A MedianPruner(n_warmup_steps=10) is configured, but each trial reports a single step, so no trial is
    ever pruned in practice.

Usage
-----
python -m ftaa.hpo --data data/predictive_maintenance.csv --strategy multi --out runs/hpo
python -m ftaa.hpo --data ... --strategy single --out runs/hpo --trials 100 --storage sqlite:///runs/hpo/single.db
Expect many hours on a T4 GPU per strategy (each trial trains the generator(s) once).
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from . import compat
from . import config as C
from . import data as D

compat.apply()
warnings.filterwarnings('ignore')
SEED = 42


def original_split(X, y):
    """The fixed partition of the earlier analysis: 70/15/15, stratified on Target only, seed 42."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)
    return X_train, X_val, y_train, y_val


def deterministic_torch():
    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        torch.set_num_threads(2)
    except Exception:
        pass


def batch_for(bs, n):
    """Batch size capped at the subset size and rounded down to a multiple of 10 (minimum 10)."""
    bs = min(bs, n)
    if bs % 10 != 0:
        bs = (bs // 10) * 10
    return bs or 10


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--strategy', required=True, choices=['multi', 'single'])
    p.add_argument('--out', required=True)
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--storage', default=None, help='optional Optuna storage URL, e.g. sqlite:///runs/hpo/multi.db, '
                                                    'so that an interrupted search can be resumed')
    p.add_argument('--n-jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = p.parse_args(argv)

    import optuna
    from tabgan.sampler import GANGenerator
    import logging
    logging.getLogger('lightgbm').setLevel(logging.ERROR)
    logging.getLogger('optuna').setLevel(logging.WARNING)
    deterministic_torch()
    os.makedirs(a.out, exist_ok=True)

    df, X, y = D.load(a.data)
    X_train, X_val, y_train, y_val = original_split(X, y)
    X_train_model = X_train.drop(columns=[C.FAILURE_COL])
    X_val_model = X_val.drop(columns=[C.FAILURE_COL])

    # Random Forest used inside every trial: tuned once on the unaugmented training data (3-fold CV, AP).
    rf_balanced = RandomForestClassifier(random_state=SEED, class_weight='balanced', n_jobs=a.n_jobs)
    grid = {'model__n_estimators': [C.RF_N_ESTIMATORS], 'model__max_depth': [10, 20], 'model__min_samples_leaf': [1, 3]}
    gs = GridSearchCV(Pipeline([('preprocessor', D.make_prep()), ('model', rf_balanced)]), grid,
                      cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
                      scoring={'ap': 'average_precision', 'roc': 'roc_auc'}, refit='ap', n_jobs=a.n_jobs)
    gs.fit(X_train_model, y_train)
    tuned_rf = gs.best_estimator_.named_steps['model']
    print('RF used inside the objective:', {k: tuned_rf.get_params()[k] for k in ('max_depth', 'min_samples_leaf')})

    prep = D.make_prep().fit(X_train_model)
    Xtr_p, Xva_p = prep.transform(X_train_model), prep.transform(X_val_model)
    n0, n1 = int((y_train == 0).sum()), int((y_train == 1).sum())
    needed = n0 - n1

    def generate(gen_p, adv_p):
        if a.strategy == 'single':
            g = X_train[y_train == 1].drop(columns=[C.FAILURE_COL], errors='ignore')
            gp_ = dict(gen_p, batch_size=batch_for(gen_p['batch_size'], len(g)))
            s = GANGenerator(gen_x_times=max(needed / float(len(g)), 1e-6), cat_cols=C.CATEGORICAL_FEATURES,
                             is_post_process=True, adversarial_model_params=adv_p, gen_params=gp_)
            out, _ = s.generate_data_pipe(train_df=g, target=pd.Series([1] * len(g), name='Target', index=g.index).to_frame(),
                                          test_df=None, only_generated_data=True)
            return [out.reindex(columns=X_train_model.columns, fill_value=np.nan)]
        parts = []
        for ft in [f for f in df[C.FAILURE_COL].unique() if f not in C.NOT_A_MECHANISM]:
            sub = X_train[X_train[C.FAILURE_COL] == ft]
            if len(sub) < 2:
                continue
            k = int(needed * len(sub) / n1)
            if k <= 0:
                continue
            gp_ = dict(gen_p, batch_size=batch_for(gen_p['batch_size'], len(sub)))
            s = GANGenerator(gen_x_times=k / len(sub), cat_cols=C.CATEGORICAL_FEATURES, is_post_process=True,
                             adversarial_model_params=adv_p, gen_params=gp_)
            g = sub.drop(columns=[C.FAILURE_COL])
            out, _ = s.generate_data_pipe(train_df=g, target=pd.Series([1] * len(g), name='Target', index=g.index).to_frame(),
                                          test_df=None, only_generated_data=True)
            parts.append(out)
        return parts

    def objective(trial):
        patience = trial.suggest_int('patience', 10, 1000, step=10)
        batch_size = trial.suggest_int('batch_size', 10, 60, step=10)
        lr = trial.suggest_float('adv_learning_rate', 0.01, 0.04, log=True)
        gen_p = {'epochs': C.GAN_EPOCHS, 'patience': patience, 'batch_size': batch_size, 'seed': SEED}
        adv_p = {'n_estimators': C.FILTER_N_ESTIMATORS, 'max_depth': C.FILTER_MAX_DEPTH, 'learning_rate': lr,
                 'random_state': SEED, 'verbose': -1, 'n_jobs': a.n_jobs}
        try:
            parts = generate(gen_p, adv_p)
            if not parts:
                return -1.0
            synth = pd.concat(parts, ignore_index=True)
            Xa = np.vstack([Xtr_p, prep.transform(synth)])
            ya = pd.concat([y_train, pd.Series([1] * len(synth), name='Target')], ignore_index=True)
            m = clone(tuned_rf).fit(Xa, ya)
            score = average_precision_score(y_val, m.predict_proba(Xva_p)[:, 1])
            print(f'trial {trial.number}: {trial.params} -> validation PR-AUC {score:.4f}')
            trial.report(score, step=1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            return score
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f'trial {trial.number} failed: {e!r}')
            return -1.0

    sampler = optuna.samplers.TPESampler(n_ei_candidates=200, multivariate=True, group=True, seed=SEED,
                                         n_startup_trials=20)
    study = optuna.create_study(study_name=f'{a.strategy}_wgan_gp', storage=a.storage, load_if_exists=True,
                                sampler=sampler, direction='maximize',
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    remaining = a.trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=1)
    study.trials_dataframe().to_csv(f'{a.out}/{a.strategy}_trials.csv', index=False)
    best = {'strategy': a.strategy, 'best_value': study.best_value, **study.best_params}
    json.dump(best, open(f'{a.out}/{a.strategy}_best.json', 'w'), indent=1)
    print('best:', best)
    print('Settings used in the paper:', C.MULTI if a.strategy == 'multi' else C.SINGLE)


if __name__ == '__main__':
    main()
