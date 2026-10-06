"""Stage 2: the repeated-split experiment (30 splits x 8 strategies).

For every split and strategy it writes one file pred_sXX_<strategy>.csv with the predicted failure probability of
every validation and test record (columns: split, method, set, row, y, failure_type, prob), and appends the
settings selected on validation and the wall-clock time to settings.csv. The run is resumable: existing
prediction files are skipped, so an interrupted run can simply be started again.

Usage
-----
python -m ftaa.experiment --data data/predictive_maintenance.csv --out runs/my_run
python -m ftaa.experiment --data ... --out ... --splits 1-3 --methods Baseline SMOTE      # quick partial run

Expected time on a Colab T4 GPU: about 6.7 hours for the full run (Multi WGAN-GP ~9 min and Single WGAN-GP
~4 min per split; all other strategies take seconds).
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from . import compat
from . import config as C
from . import data as D

compat.apply()
warnings.filterwarnings('ignore')


def rf(depth, leaf, weighted, seed, n_jobs):
    return RandomForestClassifier(n_estimators=C.RF_N_ESTIMATORS, max_depth=depth, min_samples_leaf=leaf,
                                  max_features=C.RF_MAX_FEATURES, class_weight='balanced' if weighted else None,
                                  random_state=seed, n_jobs=n_jobs)


def tune_and_predict(Zr, yr, Zv, yv, weighted, seed, n_jobs):
    """Fit the RF grid on (Zr, yr); keep the setting with the highest validation PR-AUC (first one on ties)."""
    best = None
    for d, l in C.RF_GRID:
        m = rf(d, l, weighted, seed, n_jobs).fit(Zr, yr)
        pv = m.predict_proba(Zv)[:, 1]
        a = average_precision_score(yv, pv)
        if best is None or a > best[0]:
            best = (a, m, pv, d, l)
    return best


def run_method(method, s, Xtr, ytr, Xv, yv, Xte, n_jobs):
    """Train one strategy on one split. Returns (validation probs, test probs, info dict)."""
    from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler
    t0 = time.perf_counter()
    info = {}
    if method in ('Single WGAN-GP', 'Multi WGAN-GP'):
        from .samplers import OneTabGANSampler, SpecializedTabGANSampler
        cfg = C.SINGLE if method.startswith('Single') else C.MULTI
        cls = OneTabGANSampler if method.startswith('Single') else SpecializedTabGANSampler
        smp = cls(gen_params=C.gen_params(cfg, s), adv_params=C.adv_params(cfg, s),
                  categorical_features=C.CATEGORICAL_FEATURES, target_pos_ratio=C.TARGET_POS_RATIO, random_state=s)
        Xr, yr = smp.fit_resample(Xtr.reset_index(drop=True), ytr.reset_index(drop=True))
        Xr = pd.DataFrame(Xr, columns=Xtr.columns) if not isinstance(Xr, pd.DataFrame) else Xr
        for c in C.NUMERICAL_FEATURES:
            Xr[c] = pd.to_numeric(Xr[c], errors='coerce')
        keep = Xr[C.NUMERICAL_FEATURES + C.CATEGORICAL_FEATURES].notna().all(axis=1).values
        Xr, yr = Xr[keep], np.asarray(yr)[keep]
        P = D.make_prep().fit(Xr)                       # fitted on the augmented training partition
        Zr = P.transform(Xr)
        best = tune_and_predict(Zr, yr, P.transform(Xv), yv, False, s, n_jobs)
    else:
        P = D.make_prep().fit(Xtr)
        Z = P.transform(Xtr)
        Zv = P.transform(Xv)
        if method == 'Baseline':
            best = tune_and_predict(Z, ytr.values, Zv, yv, True, s, n_jobs)
        elif method == 'RUS':
            Zr, yr = RandomUnderSampler(random_state=s).fit_resample(Z, ytr.values)
            best = tune_and_predict(Zr, yr, Zv, yv, False, s, n_jobs)
        else:
            best = None
            for k in C.K_GRID:
                if method == 'SMOTE':
                    sm = SMOTE(k_neighbors=k, random_state=s)
                elif method == 'SMOTENC':
                    sm = SMOTENC(categorical_features=D.CAT_IDX, k_neighbors=k, random_state=s)
                elif method == 'Borderline-SMOTE':
                    sm = BorderlineSMOTE(k_neighbors=k, random_state=s)
                elif method == 'ADASYN':
                    sm = ADASYN(n_neighbors=k, random_state=s)
                else:
                    raise ValueError(f'unknown method {method!r}')
                try:
                    Zr, yr = sm.fit_resample(Z, ytr.values)
                except Exception as e:
                    print(f'   {method} k={k} skipped: {e}')
                    continue
                b = tune_and_predict(Zr, yr, Zv, yv, False, s, n_jobs)
                if best is None or b[0] > best[0]:
                    best = b
                    info['k'] = k
    val_ap, model, pv, d, l = best
    pt = model.predict_proba(P.transform(Xte))[:, 1]
    info.update({'max_depth': d, 'min_samples_leaf': l, 'val_PR_AUC': val_ap,
                 'fit_seconds': round(time.perf_counter() - t0, 1)})
    return pv, pt, info


def seed_everything(s):
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except Exception:
        pass


def parse_splits(text):
    out = []
    for part in text.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument('--data', required=True, help='path to predictive_maintenance.csv (Kaggle single-label version)')
    ap_.add_argument('--out', required=True, help='output folder (created if missing; resumable)')
    ap_.add_argument('--splits', default=f'1-{C.N_SPLITS}', help='e.g. 1-30 (default), 1-3, or 1,5,9')
    ap_.add_argument('--methods', nargs='+', default=C.METHODS, choices=C.METHODS, metavar='METHOD',
                     help='subset of: ' + ', '.join(C.METHODS))
    ap_.add_argument('--n-jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap_.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    df, X_all, y_all = D.load(a.data)
    audit = D.label_audit(df)
    audit.to_csv(f'{a.out}/label_audit.csv')
    print('Label audit (FailureType x Target):')
    print(audit)
    strat = D.strata(df)
    splits = parse_splits(a.splits)
    json.dump({'N_SPLITS': C.N_SPLITS, 'RF_GRID': C.RF_GRID, 'K_GRID': C.K_GRID, 'MULTI': C.MULTI, 'SINGLE': C.SINGLE,
               'METHODS': C.METHODS, 'split': 'stratified 70/15/15 on Target|FailureType, random_state=split'},
              open(f'{a.out}/config.json', 'w'), indent=1)
    print(f'\nStarting splits {splits[0]}..{splits[-1]} x {len(a.methods)} strategies -> {a.out}')

    for s in splits:
        tr, va, te = D.split_indices(len(df), strat, s)
        Xtr, ytr = X_all.iloc[tr], y_all.iloc[tr]
        Xv, yv = X_all.iloc[va], y_all.iloc[va].values
        Xte, yte = X_all.iloc[te], y_all.iloc[te].values
        for method in [m for m in C.METHODS if m in a.methods]:
            out = f"{a.out}/pred_s{s:02d}_{method.replace(' ', '_')}.csv"
            if os.path.exists(out):
                continue
            seed_everything(s)
            try:
                pv, pt, info = run_method(method, s, Xtr, ytr, Xv, yv, Xte, a.n_jobs)
            except Exception as e:
                print(f'split {s} {method} FAILED: {e}')
                with open(f'{a.out}/errors.log', 'a') as f:
                    f.write(f'{s},{method},{e!r}\n')
                continue
            rows = pd.concat([
                pd.DataFrame({'split': s, 'method': method, 'set': 'val', 'row': va, 'y': yv,
                              'failure_type': df[C.FAILURE_COL].values[va], 'prob': pv}),
                pd.DataFrame({'split': s, 'method': method, 'set': 'test', 'row': te, 'y': yte,
                              'failure_type': df[C.FAILURE_COL].values[te], 'prob': pt})])
            rows['prob'] = rows['prob'].round(6)
            rows.to_csv(out + '.tmp', index=False)
            os.replace(out + '.tmp', out)                 # atomic: a file exists only when it is complete
            cols = ['split', 'method', 'k', 'max_depth', 'min_samples_leaf', 'val_PR_AUC', 'fit_seconds']
            rec = {'split': s, 'method': method, 'k': np.nan, **info}
            settings = f'{a.out}/settings.csv'
            pd.DataFrame([rec])[cols].to_csv(settings, mode='a', header=not os.path.exists(settings), index=False)
            print(f"split {s:02d} | {method:17s} | test PR-AUC {average_precision_score(yte, pt):.4f} | "
                  f"{info['fit_seconds']}s")
        print(f'===== split {s} complete =====')
    from .versions import write as write_versions
    write_versions(f'{a.out}/versions.json')
    print('\nDone. Next: python -m ftaa.analysis --run', a.out, '--out <analysis folder>')


if __name__ == '__main__':
    main()
