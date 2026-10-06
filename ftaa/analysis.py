"""Stage 3: every table, statistic and figure of the paper's Results, from the archived predictions alone.

python -m ftaa.analysis --run results/predictions --out analysis_out

Inputs (in --run): pred_sXX_<strategy>.csv (as files or inside predictions.zip), settings.csv, and optionally versions.json, errors.log.
Outputs (in --out): revision_results_tables.xlsx, one CSV per table, per_split_metrics.csv,
per_split_mechanism.csv, summary.json, and figures/ (PNG at 300 dpi and PDF).

Paper element            file / sheet                                   rows used
Table 3                  performance.csv                                all
Figure 5                 figures/Fig_PR_AUC_by_split                    -
Figure 6, Friedman test  figures/Fig_CD_PR_AUC; summary.json 'friedman' 'PR-AUC'
Table 4, Figure 7        paired_differences.csv; Fig_paired_diff_PR_AUC Measure == 'PR-AUC'
Table 5, Figure 8        per_mechanism.csv; Fig_per_mechanism_recall_rec90  Operating point == 'rec90'
Table 6                  operating_points.csv                           Operating point in {'rec90', 'fpr1'}
Table 7                  compute.csv + settings.csv (selected settings) all

Operating points: thresholds are chosen on each split's validation predictions (F1-optimal; recall 0.80 / 0.90;
false-positive rate 1 % / 2 %) and applied unchanged to its test predictions; 't0.5' is the fixed 0.5 threshold.
Confidence intervals for paired differences use the corrected resampled t-statistic (Nadeau & Bengio, 2003),
variance factor (1/J + n_test/n_train) with n_test = 1,500 and n_train = 7,000.
Runs on CPU in about a minute; no GPU, tabgan or imbalanced-learn needed.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import FOCAL, METHODS

MECH = {'Heat Dissipation Failure': 'Heat dissipation', 'Power Failure': 'Power',
        'Overstrain Failure': 'Overstrain', 'Tool Wear Failure': 'Tool wear', 'No Failure': 'No mechanism recorded'}
MECH_ORDER = ['Heat dissipation', 'Power', 'Overstrain', 'Tool wear', 'No mechanism recorded']
N_TEST, N_TRAIN = 1500, 7000
REC_TARGETS, FPR_TARGETS = (0.80, 0.90), (0.01, 0.02)


def main(argv=None):
    p = argparse.ArgumentParser(description='Build all result tables and figures from archived predictions.')
    p.add_argument('--run', default='results/predictions', help='folder with pred_sXX_<strategy>.csv and settings.csv')
    p.add_argument('--out', default='analysis_out', help='output folder')
    a = p.parse_args(argv)
    RUN, OUT = a.run, a.out
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f'{OUT}/figures', exist_ok=True)

    # ---------------------------------------------------------------- load
    files = sorted(glob.glob(f'{RUN}/pred_s*_*.csv'))
    if files:
        P = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:                                   # archive stored as one zip (results/predictions/predictions.zip)
        import zipfile
        with zipfile.ZipFile(f'{RUN}/predictions.zip') as z:
            names = sorted(n for n in z.namelist() if os.path.basename(n).startswith('pred_s') and n.endswith('.csv'))
            P = pd.concat([pd.read_csv(z.open(n)) for n in names], ignore_index=True)
    P['method'] = P['method'].astype(str)
    splits = sorted(P['split'].unique())
    grid = P.groupby(['split', 'method']).size().unstack()
    missing = [(s, m) for s in splits for m in METHODS if m not in grid.columns or pd.isna(grid.loc[s].get(m))]
    complete = [s for s in splits if all((s, m) not in missing for m in METHODS)]
    J = len(complete)
    settings = pd.read_csv(f'{RUN}/settings.csv') if os.path.exists(f'{RUN}/settings.csv') else None
    versions = json.load(open(f'{RUN}/versions.json')) if os.path.exists(f'{RUN}/versions.json') else {}
    errors = open(f'{RUN}/errors.log').read().strip() if os.path.exists(f'{RUN}/errors.log') else ''
    P = P[P['split'].isin(complete)]

    # ---------------------------------------------------------------- thresholds (chosen on validation)
    def thr_f1(y, p):
        pr, rc, th = precision_recall_curve(y, p)
        f1 = np.where(pr[:-1] + rc[:-1] > 0, 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12), 0)
        return th[int(np.argmax(f1))]

    def thr_recall(y, p, target):
        pr, rc, th = precision_recall_curve(y, p)          # rc decreasing along th increasing
        ok = np.where(rc[:-1] >= target)[0]
        return th[ok.max()] if len(ok) else th.min()       # highest threshold still reaching target

    def thr_fpr(y, p, target):
        fpr, tpr, th = roc_curve(y, p)                      # th decreasing
        ok = np.where(fpr <= target)[0]
        t = th[ok.max()] if len(ok) else th[0]
        return min(t, 1.0) if np.isfinite(t) else 1.0      # lowest threshold keeping FPR <= target

    def at(y, p, t):
        yh = (p >= t).astype(int)
        tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum())
        fn = int(((yh == 0) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return dict(thr=t, precision=prec, recall=rec, f1=f1, fpr=fp / (fp + tn) if fp + tn else 0.0,
                    tp=tp, fp=fp, fn=fn, tn=tn, n=len(y))

    OPS = {'t0.5': None, 'F1opt': None}
    OPS.update({f'rec{int(r*100)}': r for r in REC_TARGETS}); OPS.update({f'fpr{int(f*100)}': f for f in FPR_TARGETS})

    rows, mech_rows = [], []
    for (s, m), g in P.groupby(['split', 'method']):
        v, t = g[g['set'] == 'val'], g[g['set'] == 'test']
        yv, pv, yt, pt = v['y'].values, v['prob'].values, t['y'].values, t['prob'].values
        rec = dict(split=s, method=m, PR_AUC=average_precision_score(yt, pt), ROC_AUC=roc_auc_score(yt, pt))
        for op in OPS:
            if op == 't0.5': th = 0.5
            elif op == 'F1opt': th = thr_f1(yv, pv)
            elif op.startswith('rec'): th = thr_recall(yv, pv, OPS[op])
            else: th = thr_fpr(yv, pv, OPS[op])
            r = at(yt, pt, th)
            for k2, val in r.items(): rec[f'{op}_{k2}'] = val
            # per mechanism (test failures only)
            tf = t[t['y'] == 1]
            for ft, lab in MECH.items():
                sub = tf[tf['failure_type'] == ft]
                if len(sub) == 0: continue
                det = int((sub['prob'].values >= th).sum())
                mech_rows.append(dict(split=s, method=m, op=op, mechanism=lab, support=len(sub), detected=det,
                                      missed=len(sub) - det, recall=det / len(sub)))
        rows.append(rec)
    R = pd.DataFrame(rows); M = pd.DataFrame(mech_rows)
    R.to_csv(f'{OUT}/per_split_metrics.csv', index=False); M.to_csv(f'{OUT}/per_split_mechanism.csv', index=False)

    # ---------------------------------------------------------------- helpers
    def corrected_ci(d, alpha=0.05):
        d = np.asarray(d, float); j = len(d); mean = d.mean(); s2 = d.var(ddof=1)
        se = math.sqrt((1 / j + N_TEST / N_TRAIN) * s2); tq = stats.t.ppf(1 - alpha / 2, j - 1)
        tstat = mean / se if se > 0 else np.inf
        p = 2 * stats.t.sf(abs(tstat), j - 1) if se > 0 else 0.0
        return mean, mean - tq * se, mean + tq * se, p

    def holm(pvals):
        p = np.asarray(pvals); o = np.argsort(p); adj = np.empty_like(p); run = 0
        for i, idx in enumerate(o):
            run = max(run, (len(p) - i) * p[idx]); adj[idx] = min(run, 1.0)
        return adj

    def wide(metric):
        return R.pivot(index='split', columns='method', values=metric)[METHODS]

    def friedman_nemenyi(metric, higher_better=True):
        W = wide(metric); X = W.values if higher_better else -W.values
        ranks = np.vstack([stats.rankdata(-row) for row in X])   # rank 1 = best
        mr = pd.Series(ranks.mean(0), index=METHODS)
        chi2, p = stats.friedmanchisquare(*[X[:, i] for i in range(X.shape[1])])
        k, n = len(METHODS), X.shape[0]
        se = math.sqrt(k * (k + 1) / (6 * n)); q = studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2)
        cd = q * se
        pairs = {}
        for a in METHODS:
            for b in METHODS:
                z = abs(mr[a] - mr[b]) / se
                pairs[(a, b)] = studentized_range.sf(z * math.sqrt(2), k, np.inf)
        return mr, chi2, p, cd, pairs

    # ---------------------------------------------------------------- Table: repeated-split performance
    measures = [('PR_AUC', 'PR-AUC'), ('ROC_AUC', 'ROC-AUC'),
                ('t0.5_precision', 'Precision @0.5'), ('t0.5_recall', 'Recall @0.5'), ('t0.5_f1', 'F1 @0.5'),
                ('F1opt_precision', 'Precision @F1-opt'), ('F1opt_recall', 'Recall @F1-opt'), ('F1opt_f1', 'F1 @F1-opt')]
    perf = []
    for m in METHODS:
        g = R[R['method'] == m]; row = {'Strategy': m}
        for c, lab in measures:
            row[f'{lab} mean'] = g[c].mean(); row[f'{lab} SD'] = g[c].std(ddof=1)
        perf.append(row)
    perf = pd.DataFrame(perf)
    fr = {}
    for c, lab in [('PR_AUC', 'PR-AUC'), ('ROC_AUC', 'ROC-AUC'), ('F1opt_f1', 'F1 @F1-opt'), ('t0.5_f1', 'F1 @0.5'),
                   ('fpr1_recall', 'Recall @FPR 1%'), ('fpr2_recall', 'Recall @FPR 2%'), ('rec90_precision', 'Precision @recall 0.90')]:
        mr, chi2, p, cd, pairs = friedman_nemenyi(c)
        fr[lab] = dict(mean_ranks=mr.round(3).to_dict(), chi2=float(chi2), p=float(p), CD=float(cd),
                       nemenyi_vs_focal={b: float(pairs[(FOCAL, b)]) for b in METHODS if b != FOCAL})
        perf[f'Mean rank {lab}'] = perf['Strategy'].map(mr)
    perf = perf.sort_values('PR-AUC mean', ascending=False)

    # ---------------------------------------------------------------- Table: paired differences (focal - other)
    diffs = []
    for c, lab in [('PR_AUC', 'PR-AUC'), ('ROC_AUC', 'ROC-AUC'), ('F1opt_f1', 'F1 @F1-opt'),
                   ('fpr1_recall', 'Recall @FPR 1%'), ('fpr2_recall', 'Recall @FPR 2%'),
                   ('rec80_precision', 'Precision @recall 0.80'), ('rec90_precision', 'Precision @recall 0.90'),
                   ('t0.5_recall', 'Recall @0.5')]:
        W = wide(c); block = []
        for b in METHODS:
            if b == FOCAL: continue
            d = (W[FOCAL] - W[b]).values
            mean, lo, hi, pc = corrected_ci(d)
            try: pw = stats.wilcoxon(W[FOCAL], W[b], zero_method='wilcox').pvalue
            except ValueError: pw = 1.0
            dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
            block.append(dict(Measure=lab, Comparison=f'{FOCAL} − {b}', Other=b, Mean_diff=mean, CI_low=lo, CI_high=hi,
                              p_corrected_t=pc, Significant=bool(lo > 0 or hi < 0), Wilcoxon_p=pw, d_z=dz,
                              Wins=int((d > 0).sum()), Ties=int((d == 0).sum()), Losses=int((d < 0).sum())))
        adj = holm([x['Wilcoxon_p'] for x in block])
        for x, a in zip(block, adj): x['Wilcoxon_p_Holm'] = a
        diffs += block
    diffs = pd.DataFrame(diffs)

    # ---------------------------------------------------------------- Table: operating points + workload per 1,000
    opr = []
    for op in ['rec80', 'rec90', 'fpr1', 'fpr2', 'F1opt', 't0.5']:
        for m in METHODS:
            g = R[R['method'] == m]; k = 1000 / g[f'{op}_n']
            opr.append({'Operating point': op, 'Strategy': m,
                        'Recall': g[f'{op}_recall'].mean(), 'Precision': g[f'{op}_precision'].mean(),
                        'FPR': g[f'{op}_fpr'].mean(), 'F1': g[f'{op}_f1'].mean(),
                        'Detected /1000': (g[f'{op}_tp'] * k).mean(), 'Missed /1000': (g[f'{op}_fn'] * k).mean(),
                        'False alarms /1000': (g[f'{op}_fp'] * k).mean(),
                        'Inspections /1000': ((g[f'{op}_tp'] + g[f'{op}_fp']) * k).mean()})
    opr = pd.DataFrame(opr)

    # ---------------------------------------------------------------- Table: per-mechanism
    mech_tab, mech_diff = [], []
    for op in ['F1opt', 'rec90', 't0.5']:
        for mech in MECH_ORDER:
            for m in METHODS:
                g = M[(M['op'] == op) & (M['mechanism'] == mech) & (M['method'] == m)]
                if g.empty: continue
                vals = g['recall'].values; mean = vals.mean()
                if len(vals) > 1 and vals.std(ddof=1) > 0:
                    se = math.sqrt((1 / len(vals) + N_TEST / N_TRAIN) * vals.var(ddof=1)); tq = stats.t.ppf(.975, len(vals) - 1)
                    lo, hi = max(0, mean - tq * se), min(1, mean + tq * se)
                else: lo = hi = mean
                mech_tab.append({'Operating point': op, 'Mechanism': mech, 'Strategy': m,
                                 'Support per split (mean)': g['support'].mean(), 'Support total': int(g['support'].sum()),
                                 'Recall mean': mean, 'CI low': lo, 'CI high': hi,
                                 'Pooled recall': g['detected'].sum() / g['support'].sum(), 'Missed total': int(g['missed'].sum())})
            Wm = M[(M['op'] == op) & (M['mechanism'] == mech)].pivot(index='split', columns='method', values='recall')
            for b in METHODS:
                if b == FOCAL or b not in Wm or FOCAL not in Wm: continue
                d = (Wm[FOCAL] - Wm[b]).dropna().values
                if len(d) < 3: continue
                mean, lo, hi, pc = corrected_ci(d) if d.std(ddof=1) > 0 else (d.mean(), d.mean(), d.mean(), 1.0)
                mech_diff.append({'Operating point': op, 'Mechanism': mech, 'Comparison': f'{FOCAL} − {b}',
                                  'Mean diff': mean, 'CI low': lo, 'CI high': hi, 'Significant': bool(lo > 0 or hi < 0)})
    mech_tab, mech_diff = pd.DataFrame(mech_tab), pd.DataFrame(mech_diff)

    # ---------------------------------------------------------------- settings + compute
    set_tab, comp = pd.DataFrame(), pd.DataFrame()
    if settings is not None:
        S = settings[settings['split'].isin(complete)].drop_duplicates(['split', 'method'], keep='last')
        st = []
        for m in METHODS:
            g = S[S['method'] == m]
            st.append({'Strategy': m,
                       'k (counts)': ', '.join(f'{int(k)}: {n}' for k, n in g['k'].dropna().value_counts().sort_index().items()) or '—',
                       'max_depth (counts)': ', '.join(f'{int(k)}: {n}' for k, n in g['max_depth'].value_counts().sort_index().items()),
                       'min_samples_leaf (counts)': ', '.join(f'{int(k)}: {n}' for k, n in g['min_samples_leaf'].value_counts().sort_index().items())})
            comp_row = {'Strategy': m, 'Seconds per split (mean)': g['fit_seconds'].mean(), 'SD': g['fit_seconds'].std(ddof=1),
                        'Total hours': g['fit_seconds'].sum() / 3600}
            comp = pd.concat([comp, pd.DataFrame([comp_row])])
        set_tab = pd.DataFrame(st)

    # ---------------------------------------------------------------- write tables
    with pd.ExcelWriter(f'{OUT}/revision_results_tables.xlsx') as xw:
        perf.to_excel(xw, sheet_name='Performance', index=False); diffs.to_excel(xw, sheet_name='Paired_differences', index=False)
        opr.to_excel(xw, sheet_name='Operating_points_workload', index=False); mech_tab.to_excel(xw, sheet_name='Per_mechanism', index=False)
        mech_diff.to_excel(xw, sheet_name='Per_mechanism_diffs', index=False); set_tab.to_excel(xw, sheet_name='Selected_settings', index=False)
        comp.to_excel(xw, sheet_name='Compute_time', index=False)
        pd.DataFrame([{'measure': k, 'chi2': v['chi2'], 'p': v['p'], 'CD': v['CD'], **{f'rank {m}': v['mean_ranks'][m] for m in METHODS}}
                      for k, v in fr.items()]).to_excel(xw, sheet_name='Friedman_Nemenyi', index=False)
    for name, df in [('performance', perf), ('paired_differences', diffs), ('operating_points', opr),
                     ('per_mechanism', mech_tab), ('per_mechanism_diffs', mech_diff), ('settings', set_tab), ('compute', comp)]:
        df.to_csv(f'{OUT}/{name}.csv', index=False)

    # ---------------------------------------------------------------- figures
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#d9d8d4', '#ffffff'
    HL = {'Multi WGAN-GP': '#2a78d6', 'Single WGAN-GP': '#eb6834'}; OTHER = '#8a8984'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.dpi': 300})
    col = lambda m: HL.get(m, OTHER)
    def save(fig, name):
        fig.savefig(f'{OUT}/figures/{name}.png', bbox_inches='tight'); fig.savefig(f'{OUT}/figures/{name}.pdf', bbox_inches='tight'); plt.close(fig)

    # (a) critical-difference diagram, PR-AUC
    mr = pd.Series(fr['PR-AUC']['mean_ranks']).sort_values(); cd = fr['PR-AUC']['CD']; k = len(METHODS)
    fig, ax = plt.subplots(figsize=(7.2, 2.9)); ax.set_xlim(0.5, k + 0.5); ax.set_ylim(0, 1); ax.axis('off')
    ax.plot([1, k], [0.80, 0.80], color=INK2, lw=1)
    for r in range(1, k + 1):
        ax.plot([r, r], [0.80, 0.83], color=INK2, lw=1); ax.text(r, 0.86, str(r), ha='center', va='bottom', color=INK2)
    ax.plot([1, 1 + cd], [0.97, 0.97], color=INK, lw=2); ax.text(1 + cd / 2, 0.99, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=8)
    left, right = list(mr.index[: k // 2]), list(mr.index[k // 2:])
    for i, m in enumerate(left):
        y = 0.68 - i * 0.14; ax.plot([mr[m], mr[m], 0.9], [0.80, y, y], color=col(m) if m in HL else INK2, lw=1.2)
        ax.text(0.85, y, f'{m} ({mr[m]:.2f})', ha='right', va='center', color=INK)
    for i, m in enumerate(right):
        y = 0.68 - i * 0.14; ax.plot([mr[m], mr[m], k + 0.1], [0.80, y, y], color=col(m) if m in HL else INK2, lw=1.2)
        ax.text(k + 0.15, y, f'({mr[m]:.2f}) {m}', ha='left', va='center', color=INK)
    # cliques: maximal groups with rank span < CD
    ms = list(mr.index); cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and mr[ms[j + 1]] - mr[ms[i]] < cd: j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques): cliques.append((i, j))
    for n, (a, b) in enumerate(cliques):
        y = 0.74 - n * 0.035; ax.plot([mr[ms[a]] - 0.03, mr[ms[b]] + 0.03], [y, y], color=INK, lw=3, solid_capstyle='round')
    save(fig, 'Fig_CD_PR_AUC')

    # (b) PR-AUC across splits, per strategy
    order = list(perf['Strategy']); W = wide('PR_AUC')
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for i, m in enumerate(order):
        v = W[m].values; jit = (np.random.RandomState(i).rand(len(v)) - 0.5) * 0.28
        ax.scatter(np.full(len(v), i) + jit, v, s=14, color=col(m), alpha=0.75, edgecolor=SURF, linewidth=0.6, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=2, solid_capstyle='round', zorder=4)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha='right')
    ax.set_ylabel('Test PR-AUC (one point per split)'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, 'Fig_PR_AUC_by_split')

    # (c) paired differences, focal minus each, PR-AUC with corrected CIs
    D = diffs[diffs['Measure'] == 'PR-AUC'].iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.axvline(0, color=INK2, lw=1)
    for i, (_, r) in enumerate(D.iterrows()):
        ax.plot([r.CI_low, r.CI_high], [i, i], color=INK, lw=1.6, solid_capstyle='round')
        ax.scatter([r.Mean_diff], [i], s=36, color='#2a78d6', edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(D))); ax.set_yticklabels([f'vs {o}' for o in D['Other']])
    ax.set_xlabel('Difference in PR-AUC, Multi WGAN-GP minus comparator\n(mean across 30 splits, corrected 95% CI)')
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, 'Fig_paired_diff_PR_AUC')

    # (d) per-mechanism recall at matched recall 0.90 (small multiples)
    T = mech_tab[mech_tab['Operating point'] == 'rec90']
    mechs = [m for m in MECH_ORDER[:4] if m in set(T['Mechanism'])]
    fig, axs = plt.subplots(1, len(mechs), figsize=(9.0, 3.4), sharey=True)
    for ax, mech in zip(np.atleast_1d(axs), mechs):
        g = T[T['Mechanism'] == mech].set_index('Strategy').reindex(order[::-1])
        for i, (m, r) in enumerate(g.iterrows()):
            ax.plot([r['CI low'], r['CI high']], [i, i], color=INK2, lw=1.4, solid_capstyle='round')
            ax.scatter([r['Recall mean']], [i], s=30, color=col(m), edgecolor=SURF, linewidth=1.2, zorder=3)
        sup = g['Support per split (mean)'].mean()
        ax.set_title(f'{mech}\n(≈{sup:.0f} test failures per split)', fontsize=9, color=INK)
        ax.set_xlim(0, 1.02); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    np.atleast_1d(axs)[0].set_yticks(range(len(order))); np.atleast_1d(axs)[0].set_yticklabels(order[::-1])
    fig.supxlabel('Recall for the mechanism at a validation-chosen overall recall of 0.90 (mean, corrected 95% CI)', fontsize=9)
    save(fig, 'Fig_per_mechanism_recall_rec90')

    # ---------------------------------------------------------------- summary
    summary = dict(splits_found=len(splits), splits_complete=J, missing=missing, errors_log=errors[:2000], versions=versions,
                   friedman=fr, performance=perf.round(4).to_dict('records'),
                   compute=comp.round(2).to_dict('records') if len(comp) else [])
    json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
    print(f'splits complete: {J}/{len(splits)}; missing: {missing[:10]}')
    print(perf[['Strategy', 'PR-AUC mean', 'PR-AUC SD', 'Mean rank PR-AUC']].round(4).to_string(index=False))
    print(diffs[diffs['Measure'] == 'PR-AUC'][['Comparison', 'Mean_diff', 'CI_low', 'CI_high', 'Significant', 'Wins', 'Losses']].round(4).to_string(index=False))
    print('Friedman PR-AUC chi2=%.2f p=%.2g CD=%.3f' % (fr['PR-AUC']['chi2'], fr['PR-AUC']['p'], fr['PR-AUC']['CD']))

    return summary


if __name__ == '__main__':
    main()
