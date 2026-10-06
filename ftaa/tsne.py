"""Illustrative t-SNE projections of one split's training data (the paper's Figures 9 and 10).

python -m ftaa.tsne --data data/predictive_maintenance.csv --settings results/predictions/settings.csv \
                    --split 1 --out results/figures

Re-creates every strategy's resampled / augmented training partition for the split with the split's seed and the
neighbourhood size k selected on that split (read from settings.csv), retrains both generators (about 12 minutes
on a T4 GPU), embeds a joint sample in one t-SNE (perplexity 30, PCA initialization), and saves
Fig9_tsne_resampling_split<S>.png/.pdf, Fig10_tsne_generative_split<S>.png/.pdf and the plotted coordinates.
GPU training is not bit-for-bit deterministic, so regenerated synthetic records (and hence the GAN panels) can
differ slightly from the published figures; tsne_split<S>_coordinates.csv in results/figures holds the published
coordinates.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from . import compat
from . import config as C
from . import data as D

compat.apply()
N_NORMAL = 1500        # random sample of normal-operation records plotted (plus those retained by RUS)
N_SYN = 600            # synthetic records drawn per strategy for the plot


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--settings', default='results/predictions/settings.csv')
    p.add_argument('--split', type=int, default=1)
    p.add_argument('--out', default='results/figures')
    a = p.parse_args(argv)
    from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from .samplers import OneTabGANSampler, SpecializedTabGANSampler
    S, out = a.split, a.out
    os.makedirs(out, exist_ok=True)
    t_all = time.perf_counter()

    df, X_all, y_all = D.load(a.data)
    strat = D.strata(df)
    tr, _, _ = D.split_indices(len(df), strat, S)
    Xtr = X_all.iloc[tr].reset_index(drop=True); ytr = y_all.iloc[tr].reset_index(drop=True)
    P = D.make_prep().fit(Xtr)                              # one preprocessor for every panel
    Z = P.transform(Xtr)
    ft_tr = Xtr[C.FAILURE_COL].values
    settings = pd.read_csv(a.settings)
    k_of = settings[settings['split'] == S].set_index('method')['k'].to_dict()
    print('training partition:', len(Xtr), 'records,', int(ytr.sum()), 'failures')

    def to_Z(synth_df):
        d = synth_df.copy()
        for c in C.NUMERICAL_FEATURES: d[c] = pd.to_numeric(d[c], errors='coerce')
        d = d[d[C.NUMERICAL_FEATURES].notna().all(axis=1) & d['Type'].isin(C.TYPE_ORDER)]
        return P.transform(d[C.NUMERICAL_FEATURES + C.CATEGORICAL_FEATURES]), d

    synth = {}
    np.random.seed(S)
    rus = RandomUnderSampler(random_state=S); rus.fit_resample(Z, ytr.values)
    rus_keep = set(rus.sample_indices_)
    for name, cls in [('SMOTE', SMOTE), ('SMOTENC', SMOTENC), ('Borderline-SMOTE', BorderlineSMOTE), ('ADASYN', ADASYN)]:
        k = int(k_of[name])
        if name == 'SMOTENC': sm = cls(categorical_features=D.CAT_IDX, k_neighbors=k, random_state=S)
        elif name == 'ADASYN': sm = cls(n_neighbors=k, random_state=S)
        else: sm = cls(k_neighbors=k, random_state=S)
        Zr, _ = sm.fit_resample(Z, ytr.values)
        synth[name] = (Zr[len(Z):], None); print(f'{name:17s} k={k}: {len(Zr) - len(Z)} synthetic records')
    for name, cls, cfg in [('Single WGAN-GP', OneTabGANSampler, C.SINGLE), ('Multi WGAN-GP', SpecializedTabGANSampler, C.MULTI)]:
        t0 = time.perf_counter(); np.random.seed(S)
        try:
            import torch; torch.manual_seed(S)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(S)
        except Exception: pass
        smp = cls(gen_params=C.gen_params(cfg, S), adv_params=C.adv_params(cfg, S), categorical_features=C.CATEGORICAL_FEATURES,
                  target_pos_ratio=C.TARGET_POS_RATIO, random_state=S)
        smp.fit_resample(Xtr, ytr)
        Zs, d = to_Z(smp.last_synth_df_)
        mech = d[C.FAILURE_COL].values if name.startswith('Multi') else None
        synth[name] = (Zs, mech); print(f'{name:17s}: {len(Zs)} synthetic records ({time.perf_counter() - t0:.0f} s)')

    rng = np.random.RandomState(S)
    maj_idx = np.where(ytr.values == 0)[0]; fail_idx = np.where(ytr.values == 1)[0]
    base_maj = np.union1d(rng.choice(maj_idx, N_NORMAL, replace=False), np.array(sorted(rus_keep & set(maj_idx))))
    parts = [('orig_maj', Z[base_maj], None, base_maj), ('orig_fail', Z[fail_idx], ft_tr[fail_idx], fail_idx)]
    for name, (Zs, mech) in synth.items():
        pick = rng.choice(len(Zs), min(N_SYN, len(Zs)), replace=False)
        parts.append((name, Zs[pick], None if mech is None else mech[pick], None))
    pool = np.vstack([q[1] for q in parts]); tag = np.concatenate([[q[0]] * len(q[1]) for q in parts])
    E = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=S).fit_transform(pool)
    seg = {}; o = 0
    for q in parts: seg[q[0]] = (E[o:o + len(q[1])], q[2], q[3]); o += len(q[1])

    # ---- plotting ----
    MECH = ['Heat Dissipation Failure', 'Power Failure', 'Overstrain Failure', 'Tool Wear Failure', 'No Failure']
    LAB = {'Heat Dissipation Failure': 'Heat dissipation', 'Power Failure': 'Power', 'Overstrain Failure': 'Overstrain',
           'Tool Wear Failure': 'Tool wear', 'No Failure': 'No mechanism recorded'}
    COL = {'Heat Dissipation Failure': '#2a78d6', 'Power Failure': '#eb6834', 'Overstrain Failure': '#1baf7a',
           'Tool Wear Failure': '#eda100', 'No Failure': '#e87ba4'}
    GREY, SYN = '#c9c8c3', '#3d3d3a'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#52514e',
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})

    def base(ax, keep=None):
        Em, _, ix = seg['orig_maj']
        m = np.ones(len(Em), bool) if keep is None else np.isin(ix, list(keep))
        ax.scatter(Em[m, 0], Em[m, 1], s=4, color=GREY, alpha=0.6, linewidths=0, label='Normal operation (sample)')
        Ef, mech, _ = seg['orig_fail']
        for k in MECH:
            sel = mech == k
            if sel.any(): ax.scatter(Ef[sel, 0], Ef[sel, 1], s=22, marker='x', color=COL[k], linewidths=1.0, label=f'{LAB[k]} (original)')
        ax.set_xticks([]); ax.set_yticks([])

    def synthetic(ax, name):
        Es, mech, _ = seg[name]
        if mech is None:
            ax.scatter(Es[:, 0], Es[:, 1], s=9, color=SYN, alpha=0.55, linewidths=0, label='Synthetic failure (mechanisms pooled)')
        else:
            for k in MECH:
                sel = mech == k
                if sel.any(): ax.scatter(Es[sel, 0], Es[sel, 1], s=9, color=COL[k], alpha=0.5, linewidths=0, label=f'{LAB[k]} (synthetic)')

    # Figure 9: original training partition, RUS, and the four interpolation-based strategies
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.8)); axs = axs.ravel()
    base(axs[0]); axs[0].set_title('Training partition (no resampling)')
    base(axs[1], keep=rus_keep); axs[1].set_title('RUS')
    for ax, name in zip(axs[2:], ['SMOTE', 'SMOTENC', 'Borderline-SMOTE', 'ADASYN']):
        base(ax); synthetic(ax, name); ax.set_title(name)
    h, l = axs[2].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f'{out}/Fig9_tsne_resampling_split{S}.png', bbox_inches='tight'); fig.savefig(f'{out}/Fig9_tsne_resampling_split{S}.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 10: single-generator vs failure-type-aware augmentation
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, name in zip(axs, ['Single WGAN-GP', 'Multi WGAN-GP']):
        base(ax); synthetic(ax, name); ax.set_title(name)
    h0, l0 = axs[0].get_legend_handles_labels(); h1, l1 = axs[1].get_legend_handles_labels()
    hh, ll = [], []
    for h, l in list(zip(h0, l0)) + list(zip(h1, l1)):
        if l not in ll: hh.append(h); ll.append(l)
    fig.legend(hh, ll, loc='lower center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(f'{out}/Fig10_tsne_generative_split{S}.png', bbox_inches='tight'); fig.savefig(f'{out}/Fig10_tsne_generative_split{S}.pdf', bbox_inches='tight')
    plt.close(fig)


    mech_col = np.concatenate([np.full(len(q[1]), '', dtype=object) if q[2] is None else np.asarray(q[2], dtype=object) for q in parts])
    pd.DataFrame({'x': E[:, 0], 'y': E[:, 1], 'set': tag, 'mechanism': mech_col}).to_csv(f'{out}/tsne_split{S}_coordinates.csv', index=False)
    print(f'Saved Figures 9-10 and coordinates to {out} ({(time.perf_counter() - t_all) / 60:.1f} min)')


if __name__ == '__main__':
    main()
