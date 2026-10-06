"""Stage 3 reproduces the paper's numbers from the archived predictions (CPU only, about a minute)."""
import os

import pandas as pd
import pytest

from conftest import RESULTS
from ftaa import analysis

PRED = os.path.join(RESULTS, 'predictions')
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(PRED, 'settings.csv')), reason='archive not present')


@pytest.fixture(scope='module')
def out(tmp_path_factory):
    o = str(tmp_path_factory.mktemp('analysis'))
    analysis.main(['--run', PRED, '--out', o])
    return o


def test_all_240_runs_complete(out):
    m = pd.read_csv(f'{out}/per_split_metrics.csv')
    assert len(m) == 240 and m.groupby('method').size().eq(30).all()


def test_table3_pr_auc(out):
    perf = pd.read_csv(f'{out}/performance.csv').set_index('Strategy')
    paper = {'Baseline': 0.818, 'Borderline-SMOTE': 0.806, 'Multi WGAN-GP': 0.803, 'SMOTE': 0.797,
             'ADASYN': 0.791, 'SMOTENC': 0.786, 'Single WGAN-GP': 0.770, 'RUS': 0.695}
    for m, v in paper.items():
        assert round(perf.loc[m, 'PR-AUC mean'], 3) == v
    assert round(perf.loc['Baseline', 'PR-AUC SD'], 3) == 0.032
    assert round(perf.loc['Baseline', 'Mean rank PR-AUC'], 2) == 1.97


def test_friedman_and_cd(out):
    import json
    fr = json.load(open(f'{out}/summary.json'))['friedman']['PR-AUC']
    assert round(fr['chi2'], 2) == 115.36 and round(fr['CD'], 2) == 1.92
    assert round(fr['nemenyi_vs_focal']['Single WGAN-GP'], 3) == 0.014


def test_table4_paired_differences(out):
    d = pd.read_csv(f'{out}/paired_differences.csv')
    d = d[d['Measure'] == 'PR-AUC'].set_index('Other')
    assert [round(d.loc['Baseline', c], 3) for c in ('Mean_diff', 'CI_low', 'CI_high')] == [-0.015, -0.051, 0.021]
    assert [round(d.loc['Single WGAN-GP', c], 3) for c in ('Mean_diff', 'CI_low', 'CI_high')] == [0.033, -0.006, 0.073]
    assert d.loc['Single WGAN-GP', 'Wins'] == 23 and round(d.loc['Single WGAN-GP', 'd_z'], 2) == 0.85
    assert bool(d.loc['RUS', 'Significant']) and d['Significant'].sum() == 1


def test_table5_tool_wear(out):
    t = pd.read_csv(f'{out}/per_mechanism.csv')
    t = t[(t['Operating point'] == 'rec90') & (t['Mechanism'] == 'Tool wear')].set_index('Strategy')
    assert round(t.loc['Multi WGAN-GP', 'Recall mean'], 2) == 0.40 and t.loc['Multi WGAN-GP', 'Missed total'] == 126
    assert round(t.loc['Baseline', 'Recall mean'], 2) == 0.50 and round(t.loc['RUS', 'Recall mean'], 2) == 0.59


def test_table6_workload(out):
    o = pd.read_csv(f'{out}/operating_points.csv')
    r = o[o['Operating point'] == 'rec90'].set_index('Strategy')
    assert round(r.loc['Baseline', 'False alarms /1000'], 1) == 58.1 and round(r.loc['Baseline', 'Inspections /1000'], 1) == 88.5
    assert round(r.loc['Multi WGAN-GP', 'False alarms /1000'], 1) == 56.8 and round(r.loc['Multi WGAN-GP', 'Inspections /1000'], 1) == 86.9
    f = o[o['Operating point'] == 'fpr1'].set_index('Strategy')
    assert [round(f.loc[m, 'Recall'], 3) for m in ('Baseline', 'Borderline-SMOTE', 'Multi WGAN-GP')] == [0.746, 0.734, 0.716]


def test_table7_compute(out):
    c = pd.read_csv(f'{out}/compute.csv').set_index('Strategy')
    sec = c['Seconds per split (mean)']          # Table 7 reports 2.6, 223.1 and 530.7 (rounded half up)
    assert abs(sec['Baseline'] - 2.58) < 0.005
    assert abs(sec['Single WGAN-GP'] - 223.05) < 0.005
    assert abs(sec['Multi WGAN-GP'] - 530.68) < 0.005
    assert round(c['Total hours'].sum(), 1) == 6.7
