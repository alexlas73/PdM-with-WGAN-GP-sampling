"""End-to-end smoke test of Stage 2 on FAKE data: Baseline, RUS and SMOTE on one split (about a minute, CPU).
The two GAN strategies are covered by test_samplers.py with a fake generator."""
import glob

import pandas as pd
import pytest

pytest.importorskip('imblearn')
from ftaa import experiment  # noqa: E402


def test_one_split_fast_strategies(fake_csv, tmp_path):
    out = str(tmp_path / 'run')
    experiment.main(['--data', fake_csv, '--out', out, '--splits', '1',
                     '--methods', 'Baseline', 'RUS', 'SMOTE', '--n-jobs', '2'])
    files = sorted(glob.glob(f'{out}/pred_s01_*.csv'))
    assert len(files) == 3
    p = pd.read_csv(files[0])
    assert list(p.columns) == ['split', 'method', 'set', 'row', 'y', 'failure_type', 'prob']
    assert (p['set'] == 'val').sum() == 1500 and (p['set'] == 'test').sum() == 1500
    s = pd.read_csv(f'{out}/settings.csv')
    assert len(s) == 3 and s.loc[s['method'] == 'SMOTE', 'k'].iloc[0] in (3, 5, 7)
    # resumable: a second call does nothing
    experiment.main(['--data', fake_csv, '--out', out, '--splits', '1', '--methods', 'Baseline', '--n-jobs', '2'])
    assert len(pd.read_csv(f'{out}/settings.csv')) == 3
