"""Allocation logic of the two WGAN-GP samplers, with the generator replaced by a fast fake.

The fake returns the requested number of records (len(train_df) * gen_x_times, rounded up) by resampling the
generator's own training rows, so the tests check bookkeeping, not generative quality."""
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('imblearn')
from ftaa import config as C, data as D, samplers  # noqa: E402

CALLS = []


class FakeGAN:
    def __init__(self, gen_x_times, cat_cols, is_post_process, adversarial_model_params, gen_params):
        self.x = gen_x_times
        self.gen_params = gen_params

    def generate_data_pipe(self, train_df, target, test_df, only_generated_data):
        n = math.ceil(len(train_df) * self.x)
        CALLS.append((len(train_df), n, self.gen_params['batch_size']))
        return train_df.sample(n=n, replace=True, random_state=0).reset_index(drop=True), None


@pytest.fixture(autouse=True)
def fake_gan(monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(samplers, 'GANGenerator', FakeGAN)


def _train(fake_csv):
    df, X, y = D.load(fake_csv)
    tr, _, _ = D.split_indices(len(df), D.strata(df), 1)
    return X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)


def test_multi_allocates_exactly_to_balance_and_by_mechanism(fake_csv):
    X, y = _train(fake_csv)
    s = samplers.SpecializedTabGANSampler(C.gen_params(C.MULTI, 1), C.adv_params(C.MULTI, 1), ['Type'], 0.5, random_state=1)
    Xa, ya = s.fit_resample(X, y)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    assert int((ya == 1).sum()) == n0                                   # exactly 1:1
    syn = s.last_synth_df_
    assert len(syn) == n0 - n1
    assert len(CALLS) == 4                                              # one generator per recorded mechanism
    assert 'No Failure' not in set(syn[C.FAILURE_COL])                  # unlabeled failures train no generator
    train_counts = X[y == 1][C.FAILURE_COL].value_counts()
    labeled = train_counts.drop('No Failure', errors='ignore')
    for ft, n in labeled.items():                                       # proportional, largest remainder
        share = (n0 - n1) * n / labeled.sum()
        assert abs((syn[C.FAILURE_COL] == ft).sum() - share) < 1
    assert all(bs % 10 == 0 for _, _, bs in CALLS)


def test_single_uses_all_failures_and_ignores_mechanism(fake_csv):
    X, y = _train(fake_csv)
    s = samplers.OneTabGANSampler(C.gen_params(C.SINGLE, 1), C.adv_params(C.SINGLE, 1), ['Type'], 0.5, random_state=1)
    Xa, ya = s.fit_resample(X, y)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    assert len(CALLS) == 1 and CALLS[0][0] == n1                        # one generator on all training failures
    assert int((np.asarray(ya) == 1).sum()) == n0
    assert s.last_synth_df_[C.FAILURE_COL].isna().all()
    assert list(Xa.columns) == list(X.columns)
