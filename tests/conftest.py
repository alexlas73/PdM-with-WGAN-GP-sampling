"""Shared fixtures. FAKE data only: a synthetic file with the AI4I schema and the same label counts, so the
pipeline can be tested without downloading the real dataset. Never use it for results."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, 'results')

COUNTS = [('No Failure', 0, 9643), ('No Failure', 1, 9), ('Random Failures', 0, 18),
          ('Heat Dissipation Failure', 1, 112), ('Power Failure', 1, 95),
          ('Overstrain Failure', 1, 78), ('Tool Wear Failure', 1, 45)]


def make_fake_ai4i(path, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for ft, t, n in COUNTS:
        for _ in range(n):
            rows.append((ft, t))
    rng.shuffle(rows)
    n = len(rows)
    ft = np.array([r[0] for r in rows]); target = np.array([r[1] for r in rows])
    air = rng.normal(300, 2, n); proc = air + rng.normal(10, 1, n)
    speed = rng.normal(1540, 180, n); torque = rng.normal(40, 10, n); wear = rng.uniform(0, 250, n)
    torque[target == 1] += 15                      # make failures learnable
    wear[ft == 'Tool Wear Failure'] = rng.uniform(200, 240, (ft == 'Tool Wear Failure').sum())
    df = pd.DataFrame({'UDI': np.arange(1, n + 1), 'Product ID': [f'X{i:05d}' for i in range(n)],
                       'Type': rng.choice(['L', 'M', 'H'], n, p=[.6, .3, .1]),
                       'Air temperature [K]': air.round(1), 'Process temperature [K]': proc.round(1),
                       'Rotational speed [rpm]': speed.round().astype(int), 'Torque [Nm]': torque.round(1),
                       'Tool wear [min]': wear.round().astype(int), 'Target': target, 'Failure Type': ft})
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope='session')
def fake_csv(tmp_path_factory):
    return make_fake_ai4i(str(tmp_path_factory.mktemp('data') / 'predictive_maintenance.csv'))
