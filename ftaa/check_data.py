"""Check that a downloaded AI4I 2020 file is the Kaggle single-label version used in the study.

python -m ftaa.check_data data/predictive_maintenance.csv

It checks the number of rows, the columns, and the full FailureType x Target label audit reported in the paper
(339 failures: 112 heat dissipation, 95 power, 78 overstrain, 45 tool wear, 9 without a mechanism; 18 random
failures marked as normal operation), and prints the file's SHA-256 so it can be recorded.
"""
import hashlib
import sys

import pandas as pd

from . import config as C

REQUIRED = ['UDI', 'Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Target', 'Failure Type']


def main(path):
    sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    df = pd.read_csv(path)
    problems = []
    if len(df) != C.EXPECTED_ROWS:
        problems.append(f'{len(df)} rows, expected {C.EXPECTED_ROWS}')
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        problems.append(f'missing columns: {missing}')
    else:
        audit = pd.crosstab(df['Failure Type'], df['Target'])
        for (ft, t), n in C.EXPECTED_AUDIT.items():
            got = int(audit.loc[ft, t]) if ft in audit.index and t in audit.columns else 0
            if got != n:
                problems.append(f'{ft} with Target={t}: {got}, expected {n}')
        if df[REQUIRED].isna().any().any():
            problems.append('missing values present')
    print(f'file:    {path}\nsha256:  {sha}\nrows:    {len(df)}')
    if problems:
        print('NOT the expected file:')
        for p in problems:
            print('  -', p)
        return 1
    print('OK: matches the dataset used in the study (row count, columns, and label audit).')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
