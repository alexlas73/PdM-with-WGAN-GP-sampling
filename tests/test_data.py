"""Data preparation, label audit, and the leakage-safe repeated splits."""
import numpy as np

from ftaa import check_data, config as C, data as D


def test_check_data_accepts_expected_schema_and_counts(fake_csv):
    assert check_data.main(fake_csv) == 0


def test_load_features(fake_csv):
    df, X, y = D.load(fake_csv)
    assert len(df) == C.EXPECTED_ROWS and int(y.sum()) == 339
    for c in C.NUMERICAL_FEATURES + C.CATEGORICAL_FEATURES + [C.FAILURE_COL]:
        assert c in X.columns
    assert C.TARGET not in X.columns
    assert np.allclose(df['Temperature_difference_C'], df['Process_temperature_C'] - df['Air_temperature_C'], atol=1e-4)


def test_label_audit_matches_paper(fake_csv):
    df, _, _ = D.load(fake_csv)
    audit = D.label_audit(df)
    for (ft, t), n in C.EXPECTED_AUDIT.items():
        assert audit.loc[ft, t] == n


def test_splits_are_disjoint_complete_and_stratified(fake_csv):
    df, _, y = D.load(fake_csv)
    strat = D.strata(df)
    for s in (1, 2, 30):
        tr, va, te = D.split_indices(len(df), strat, s)
        assert (len(tr), len(va), len(te)) == (7000, 1500, 1500)
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        for part, n in ((tr, 7000), (te, 1500)):
            assert abs(y.values[part].sum() - 339 * n / 10000) <= 2      # 3.4 % prevalence preserved
        ft = df[C.FAILURE_COL].values
        assert abs((ft[te] == 'Tool Wear Failure').sum() - 45 * 0.15) <= 1


def test_splits_are_reproducible(fake_csv):
    df, _, _ = D.load(fake_csv)
    strat = D.strata(df)
    a, b = D.split_indices(len(df), strat, 7), D.split_indices(len(df), strat, 7)
    assert all((x == z).all() for x, z in zip(a, b))


def test_preprocessor_never_uses_failure_type(fake_csv):
    df, X, _ = D.load(fake_csv)
    Z = D.make_prep().fit(X).transform(X)
    assert Z.shape[1] == len(C.NUMERICAL_FEATURES) + 1          # FailureType is dropped
