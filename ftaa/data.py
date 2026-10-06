"""Data loading, preparation, label audit, stratification, and the repeated 70/15/15 splits."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from . import config as C


def load(path):
    """Read the Kaggle single-label AI4I 2020 file and apply the study's preparation.

    - drops the identifiers UDI and Product ID
    - converts the two temperatures from kelvin to degrees Celsius
    - adds Temperature_difference_C = process - air temperature (linked to heat dissipation failure)
    - removes brackets and spaces from column names
    - casts the six numerical features to float32
    Returns (df, X_all, y_all). X_all keeps FailureType: the samplers need it, the classifier never sees it
    (make_prep() drops every column it does not name).
    """
    df = pd.read_csv(path).drop(C.DROP_COLUMNS, axis=1)
    df['Air temperature [K]'] -= 273.15
    df['Process temperature [K]'] -= 273.15
    df = df.rename(columns={'Air temperature [K]': 'Air_temperature_C',
                            'Process temperature [K]': 'Process_temperature_C'})
    df['Temperature_difference_C'] = df['Process_temperature_C'] - df['Air_temperature_C']
    df = df.rename(columns={c: c.replace('[', '').replace(']', '').replace(' ', '').replace('(', '').replace(')', '')
                            for c in df.columns})
    df[C.NUMERICAL_FEATURES] = df[C.NUMERICAL_FEATURES].astype(np.float32)
    X_all = df.drop(columns=[C.TARGET])
    y_all = df[C.TARGET].astype(int)
    return df, X_all, y_all


def label_audit(df):
    """FailureType x Target cross-tabulation (the paper's Label Audit)."""
    return pd.crosstab(df[C.FAILURE_COL], df[C.TARGET])


def strata(df):
    """Joint stratification label Target|FailureType; strata with fewer than MIN_STRATUM records become 'rare'."""
    s = (df[C.TARGET].astype(str) + '|' + df[C.FAILURE_COL]).values
    vc = pd.Series(s).value_counts()
    return np.where(pd.Series(s).map(vc).values < C.MIN_STRATUM, 'rare', s)


def split_indices(n, strat, seed):
    """Row indices (train, validation, test) of one repeated split; seed = split index."""
    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=C.TEST_SIZE_FIRST, random_state=seed, stratify=strat)
    va, te = train_test_split(tmp, test_size=C.TEST_SIZE_SECOND, random_state=seed, stratify=strat[tmp])
    return tr, va, te


def make_prep():
    """Standardize the numerical features and ordinally encode Type (L < M < H); drop everything else.

    Always fitted on training data only (for the generative strategies: on the augmented training partition).
    """
    return ColumnTransformer([('num', StandardScaler(), C.NUMERICAL_FEATURES),
                              ('cat', OrdinalEncoder(categories=[C.TYPE_ORDER]), C.CATEGORICAL_FEATURES)],
                             remainder='drop')


CAT_IDX = [len(C.NUMERICAL_FEATURES)]    # column index of Type after make_prep(), for SMOTENC
