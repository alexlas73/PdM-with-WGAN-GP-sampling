"""Compatibility shims for tabgan 2.2.2 (and its bundled _ctgan) with NumPy 2.x and scikit-learn >= 1.2.

1. NumPy 2 removed the aliases np.Inf, np.NaN, ... that tabgan still references -> restored.
2. scikit-learn renamed OneHotEncoder(sparse=...) to sparse_output=... -> the old keyword is mapped.

Neither change alters any computation. apply() is idempotent and must run before tabgan is used.
"""
import numpy as np


def apply():
    for old, new in [('Inf', np.inf), ('NaN', np.nan), ('NAN', np.nan), ('infty', np.inf),
                     ('Infinity', np.inf), ('PINF', np.inf), ('NINF', -np.inf)]:
        if not hasattr(np, old):
            setattr(np, old, new)
    import sklearn.preprocessing as skp
    if not getattr(skp.OneHotEncoder, '_is_compat_shim', False):
        real = skp.OneHotEncoder

        def ohe_compat(*a, **k):
            if 'sparse' in k:
                k['sparse_output'] = k.pop('sparse')
            return real(*a, **k)
        ohe_compat._is_compat_shim = True
        skp.OneHotEncoder = ohe_compat
    try:
        import _ctgan.transformer as ct
        ct.OneHotEncoder = skp.OneHotEncoder
    except Exception:          # _ctgan is imported lazily by tabgan; the patch above is enough then
        pass
