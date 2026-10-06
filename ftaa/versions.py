"""Record the Python, library and GPU versions of the current environment (python -m ftaa.versions [out.json])."""
import importlib
import json
import platform
import sys

MODULES = ['sklearn', 'imblearn', 'tabgan', 'torch', 'optuna', 'lightgbm', 'pandas', 'numpy', 'scipy', 'matplotlib']


def collect():
    v = {'python': sys.version.split()[0], 'platform': platform.platform()}
    for m in MODULES:
        try:
            v[m] = importlib.import_module(m).__version__
        except Exception as e:
            v[m] = f'unavailable ({e.__class__.__name__})'
    try:
        import torch
        v['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
    except Exception:
        pass
    return v


def write(path):
    v = collect()
    json.dump(v, open(path, 'w'), indent=1)
    return v


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else None
    v = write(out) if out else collect()
    print(json.dumps(v, indent=1))
