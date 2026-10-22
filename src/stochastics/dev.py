"""Developer helpers for the workbench notebook.

    from stochastics import dev
    dev.health()            # does every subpackage import? where is it loaded from?
    dev.test()              # run the whole test suite (in a fresh process)
    dev.test('baseline')    # just tests/test_baseline.py
    dev.test('finance')     # every tests/test_finance*.py (insurance, markets, order book...)
    dev.test('popdyn', '-k', 'wealth')   # extra pytest arguments pass through

Tests run in a subprocess so they always see the code currently on disk and
never change the notebook's plotting backend.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

__all__ = ['root', 'health', 'test', 'SUBPACKAGES']

SUBPACKAGES = ('physics', 'baseline', 'popdyn', 'finance')


def root() -> Path:
    """Project folder (the one containing ``stochastics/`` and ``tests/``)."""
    return Path(__file__).resolve().parents[1]


def health() -> pd.DataFrame:
    """Import every subpackage and report status, size of its public API and location."""
    rows = []
    for name in SUBPACKAGES:
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(f'stochastics.{name}')
            public = getattr(mod, '__all__', [n for n in dir(mod) if not n.startswith('_')])
            modules = sorted(p.stem for p in Path(mod.__file__).parent.glob('*.py')
                             if p.stem != '__init__')
            broken = []
            for sub in modules:                         # packages may load submodules lazily: import them all
                if sub.startswith('_'):
                    continue
                try:
                    importlib.import_module(f'stochastics.{name}.{sub}')
                except Exception as exc:  # noqa: BLE001
                    broken.append(f'{sub}: {type(exc).__name__}: {exc}')
            rows.append({'subpackage': name, 'status': 'FAILED' if broken else 'ok', 'public names': len(public),
                         'modules': ', '.join(modules), 'import (s)': round(time.perf_counter() - t0, 2),
                         'error': '; '.join(broken)})
        except Exception as exc:  # noqa: BLE001 - report anything that breaks an import
            rows.append({'subpackage': name, 'status': 'FAILED', 'public names': 0, 'modules': '',
                         'import (s)': round(time.perf_counter() - t0, 2),
                         'error': f'{type(exc).__name__}: {exc}'})
    return pd.DataFrame(rows).set_index('subpackage')


def test(part: str | None = None, *pytest_args: str, quiet: bool = True) -> int:
    """Run pytest for one part (``'physics'``, ``'baseline'``, ``'finance'``...) or everything.

    Prints pytest's report and returns its exit code (0 = all passed).
    """
    tests = root() / 'tests'
    targets = sorted(tests.glob(f'test_{part}*.py')) if part else [tests]
    if not targets:
        raise FileNotFoundError(f'no tests matching tests/test_{part}*.py')
    cmd = [sys.executable, '-m', 'pytest', *map(str, targets), '-p', 'no:cacheprovider',
           '-q' if quiet else '-v', *pytest_args]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=root(), capture_output=True, text=True)
    out = proc.stdout.strip() or proc.stderr.strip()
    print(out[-6000:])
    print(f'\n[{part or "all"}] exit code {proc.returncode} in {time.perf_counter() - t0:.1f}s')
    return proc.returncode
