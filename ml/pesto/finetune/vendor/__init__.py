"""Vendored minimum of pesto-full needed for fine-tuning.

Source: https://github.com/SonyCSLParis/pesto-full (commit at time of vendoring).
Local copy: ../../PESTO_understanding/pesto-full/

Why vendored instead of pip-installed: pesto-full uses `rootutils` + `src` as a
top-level package name and is not designed as a library — installing as pip
package fights its conventions. Since we only need ~5 files and they're stable
(research code), vendoring is cleaner.

Adapted changes vs upstream:
 - merged `src/models/pesto.py` + `src/data/pitch_shift.py` into `pesto_module.py`
 - merged `src/losses/{base,entropy,equivariance}.py` into `losses.py`
 - replaced `from src.X` imports with relative imports
 - removed `remove_omegaconf_dependencies` and `omegaconf` usage (we use argparse)
"""
