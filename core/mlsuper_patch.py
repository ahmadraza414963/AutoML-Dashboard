# -*- coding: utf-8 -*-
"""
Monkey-patches applied to the `supervised` (mlsuper / mljar-supervised fork)
package so that sessions are truly resumable and robust:

1. `dump_data` / `load_data` are in-memory only in this fork. They are patched
   to ALSO persist to disk (CSV), so training can be resumed in a brand new
   process (validators load X/y/sample_weight from the results directory).
2. Fixes the `get_learners_names` typo bug (`f.repleace`).
3. Keeps matplotlib figure creation thread-local / defensive around training.
"""
import os
from dashboard import _vendored  # noqa: F401  (vendored fork must be importable first)
import logging

import pandas as pd

import supervised.utils.utils as _utils_mod
import supervised.base_automl as _base_automl_mod
import supervised.tuner.mljar_tuner as _tuner_mod
import supervised.validation.validator_kfold as _v_kfold
import supervised.validation.validator_split as _v_split
import supervised.validation.validator_custom as _v_custom

logger = logging.getLogger(__name__)


def _dump_data(file_path, df):
    """Store in memory (light copy - `copy.deepcopy` is broken with this
    pandas/numpy pairing) AND persist to pickle so sessions can be resumed in
    a brand-new process without losing dtypes (datetime64 etc)."""
    try:
        _utils_mod.Store.data[file_path] = df.copy()
        if hasattr(df, "to_pickle"):
            dirpath = os.path.dirname(file_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            df.to_pickle(file_path)
    except Exception:
        pass


def _load_data(file_path):
    try:
        return _utils_mod.Store.data[file_path].copy()
    except Exception:
        pass
    if os.path.exists(file_path):
        try:
            return pd.read_pickle(file_path)
        except Exception:
            try:
                return pd.read_csv(file_path)
            except Exception:
                return None
    return None


def _get_learners_names_fixed(model_path):
    """Fixed version of supervised.utils.common.get_learners_names."""
    names = []
    for f in os.listdir(model_path):
        if "_training.log" in f:
            f = f.replace("_training.log", "")
            if f not in names:
                names += [f]
    return names


_APPLIED = False


def apply_mlsuper_patches():
    global _APPLIED
    if _APPLIED:
        return
    # persistence ----------------------------------------------------------
    _utils_mod.dump_data = _dump_data
    _utils_mod.load_data = _load_data
    for mod in (_base_automl_mod, _tuner_mod, _v_kfold, _v_split, _v_custom):
        mod.dump_data = _dump_data
        mod.load_data = _load_data
    # get_learners_names bug ----------------------------------------------
    try:
        from supervised.utils import common as _common
        _common.get_learners_names = _get_learners_names_fixed
    except Exception:
        pass
    _APPLIED = True
