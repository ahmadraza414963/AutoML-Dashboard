# -*- coding: utf-8 -*-
"""
CPU/GPU acceleration for AutoML training.

CPU: the fork already maps AutoML's `n_jobs` into per-library thread settings
(LightGBM `num_threads`, XGBoost `n_jobs`, CatBoost `thread_count`) and the
sklearn extras receive `n_jobs=-1` in `_build`, so core utilisation follows
the `automl.n_jobs` config value (-1 = all cores).

GPU: the fork's algorithm classes build whitelisted learner params and never
pass device settings through.  We patch the three GBDT algorithm `__init__`s
so that when `automl.use_gpu` is enabled (default) and the installed library
is actually GPU-capable, training runs on the GPU.  Capability is probed once
per process with a tiny fit; libraries without GPU support silently fall back
to CPU.
"""
import numpy as np
from dashboard import _vendored  # noqa: F401

from supervised.algorithms.catboost import CatBoostAlgorithm
from supervised.algorithms.lightgbm import LightgbmAlgorithm
from supervised.algorithms.xgboost import XgbAlgorithm

_USE_GPU = True
_PROBE_CACHE = {}
_PATCHED = False


def _probe_impl(lib):
    X = np.random.RandomState(0).rand(64, 5)
    y = np.random.RandomState(1).rand(64)
    if lib == "lightgbm":
        import lightgbm as lgb

        lgb.LGBMRegressor(n_estimators=1, max_depth=1, device="gpu").fit(X, y)
    elif lib == "xgboost":
        import xgboost as xgb

        xgb.XGBRegressor(
            n_estimators=1, max_depth=1, tree_method="hist", device="cuda"
        ).fit(X, y)
    elif lib == "catboost":
        from catboost import CatBoostRegressor

        CatBoostRegressor(iterations=1, depth=2, task_type="GPU", verbose=False).fit(
            X, y
        )
    return True


def _probe(lib):
    if lib not in _PROBE_CACHE:
        try:
            _PROBE_CACHE[lib] = _probe_impl(lib)
        except Exception:
            _PROBE_CACHE[lib] = False
    return _PROBE_CACHE[lib]


def gpu_available(lib):
    return _USE_GPU and _probe(lib)


def gpu_status():
    return {lib: gpu_available(lib) for lib in ("lightgbm", "xgboost", "catboost")}


def _patch_lgb():
    orig = LightgbmAlgorithm.__init__

    def _init(self, params):
        orig(self, params)
        if gpu_available("lightgbm") and isinstance(self.learner_params, dict):
            self.learner_params["device"] = "gpu"

    LightgbmAlgorithm.__init__ = _init


def _patch_xgb():
    orig = XgbAlgorithm.__init__

    def _init(self, params):
        orig(self, params)
        learner = getattr(self, "learner_params", None)
        if gpu_available("xgboost") and isinstance(learner, dict):
            learner["device"] = "cuda"

    XgbAlgorithm.__init__ = _init


def _patch_xgb_iteration_range():
    """xgb 2.x Booster has no `best_ntree_limit` attribute, so the fork's
    predict() falls back to `boosting_rounds` (10000) even when early
    stopping stopped training much earlier -> "Out of range for tree
    layers" crash on small datasets (CPU and GPU alike).  Clamp the
    iteration range to the rounds actually trained."""
    orig_predict = XgbAlgorithm.predict

    def _predict(self, X):
        self.reload()
        if self.model is not None:
            try:
                limit = int(getattr(self, "best_ntree_limit", 0) or self.boosting_rounds)
                self.best_ntree_limit = min(limit, self.model.num_boosted_rounds())
            except Exception:
                pass
        return orig_predict(self, X)

    XgbAlgorithm.predict = _predict


def _patch_cat():
    orig_init = CatBoostAlgorithm.__init__

    def _init(self, params):
        orig_init(self, params)
        model = getattr(self, "model", None)
        if gpu_available("catboost") and model is not None:
            # column sampling (rsm < 1) is unsupported on GPU -> crash
            try:
                model.set_params(task_type="GPU", rsm=1.0)
                self.params["rsm"] = 1.0
            except Exception:
                pass

    CatBoostAlgorithm.__init__ = _init

    orig_fit = CatBoostAlgorithm.fit

    def _fit(self, X, y, sample_weight=None, X_validation=None, y_validation=None,
             sample_weight_validation=None, log_to_file=None, max_time=None):
        # GPU does not support training continuation (init_model), which the
        # fork's warm-up path uses; skip the warm-up on GPU and train with
        # early stopping from scratch instead.
        if gpu_available("catboost") and self.params.get("num_boost_round") is None:
            self.params["num_boost_round"] = self.rounds
        return orig_fit(
            self, X, y, sample_weight=sample_weight,
            X_validation=X_validation, y_validation=y_validation,
            sample_weight_validation=sample_weight_validation,
            log_to_file=log_to_file, max_time=max_time,
        )

    CatBoostAlgorithm.fit = _fit


def apply_patches():
    global _PATCHED
    if _PATCHED:
        return
    _patch_lgb()
    _patch_xgb()
    _patch_xgb_iteration_range()
    _patch_cat()
    _PATCHED = True


def apply_acceleration(use_gpu):
    global _USE_GPU
    _USE_GPU = bool(use_gpu)
    apply_patches()
