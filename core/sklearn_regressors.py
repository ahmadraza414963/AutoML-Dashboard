# -*- coding: utf-8 -*-
"""
Registers a comprehensive set of scikit-learn regressors into the `mlsuper`
(mljar-supervised) AlgorithmsRegistry so they can be used inside AutoML runs
with exactly the same preprocessing / validation / reporting pipeline.

Also patches `MljarTuner.generate_params` so that:
  * selected custom regressors are actually generated for the
    "default_algorithms" and "not_so_random" steps (the built-in tuner only
    generates a hard-coded list of library algorithms),
  * user supplied hyper-parameter overrides are injected into every generated
    model's `params["learner"]` dict.
"""
import copy
import inspect
import logging
import sys
import warnings

from dashboard import _vendored  # noqa: F401  (vendored fork must be importable first)

import numpy as np

import supervised
from supervised.algorithms.sklearn import SklearnAlgorithm
from supervised.algorithms.registry import (
    AlgorithmsRegistry,
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.tuner.mljar_tuner import MljarTuner
from supervised.tuner.random_parameters import RandomParameters
from supervised.utils.metric import Metric
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# --------------------------------------------------------------------------
# estimator factory map
# --------------------------------------------------------------------------
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    ARDRegression,
    HuberRegressor,
    SGDRegressor,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    TheilSenRegressor,
    OrthogonalMatchingPursuit,
    TweedieRegressor,
)
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor


def _param_names(func):
    try:
        sig = inspect.signature(func)
        return set(sig.parameters.keys())
    except Exception:
        return set()


def _build(estimator_cls, params):
    """Instantiate `estimator_cls` honouring only parameters it actually
    accepts, and inject random_state / n_jobs from the AutoML framework."""
    supported = _param_names(estimator_cls.__init__)
    kwargs = {}
    seed = params.get("seed")
    n_jobs = params.get("n_jobs", -1)
    if "random_state" in supported:
        kwargs["random_state"] = params.get("random_state", seed)
    if "n_jobs" in supported and params.get("n_jobs") is not None:
        kwargs["n_jobs"] = n_jobs
    for k, v in params.items():
        if k in ("seed", "model_type", "ml_task", "n_jobs", "eval_metric",
                 "eval_metric_name", "custom_eval_metric_name", "explain_level",
                 "num_class", "model_architecture_json", "objective", "metric"):
            continue
        if k in supported:
            kwargs[k] = v
    return estimator_cls(**kwargs)


def _reg(name, default_params, grid, estimator_cls, requires=None):
    """Register one sklearn regressor."""
    display = name
    cls = type(
        str(name).replace(" ", "") + "SklearnReg",
        (SklearnRegressorAlgorithm,),
        {
            "algorithm_name": display,
            "algorithm_short_name": display,
            "ESTIMATOR": estimator_cls,
        },
    )
    # expose in module namespace so pickle can resolve the class on load
    setattr(sys.modules[__name__], cls.__name__, cls)
    SKLEARN_REGRESSORS[display] = {
        "estimator": estimator_cls,
        "defaults": default_params,
        "grid": grid,
        "class": cls,
        "requires": requires or [],
    }
    return display


class SklearnRegressorAlgorithm(SklearnAlgorithm):
    """Single-pass sklearn regressor wrapped in the mlsuper algorithm API."""

    algorithm_name = "Sklearn Regressor"
    algorithm_short_name = "Sklearn Regressor"
    ESTIMATOR = LinearRegression

    def __init__(self, params):
        super(SklearnRegressorAlgorithm, self).__init__(params)
        self.max_iters = 1
        self.log_metric = Metric(
            {"name": self.params.get("eval_metric_name", "rmse")}
        )
        self.model = _build(self.ESTIMATOR, params)

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_validation=None,
        y_validation=None,
        sample_weight_validation=None,
        log_to_file=None,
        max_time=None,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore")
            if sample_weight is not None:
                try:
                    self.model.fit(X, np.ravel(y), sample_weight=sample_weight)
                    self._fitted_with_weights = True
                    return
                except TypeError:
                    pass
            self.model.fit(X, np.ravel(y))

    def predict(self, X):
        self.reload()
        if hasattr(X, "columns") and not hasattr(self.model, "feature_names_in_"):
            X = X.values
        return self.model.predict(X)

    def get_metric_name(self):
        return self.params.get("eval_metric_name", "rmse")

    def file_extension(self):
        return "sklearn_reg"


SKLEARN_REGRESSORS = {}
# canonical display-name -> entry

# --------------------------------------------------------------------------
# The full scikit-learn regressor catalogue
# --------------------------------------------------------------------------
COMMON = {
    "Ridge Regression": _reg(
        "Ridge Regression", {"alpha": 1.0},
        {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
        Ridge,
    ),
    "Lasso Regression": _reg(
        "Lasso Regression", {"alpha": 1.0},
        {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
        Lasso,
    ),
    "Elastic Net": _reg(
        "Elastic Net", {"alpha": 1.0, "l1_ratio": 0.5},
        {"alpha": [0.0001, 0.01, 0.1, 1.0, 10.0],
         "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        ElasticNet,
    ),
    "Bayesian Ridge": _reg(
        "Bayesian Ridge", {"alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-6, "lambda_2": 1e-6},
        {"alpha_1": [1e-6, 1e-4, 1e-2], "lambda_1": [1e-6, 1e-4, 1e-2]},
        BayesianRidge,
    ),
    "ARD Regression": _reg(
        "ARD Regression", {"alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-6, "lambda_2": 1e-6},
        {"alpha_1": [1e-6, 1e-4, 1e-2], "lambda_1": [1e-6, 1e-4, 1e-2]},
        ARDRegression,
    ),
    "Huber Regressor": _reg(
        "Huber Regressor", {"epsilon": 1.35, "alpha": 0.0001},
        {"epsilon": [1.1, 1.35, 1.7, 2.0], "alpha": [0.0001, 0.001, 0.01]},
        HuberRegressor,
    ),
    "SGD Regressor": _reg(
        "SGD Regressor", {"alpha": 0.0001, "loss": "squared_error"},
        {"alpha": [0.00001, 0.0001, 0.001], "loss": ["squared_error", "huber"]},
        SGDRegressor,
    ),
    "Passive Aggressive": _reg(
        "Passive Aggressive", {"C": 1.0, "epsilon": 0.1},
        {"C": [0.01, 0.1, 1.0, 10.0], "epsilon": [0.01, 0.1, 1.0]},
        PassiveAggressiveRegressor,
    ),
    "RANSAC": _reg(
        "RANSAC", {"min_samples": None, "residual_threshold": None},
        {"min_samples": [0.3, 0.5, 0.7], "residual_threshold": [1.0, 2.0]},
        RANSACRegressor,
    ),
    "Theil-Sen": _reg(
        "Theil-Sen", {"max_subpopulation": 1000},
        {"max_subpopulation": [500, 1000, 2000]},
        TheilSenRegressor,
    ),
    "Orthogonal Matching Pursuit": _reg(
        "Orthogonal Matching Pursuit", {"n_nonzero_coefs": None},
        {"n_nonzero_coefs": [5, 10, 20]},
        OrthogonalMatchingPursuit,
    ),
    "Tweedie Regressor": _reg(
        "Tweedie Regressor", {"power": 0.0, "alpha": 1.0},
        {"power": [0.0, 1.0, 2.0], "alpha": [0.01, 0.1, 1.0, 10.0]},
        TweedieRegressor,
    ),
    "SVR": _reg(
        "SVR", {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"},
        {"C": [0.01, 0.1, 1.0, 10.0, 100.0],
         "epsilon": [0.01, 0.1, 1.0],
         "gamma": ["scale", "auto"]},
        SVR,
    ),
    "Linear SVR": _reg(
        "Linear SVR", {"epsilon": 0.1},
        {"C": [0.1, 1.0, 10.0], "epsilon": [0.01, 0.1, 1.0]},
        LinearSVR,
    ),
    "NuSVR": _reg(
        "NuSVR", {"nu": 0.5, "C": 1.0, "gamma": "scale"},
        {"nu": [0.25, 0.5, 0.75], "C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
        NuSVR,
    ),
    "Kernel Ridge": _reg(
        "Kernel Ridge", {"alpha": 1.0, "kernel": "rbf"},
        {"alpha": [0.01, 0.1, 1.0, 10.0], "kernel": ["linear", "rbf", "polynomial"]},
        KernelRidge,
    ),
    "Gaussian Process": _reg(
        "Gaussian Process", {"normalize_y": True},
        {"normalize_y": [True, False]},
        GaussianProcessRegressor,
    ),
    "KNN Regressor": _reg(
        "KNN Regressor", {"n_neighbors": 5, "weights": "distance"},
        {"n_neighbors": [3, 5, 10, 20, 50], "weights": ["uniform", "distance"]},
        KNeighborsRegressor,
    ),
    "Decision Tree (sklearn)": _reg(
        "Decision Tree (sklearn)", {"max_depth": 6, "min_samples_split": 4},
        {"max_depth": [3, 4, 5, 6, 8, 10], "min_samples_split": [2, 4, 10, 30],
         "criterion": ["squared_error", "friedman_mse"]},
        DecisionTreeRegressor,
    ),
    "Random Forest (sklearn)": _reg(
        "Random Forest (sklearn)", {"n_estimators": 100, "max_depth": 12, "max_features": 0.6},
        {"max_depth": [4, 6, 8, 12, 18], "max_features": [0.5, 0.6, 0.8, 1.0],
         "min_samples_split": [10, 30, 50]},
        RandomForestRegressor,
    ),
    "Extra Trees (sklearn)": _reg(
        "Extra Trees (sklearn)", {"n_estimators": 100, "max_depth": 12, "max_features": 0.6},
        {"max_depth": [4, 6, 8, 12, 18], "max_features": [0.5, 0.6, 0.8, 1.0],
         "min_samples_split": [10, 30, 50]},
        ExtraTreesRegressor,
    ),
    "Gradient Boosting": _reg(
        "Gradient Boosting",
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        {"learning_rate": [0.01, 0.05, 0.1, 0.2],
         "max_depth": [2, 3, 4, 5, 6],
         "subsample": [0.8, 0.9, 1.0]},
        GradientBoostingRegressor,
    ),
    "Hist Gradient Boosting": _reg(
        "Hist Gradient Boosting",
        {"max_iter": 100, "learning_rate": 0.1, "max_depth": None},
        {"learning_rate": [0.01, 0.05, 0.1, 0.2],
         "max_iter": [50, 100, 200, 300],
         "min_samples_leaf": [10, 30, 100]},
        HistGradientBoostingRegressor,
    ),
    "AdaBoost": _reg(
        "AdaBoost", {"n_estimators": 100, "learning_rate": 1.0},
        {"n_estimators": [25, 50, 100, 200], "learning_rate": [0.1, 0.5, 1.0, 2.0]},
        AdaBoostRegressor,
    ),
    "Bagging Regressor": _reg(
        "Bagging Regressor", {"n_estimators": 25, "max_features": 1.0, "max_samples": 1.0},
        {"n_estimators": [10, 25, 50, 100], "max_samples": [0.5, 0.8, 1.0],
         "max_features": [0.5, 0.8, 1.0]},
        BaggingRegressor,
    ),
    "PLS Regression": _reg(
        "PLS Regression", {"n_components": 2, "scale": True},
        {"n_components": [1, 2, 3, 5], "scale": [True, False]},
        PLSRegression,
    ),
    "MLP Regressor (sklearn)": _reg(
        "MLP Regressor (sklearn)",
        {"hidden_layer_sizes": (50, 25), "activation": "relu", "alpha": 0.001, "max_iter": 500},
        {"hidden_layer_sizes": [(25,), (50,), (50, 25), (100, 50)],
         "activation": ["relu", "tanh"], "alpha": [0.0001, 0.001, 0.01]},
        MLPRegressor,
    ),
}

REQUIRED_PREPROCESSING = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "target_scale",
]
ADDITIONAL = {
    "max_rows_limit": None,
    "max_cols_limit": None,
}

_ALREADY_REGISTERED = False


def _safe_random_get(params, seed=1):
    """Defensive replacement for the fork's RandomParameters.get.

    The original does ``np.random.permutation(params[k])[0].item()``, which
    crashes when a sampled option is None (AttributeError) or when the option
    list is ragged — e.g. MLP's hidden_layer_sizes with tuples of different
    lengths (ValueError: inhomogeneous shape). Sampling by index keeps the
    original Python objects (tuples stay tuples) and works for any list."""
    rng = np.random.RandomState(seed)
    out = {"seed": seed}
    for k, v in (params or {}).items():
        if isinstance(v, (list, tuple)):
            opts = [o for o in v if o is not None]
        else:
            opts = [v] if v is not None else []
        if not opts:
            continue
        out[k] = opts[int(rng.randint(len(opts)))]
    return out


def _patch_random_parameters():
    RandomParameters.get = staticmethod(_safe_random_get)


def register_sklearn_regressors():
    """Idempotent registration of the sklearn regressors + tuner patches."""
    global _ALREADY_REGISTERED
    if _ALREADY_REGISTERED:
        return
    reg = AlgorithmsRegistry.registry[REGRESSION]
    for display, info in SKLEARN_REGRESSORS.items():
        if display in reg:
            continue
        AlgorithmsRegistry.add(
            REGRESSION,
            info["class"],
            info["grid"],
            REQUIRED_PREPROCESSING,
            ADDITIONAL,
            copy.deepcopy(info["defaults"]),
        )
    _patch_tuner_generate_params()
    _patch_random_parameters()
    _ALREADY_REGISTERED = True


# --------------------------------------------------------------------------
# per-run state injected by the chain runner (single-process assumption)
# --------------------------------------------------------------------------
_ACTIVE_EXTRAS = {}          # algorithm display name -> info entry
_ACTIVE_OVERRIDES = {}       # algorithm display name -> {param: value}
_ACTIVE_EVAL_METRIC = "rmse"


def set_active_config(extras, overrides, eval_metric):
    global _ACTIVE_EXTRAS, _ACTIVE_OVERRIDES, _ACTIVE_EVAL_METRIC
    _ACTIVE_EXTRAS = dict(extras or {})
    _ACTIVE_OVERRIDES = dict(overrides or {})
    _ACTIVE_EVAL_METRIC = eval_metric or "rmse"


def _apply_overrides_to_params(params):
    if not _ACTIVE_OVERRIDES or not isinstance(params, dict):
        return params
    model_type = (params.get("learner") or {}).get("model_type")
    if model_type is None:
        return params
    ov = _ACTIVE_OVERRIDES.get(model_type)
    if ov:
        learner = dict(params.get("learner") or {})
        for k, v in ov.items():
            if k in ("seed", "model_type", "ml_task"):
                continue
            learner[k] = copy.deepcopy(v)
        params["learner"] = learner
    return params


def _extra_default_params(tuner, models_cnt):
    """Build params for selected custom regressors (default params variant)."""
    generated = []
    for model_type, info in _ACTIVE_EXTRAS.items():
        params = tuner._get_model_params(model_type, seed=models_cnt + 1, params_type="default")
        if params is None:
            continue
        params["name"] = tuner.get_model_name(model_type, models_cnt + 1)
        params["status"] = "initialized"
        params["final_loss"] = None
        params["train_time"] = None
        params["data_type"] = "original"
        learner = params.get("learner", {})
        learner["eval_metric_name"] = _ACTIVE_EVAL_METRIC
        learner["ml_task"] = tuner._ml_task
        params["learner"] = learner
        generated.append(params)
        models_cnt += 1
    return generated


def _extra_random_params(tuner, models_cnt):
    """Build params for selected custom regressors (random-search variant)."""
    generated = []
    for model_type, info in _ACTIVE_EXTRAS.items():
        params = tuner._get_model_params(model_type, seed=models_cnt + 1)
        if params is None:
            continue
        params["name"] = tuner.get_model_name(model_type, models_cnt + 1)
        params["status"] = "initialized"
        params["final_loss"] = None
        params["train_time"] = None
        params["data_type"] = "original"
        learner = params.get("learner", {})
        learner["eval_metric_name"] = _ACTIVE_EVAL_METRIC
        learner["ml_task"] = tuner._ml_task
        params["learner"] = learner
        generated.append(params)
        models_cnt += 1
    return generated


_ORIG_GENERATE_PARAMS = None


def _patched_generate_params(self, step, models, results_path, stacked_models, total_time_limit):
    params = _ORIG_GENERATE_PARAMS(self, step, models, results_path, stacked_models, total_time_limit)
    params = list(params) if params else []
    models_cnt = len(models) + len(params)
    if step in ("default_algorithms", "not_so_random") and _ACTIVE_EXTRAS:
        if step == "default_algorithms":
            params = params + _extra_default_params(self, models_cnt)
        else:
            params = params + _extra_random_params(self, models_cnt)
    out = [_apply_overrides_to_params(p) if isinstance(p, dict) else p for p in params]
    return out if out else None


def _patch_tuner_generate_params():
    global _ORIG_GENERATE_PARAMS
    if _ORIG_GENERATE_PARAMS is None:
        _ORIG_GENERATE_PARAMS = MljarTuner.generate_params
        MljarTuner.generate_params = _patched_generate_params