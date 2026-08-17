# -*- coding: utf-8 -*-
"""
Feature-engineering pipeline preview: runs the EXACT feature engineering used by
mlsuper (mljar-supervised) standalone on the selected features + target, so the
user can inspect what feature engineering will add/remove before training.

The pipeline (mirroring the library):

  1. DataInfo.compute  -> per-column flags (missing, categorical, datetime,
     text, scale...)
  2. PreprocessingTuner.get -> per-column transforms + target transforms
  3. golden_features / kmeans_features injected from the UI config
  4. Preprocessing(params).fit_and_transform -> final engineered X
"""
import io
import os
import sys
from dashboard import _vendored  # noqa: F401  (vendored fork must be importable first)
import contextlib

import numpy as np
import pandas as pd

from supervised.preprocessing.preprocessing import Preprocessing
from supervised.tuner.data_info import DataInfo
from supervised.tuner.preprocessing_tuner import PreprocessingTuner
from supervised.algorithms.registry import REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION

REQUIRED_PREPROCESSING = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "target_scale",
]


def detect_task(y):
    n = y.nunique(dropna=True)
    if n == 2:
        return BINARY_CLASSIFICATION
    if n <= 20:
        return MULTICLASS_CLASSIFICATION
    return REGRESSION


def build_preprocessing_params(features_df, y, fe_cfg, results_path=None):
    fe_cfg = fe_cfg or {}
    ml_task = detect_task(y)
    data_info = DataInfo.compute(features_df, y, ml_task)
    strategy = (PreprocessingTuner.CATEGORICALS_MIX
                if fe_cfg.get("mix_encoding", True)
                else PreprocessingTuner.CATEGORICALS_ALL_INT)
    params = PreprocessingTuner.get(
        REQUIRED_PREPROCESSING, data_info, ml_task, categorical_strategy=strategy
    )
    golden = fe_cfg.get("golden", False)
    golden_count = fe_cfg.get("golden_count")
    if golden or (isinstance(golden_count, int) and golden_count > 0):
        g = {"results_path": results_path or ".", "ml_task": ml_task, "n_jobs": -1}
        if isinstance(golden_count, int) and golden_count > 0:
            g["features_count"] = golden_count
        params["golden_features"] = g
    if fe_cfg.get("kmeans", False):
        params["kmeans_features"] = {"results_path": results_path or "."}
    return params, data_info, ml_task


def run_fe_preview(df, features, target, fe_cfg, results_path=None):
    """Return a dict describing the pipeline result + the engineered frame."""
    if results_path is not None:
        os.makedirs(results_path, exist_ok=True)
    X = df[features].copy()
    y = df[target].copy()

    params, data_info, ml_task = build_preprocessing_params(X, y, fe_cfg, results_path)
    before_cols = list(X.columns)

    buf = io.StringIO()
    preproces = Preprocessing(params, model_name="fe_preview", k_fold=0, repeat=0)
    with contextlib.redirect_stdout(buf):
        with contextlib.redirect_stderr(buf):
            X_fe, y_fe, _ = preproces.fit_and_transform(X.copy(deep=True), y)

    after_cols = list(X_fe.columns)
    added = [c for c in after_cols if c not in before_cols]
    removed = [c for c in before_cols if c not in after_cols]

    golden_feats = []
    if getattr(preproces, "_golden_features", None) is not None:
        try:
            golden_feats = list(preproces._golden_features._new_columns)
        except Exception:
            golden_feats = []

    kmeans_meta = None
    if getattr(preproces, "_kmeans", None) is not None:
        try:
            kmeans_meta = {
                "new_features": preproces._kmeans._new_features,
                "n_clusters": len([f for f in preproces._kmeans._new_features
                                   if f.startswith("Dist_Cluster")]),
            }
        except Exception:
            kmeans_meta = None

    # column -> transforms applied (only original columns)
    column_transforms = {k: list(v) for k, v in params.get("columns_preprocessing", {}).items()}

    return {
        "ml_task": ml_task,
        "target_preprocessing": params.get("target_preprocessing", []),
        "column_transforms": column_transforms,
        "before_cols": before_cols,
        "after_cols": after_cols,
        "added_columns": added,
        "removed_columns": removed,
        "golden_features": golden_feats,
        "kmeans": kmeans_meta,
        "log": buf.getvalue()[-4000:],
        "X": X_fe,
        "y": y_fe,
        "n_features_before": len(before_cols),
        "n_features_after": len(after_cols),
    }


def summarize_features_df(df):
    """Compact numeric/one-hot summary used for correlation plots."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        return df.apply(lambda c: pd.to_numeric(c, errors="coerce")).iloc[:, :40]
    return numeric.iloc[:, :40]