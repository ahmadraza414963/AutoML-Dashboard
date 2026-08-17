# -*- coding: utf-8 -*-
"""
Chain runner: trains one AutoML per selected target.

  * single target  -> plain AutoML fit
  * multiple targets -> a multi-output regression chain: for each target in the
    user-selected order, the out-of-fold predictions of the previous targets are
    appended as additional input features before training the next target
    (leakage-free chaining, like sklearn's RegressorChain but with OOF values).

Everything is saved to the session directory (results_path per target) so a
session can be interrupted and resumed from disk.
"""
import os
from dashboard import _vendored  # noqa: F401  (vendored fork must be importable first)
import time

import numpy as np
import pandas as pd

from supervised import AutoML

from . import sklearn_regressors
from .accelerate import apply_acceleration, gpu_status
from .session import sanitize_name


def get_automl_kwargs(config):
    a = config.get("automl", {})
    val = {
        "validation_type": a.get("validation_type", "kfold"),
        "shuffle": True,
    }
    if val["validation_type"] == "kfold":
        val["k_folds"] = int(a.get("k_folds", 5))
    else:
        val["train_ratio"] = float(a.get("train_ratio", 0.75))
    kwargs = dict(
        mode=a.get("mode", "Compete"),
        ml_task=a.get("ml_task", "auto"),
        eval_metric=a.get("eval_metric", "rmse"),
        validation_strategy=val,
        total_time_limit=_opt_int(a.get("total_time_limit")),
        model_time_limit=_opt_int(a.get("model_time_limit")),
        start_random_models=int(a.get("start_random_models", 10)),
        hill_climbing_steps=int(a.get("hill_climbing_steps", 3)),
        top_models_to_improve=int(a.get("top_models_to_improve", 3)),
        train_ensemble=bool(a.get("train_ensemble", True)),
        stack_models=bool(a.get("stack_models", True)),
        explain_level=int(a.get("explain_level", 2)),
        boost_on_errors=bool(a.get("boost_on_errors", True)),
        random_state=int(a.get("random_state", 1234)),
        n_jobs=int(a.get("n_jobs", -1)),
        verbose=1,
    )
    fe = config.get("feature_engineering", {})
    golden_cnt = fe.get("golden_count")
    try:
        golden_cnt = int(golden_cnt) if golden_cnt not in (None, "") else None
    except Exception:
        golden_cnt = None
    kwargs["golden_features"] = bool(fe.get("golden", True)) or (golden_cnt is not None and golden_cnt > 0)
    if golden_cnt is not None and golden_cnt > 0:
        kwargs["golden_features"] = golden_cnt
    kwargs["kmeans_features"] = bool(fe.get("kmeans", True))
    kwargs["mix_encoding"] = bool(fe.get("mix_encoding", True))
    kwargs["features_selection"] = bool(a.get("features_selection", True))
    return kwargs


def _opt_int(v):
    if v in (None, "", 0):
        return None
    try:
        return int(v)
    except Exception:
        return None


def build_algorithm_lists(config):
    """Return (builtin_names, sklearn_extras, overrides dict)."""
    alg = config.get("algorithms", {})
    builtin = [a for a in alg.get("builtin", []) if a]
    extras = {}
    for name in alg.get("sklearn", []):
        if not name:
            continue
        info = sklearn_regressors.SKLEARN_REGRESSORS.get(name)
        if info:
            extras[name] = info
    overrides = {k: dict(v) for k, v in alg.get("params", {}).items() if v}
    return builtin, extras, overrides


def build_working_data(session, df=None, data_path=None, sheet=None):
    """Load + prepare the working dataframe (features + targets), cached."""
    if df is not None:
        session._df = df.copy()
    elif session._df is None:
        if data_path is None:
            data_path = session.config.get("data", {}).get("path")
        sheet = sheet if sheet is not None else session.config.get("data", {}).get("sheet")
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        from .data_io import load_file
        session._df = load_file(data_path, sheet=sheet)
    df = session._df.copy()
    df = df.reset_index(drop=True)
    session._df = df
    session.save_data_snapshot(df)
    return df


def _positional_oof(best_model, X):
    """Row-aligned out-of-fold predictions of the best model (fall back to
    in-sample predictions for rows missing OOF values)."""
    try:
        oof = best_model.get_out_of_folds()
    except Exception:
        oof = None
    out = pd.Series(np.nan, index=X.index)
    if oof is not None and len(oof) > 0:
        cols = [c for c in oof.columns if "prediction" in c]
        if cols:
            ser = oof[cols].mean(axis=1)
            ser.index = ser.index.astype(X.index.dtype)
            out = out.astype(float)
            out.loc[ser.index.intersection(X.index)] = ser.reindex(
                ser.index.intersection(X.index)
            ).values
    mask = out.isna()
    if mask.any():
        pred = best_model.predict(X.loc[mask])
        pred_vals = np.asarray(pred).ravel()
        out.loc[mask] = pred_vals
    return out.astype(float)


def run_chain(session, log, stop_check=None, progress=None):
    """Train all targets in chain order. `log` -> session.log; `stop_check`
    -> callable returning True to stop; `progress` -> callable(target, msg)."""
    cfg = session.config
    columns = cfg.get("columns", {})
    features = list(columns.get("features", []))
    targets = list(columns.get("targets", []))
    if not targets:
        raise ValueError("Select at least one target column.")
    if not features:
        raise ValueError("Select at least one feature column.")

    df = build_working_data(session)

    split_cfg = cfg.get("split", {})
    test_ratio = float(split_cfg.get("test_ratio", 0.15))
    use_test = bool(split_cfg.get("enabled", False))
    split_seed = int(split_cfg.get("seed", 123))

    df_train = df
    df_test = None
    if use_test and len(df) > 5:
        test_idx = np.random.RandomState(split_seed).choice(
            len(df), size=max(1, int(len(df) * test_ratio)), replace=False
        )
        mask = np.zeros(len(df), dtype=bool)
        mask[test_idx] = True
        df_test = df.loc[mask].copy()
        df_train = df.loc[~mask].copy()

    # validate columns
    for col in features + targets:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    for t in targets:
        if t in features:
            raise ValueError(f"Target '{t}' is also selected as a feature.")

    log(f"Data: {len(df_train)} train rows"
                + (f", {len(df_test)} test rows (holdout)" if df_test is not None else "")
                + f" | features: {len(features)} | targets: {len(targets)}")
    log(f"Chain order: {targets}")

    # baseline dataset
    X_train = df_train[features].copy()
    y_train = df_train[targets].copy()

    builtin, extras, overrides = build_algorithm_lists(cfg)
    if not builtin and not extras:
        raise ValueError("Select at least one algorithm.")
    log(f"Built-in algorithms: {builtin}")
    log(f"Sklearn regressors: {list(extras.keys())}")

    automl_kwargs = get_automl_kwargs(cfg)
    automl_kwargs["algorithms"] = builtin + list(extras.keys())
    apply_acceleration(cfg.get("automl", {}).get("use_gpu", True))
    gpu = gpu_status()
    log("Acceleration: GPU enabled=%s | GPU available %s | n_jobs=%s"
        % (bool(cfg.get("automl", {}).get("use_gpu", True)),
           {k: v for k, v in gpu.items()},
           automl_kwargs.get("n_jobs")))
    log(f"AutoML kwargs: {automl_kwargs}")

    chain_preds = session.chain_preds if session.chain_preds else {}
    chain_enabled = bool(cfg.get("chain", {}).get("enabled", False)) and len(targets) > 1

    for i, target in enumerate(targets):
        if stop_check:
            action = stop_check()
            if action:
                if action == "pause":
                    log("Pause requested at target boundary. Session paused — "
                        "resume to continue from disk.")
                    session.mark_paused()
                else:
                    log("Stop requested by user. Session stopped — resume to continue.")
                    session.mark_stopped()
                return False

        rp = session.target_results_path(target)
        os.makedirs(rp, exist_ok=True)
        y_i = y_train[target]
        X_i = X_train.copy()
        used_chain = {}

        if target in session._finished_targets and session.resume_mode:
            log(f"==== Target {i+1}/{len(targets)}: '{target}' — already finished, "
                "reusing results from disk ====")
            automl = session.load_automl_pickle(target)
            if automl is not None:
                session.automl[target] = automl
            if chain_enabled:
                preds = _fetch_chain_preds(session, target, X_train, X_i)
                if preds is not None:
                    chain_preds[target] = np.asarray(preds).ravel()
                    session.chain_preds = chain_preds
            session.mark_finished_target(target)
            if progress:
                progress(target, "finished")
            continue

        if chain_enabled:
            for prev in targets[:i]:
                col = f"chain_{sanitize_name(prev)}"
                if prev in chain_preds and len(chain_preds[prev]) == len(X_i):
                    X_i[col] = np.asarray(chain_preds[prev]).ravel()
                    used_chain[col] = True
                else:
                    # re-derive from pickle / disk if available
                    preds = _fetch_chain_preds(session, prev, X_train, X_i)
                    if preds is not None:
                        X_i[col] = preds
                        used_chain[col] = True

        log(f"==== Target {i+1}/{len(targets)}: '{target}' ====")
        if used_chain:
            log(f"  chained features: {list(used_chain.keys())}")
        if progress:
            progress(target, "starting")

        sklearn_regressors.set_active_config(extras, overrides, cfg["automl"].get("eval_metric", "rmse"))

        automl = AutoML(**automl_kwargs, results_path=rp)
        automl._cha_target = target  # keep ref for reports (not required by lib)
        t0 = time.time()
        automl.fit(X_i, y_i)
        log(f"  '{target}' AutoML fit took {time.time()-t0:.1f}s")

        best = automl._best_model
        if best is not None:
            log(f"  best model: {best.get_name()} | metric {automl._eval_metric}: "
                        f"{best.get_final_loss():.6f} | train time {best.get_train_time():.1f}s")

        session.automl[target] = automl
        session.pickle_automl(target, automl)

        if chain_enabled and i < len(targets) - 1:
            oof = _positional_oof(best, X_i) if best is not None else \
                pd.Series(np.asarray(automl._predict(X_i)).ravel(), index=X_i.index)
            chain_preds[target] = oof.values
            session.chain_preds = chain_preds
            _save_chain_preds(session, targets, chain_preds)

        session.mark_finished_target(target)
        if progress:
            progress(target, "finished")
        log(f"  '{target}' done.")

    # holdout evaluation (chained predictions on test set)
    if df_test is not None and len(df_test):
        log("----- Holdout test evaluation (chained) -----")
        X_test = df_test[features].copy()
        pred_rows = {}
        for target in targets:
            automl = session.automl.get(target)
            if automl is None:
                automl = session.load_automl_pickle(target)
            if automl is None:
                continue
            pred = np.asarray(automl._predict(X_test)).ravel()
            pred_rows[target] = pred
            X_test = X_test.copy()
            X_test[f"chain_{sanitize_name(target)}"] = pred

        if pred_rows:
            import json
            test_pred = df_test[targets].copy()
            for t, p in pred_rows.items():
                test_pred[f"predicted_{t}"] = p
                test_pred[f"chain_feature_pred_{t}"] = p
            test_pred = test_pred.reset_index(drop=True)
            test_path = os.path.join(session.dir, "test_predictions.csv")
            test_pred.to_csv(test_path, index=False)
            session.config.setdefault("results", {})["test_predictions_csv"] = test_path
            session.config["results"]["test_predictions_csv"] = test_path
            metrics = _regression_metrics(df_test, pred_rows)
            session.config.setdefault("results", {})["test_metrics"] = metrics
            log(f"  holdout metrics: {json.dumps(metrics, default=str)}")

    log("All targets done.")
    session.save()
    return True


def _fetch_chain_preds(session, prev, X_train, X_i):
    """Reconstruct chain predictions for a previous target from disk.

    Uses the stored model's own training columns so out-of-fold predictions
    can be re-derived row-aligned. If the target's model was trained on
    chained feature columns (e.g. it consumed chain_target_a) that are not
    present in the raw feature frame, prediction is impossible after a
    reload — degrade gracefully instead of surfacing the fork's error log.
    """
    automl = session.load_automl_pickle(prev)
    if automl is None:
        return None
    best = getattr(automl, "_best_model", None)
    if best is None:
        return None
    try:
        oof = best.get_out_of_folds()
    except Exception:
        oof = None
    cols = [c for c in getattr(automl, "_data_info", {}).get("columns", [])
            if c in X_train.columns]
    try:
        X_ok = X_train[cols] if cols else X_train
    except Exception:
        X_ok = X_train
    try:
        ser = _positional_oof(best, X_ok)
        if len(ser) == len(X_i):
            return ser.values
    except Exception:
        pass
    return None


def _save_chain_preds(session, targets, chain_preds):
    frame = pd.DataFrame({f"chain_{sanitize_name(t)}": chain_preds.get(t, [])
                          for t in targets if t in chain_preds})
    if len(frame):
        session.save_chain_preds(frame)


def _regression_metrics(df_test, pred_rows):
    y_true = df_test[list(pred_rows.keys())]
    out = {}
    for t, p in pred_rows.items():
        y = y_true[t].astype(float)
        pred = np.asarray(p, dtype=float)
        mae = float(np.mean(np.abs(y - pred)))
        mse = float(np.mean((y - pred) ** 2))
        rmse = float(np.sqrt(mse))
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        mape = float(np.mean(np.abs((y - pred) / (np.abs(y) + 1e-12)))) * 100.0
        out[t] = {"RMSE": round(rmse, 6), "MAE": round(mae, 6), "MSE": round(mse, 6),
                  "R2": round(r2, 6), "MAPE%": round(mape, 3), "n": int(len(y))}
    return out