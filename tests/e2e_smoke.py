# -*- coding: utf-8 -*-
"""
End-to-end smoke test for the AutoML dashboard backend:
  - patches + registry (sklearn regressors)
  - feature-engineering preview pipeline
  - full chained multi-target training run (3 targets)
  - disk artifacts: params.json, leaderboard.csv, plots (dpi fix), chain preds,
    holdout test predictions + metrics, pickled AutoML objects
  - resume: re-running the same session continues from disk
  - HTML report generation
  - Dash app boots and serves the index page + report route

Run from the project root:
    env\\python.exe -m dashboard.tests.e2e_smoke
"""
import base64
import glob
import io
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dashboard import _vendored  # noqa: F401  (vendored fork must be importable first)

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


def make_dataset(path, n=300, seed=7):
    rng = np.random.RandomState(seed)
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-3, 3, n)
    x3 = rng.randn(n) + 0.5 * x1
    x4 = rng.binomial(1, 0.4, n)
    cat = rng.choice(["A", "B", "C", "D"], size=n)
    x5 = rng.gamma(2.0, 2.0, n)
    df = pd.DataFrame({
        "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5,
        "cat": cat, "date": pd.date_range("2020-01-01", periods=n),
    })
    df.loc[rng.choice(n, 12, replace=False), "x3"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "x1"] = np.nan
    df["target_a"] = 2.0 * x1 - 1.5 * x2 + 0.8 * x3 + 0.3 * x5 + rng.normal(0, 0.4, n)
    df["target_b"] = 0.6 * x1 + 2.2 * x2 - 0.9 * df["target_a"] + rng.normal(0, 0.5, n)
    df["target_c"] = 1.1 * x3 - 0.4 * x4 + 0.5 * df["target_b"] + rng.normal(0, 0.3, n)
    df.to_csv(path, index=False)
    return df


def build_config():
    return {
        "automl": {
            "mode": "Compete",
            "eval_metric": "rmse",
            "validation_type": "kfold",
            "k_folds": 3,
            "train_ratio": 0.75,
            "total_time_limit": 150,
            "model_time_limit": None,
            "start_random_models": 2,
            "hill_climbing_steps": 1,
            "top_models_to_improve": 1,
            "train_ensemble": False,
            "stack_models": False,
            "explain_level": 2,
            "boost_on_errors": False,
            "random_state": 42,
            "n_jobs": -1,
            "features_selection": False,
        },
        "columns": {
            "features": ["x1", "x2", "x3", "x4", "x5", "cat", "date"],
            "targets": ["target_a", "target_b", "target_c"],
        },
        "split": {"enabled": True, "test_ratio": 0.15, "seed": 123},
        "algorithms": {
            "builtin": ["LightGBM", "Random Forest"],
            "sklearn": ["Ridge Regression", "Gradient Boosting"],
            "params": {"Ridge Regression": {"alpha": 2.5}},
        },
        "feature_engineering": {
            "golden": True,
            "golden_count": None,
            "kmeans": True,
            "mix_encoding": True,
        },
        "chain": {"enabled": True},
    }


def main():
    t0 = time.time()
    print("== 1. patches + registry ==")
    from dashboard.app import _apply_patches
    _apply_patches()

    from supervised.algorithms.registry import AlgorithmsRegistry, REGRESSION
    from dashboard.core.sklearn_regressors import SKLEARN_REGRESSORS
    reg = AlgorithmsRegistry.registry[REGRESSION]
    check("27 sklearn regressors registered", len(SKLEARN_REGRESSORS) >= 27,
          f"got {len(SKLEARN_REGRESSORS)}")
    check("'Ridge Regression' in registry", "Ridge Regression" in reg)
    check("'Gradient Boosting' in registry", "Gradient Boosting" in reg)

    workdir = tempfile.mkdtemp(prefix="automl_dash_test_")
    print(f"  workdir: {workdir}")

    print("== 2. data + upload parsing ==")
    csv_path = os.path.join(workdir, "data.csv")
    df = make_dataset(csv_path)
    from dashboard.core.data_io import load_file, parse_contents, profile_columns, excel_sheets
    df2 = load_file(csv_path)
    check("CSV loads", df2.shape == df.shape)
    with open(csv_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    df3, err = parse_contents("data:application/octet-stream;base64," + b64, "data.csv")
    check("base64 upload parses", df3 is not None and err is None and df3.shape == df.shape)
    prof = profile_columns(df)
    check("column profiling", prof["cat"]["kind"] == "categorical" and prof["x1"]["kind"] == "numeric")
    check("excel_sheets on csv -> []", excel_sheets(csv_path) == [])

    print("== 3. session management ==")
    from dashboard.core import session as sc
    s = sc.create_session(workdir, "e2e test")
    s.set_config(build_config())
    s.save()
    check("session saved", os.path.exists(s.config_path))
    s2 = sc.load_session(s.dir)
    check("session reloads", s2.config["automl"]["eval_metric"] == "rmse")
    check("session registry", len(sc.list_sessions(workdir)) == 1)
    s = s2

    print("== 4. feature engineering preview ==")
    from dashboard.core.fe_pipeline import run_fe_preview
    fe_cfg = s.config["feature_engineering"]
    fe = run_fe_preview(df, ["x1", "x2", "x3", "x4", "x5", "cat", "date"],
                        "target_a", fe_cfg, results_path=os.path.join(workdir, "fe"))
    check("FE adds engineered features", len(fe["added_columns"]) > 0,
          str(fe["added_columns"][:10]))
    check("FE keeps numeric + target", fe["n_features_after"] > fe["n_features_before"],
          f"before={fe['n_features_before']} after={fe['n_features_after']}")
    check("FE golden_features list", len(fe["golden_features"]) > 0)
    check("FE date transformed/removed", "date" not in fe["after_cols"] or
          any("datetime" in str(t) for t in fe["column_transforms"].get("date", [])),
          str(fe["column_transforms"].get("date")))

    print("== 5. chained multi-target training ==")
    from dashboard.core.chain import run_chain, get_automl_kwargs, build_algorithm_lists
    kw = get_automl_kwargs(s.config)
    check("automl kwargs mapped", kw["golden_features"] is True and kw["kmeans_features"] is True
          and kw["eval_metric"] == "rmse" and kw["mode"] == "Compete")
    b, e, o = build_algorithm_lists(s.config)
    check("algorithm lists", b == ["LightGBM", "Random Forest"]
          and set(e) == {"Ridge Regression", "Gradient Boosting"}
          and o == {"Ridge Regression": {"alpha": 2.5}})

    logs = []
    s._df = df  # in-process data (like the UI does after upload)
    ok = run_chain(s, log=lambda m: logs.append(m), stop_check=None, progress=lambda t, m: None)
    if ok:
        s.mark_finished()  # normally done by the runner thread
    check("run_chain returned ok", ok is True)
    check("session finished", s.status == "finished", s.status)
    check("logs captured", len(logs) > 15, f"got {len(logs)} lines")

    for t in ["target_a", "target_b", "target_c"]:
        rp = s.target_results_path(t)
        pj = os.path.join(rp, "params.json")
        exists = os.path.exists(pj)
        fit = False
        if exists:
            import json
            with open(pj, "r", encoding="utf-8") as f:
                fit = json.load(f).get("fit_level") == "finished"
        check(f"{t}: params.json + fit_level finished", exists and fit)
        ldb = os.path.join(rp, "leaderboard.csv")
        check(f"{t}: leaderboard.csv", os.path.exists(ldb))
        if os.path.exists(ldb):
            l = pd.read_csv(ldb)
            check(f"{t}: leaderboard non-empty", len(l) >= 1, f"rows={len(l)}")
            check(f"{t}: sklearn model on leaderboard",
                  l["model_type"].str.contains("Ridge|Gradient|RandomForest|LightGBM").any(),
                  str(l["model_type"].unique()[:6]))

    imgs = glob.glob(os.path.join(workdir, "sessions", "*", "results", "*.automl", "*.png"))
    check("plot images generated", len(imgs) >= 4, f"got {len(imgs)} pngs")
    big = [p for p in imgs if os.path.getsize(p) > 40_000]
    check("plots have decent size (dpi fix)", len(big) >= 3)
    try:
        from PIL import Image
        lc = [p for p in imgs if "learning" in os.path.basename(p)]
        if lc:
            w, h = Image.open(lc[0]).size
            check("learning curve image >= 1200px wide (dpi 150)", w >= 1200, f"{w}x{h}")
    except Exception:
        print("  (PIL not available, skipping image size check)")

    cp = os.path.join(s.dir, "chain_preds.csv")
    check("chain_preds.csv", os.path.exists(cp))
    if os.path.exists(cp):
        cpdf = pd.read_csv(cp)
        check("chain_preds has chain_ columns", "chain_target_a" in cpdf.columns
              and "chain_target_b" in cpdf.columns)

    tp = os.path.join(s.dir, "test_predictions.csv")
    check("test_predictions.csv", os.path.exists(tp))
    tm = s.config.get("results", {}).get("test_metrics", {})
    check("holdout metrics for all targets",
          set(tm) == {"target_a", "target_b", "target_c"}, str(tm))
    if tm:
        check("metrics contain RMSE/R2", all("RMSE" in m and "R2" in m for m in tm.values()))
        for t, m in tm.items():
            print(f"    {t}: RMSE={m['RMSE']} R2={m['R2']}")

    for t in ["target_a", "target_b", "target_c"]:
        check(f"pickle for {t}", os.path.exists(s.target_pickle_path(t)))
        aml = s.load_automl_pickle(t)
        check(f"pickle loads for {t}", aml is not None)
        if aml is not None:
            base = df[["x1", "x2", "x3", "x4", "x5", "cat", "date"]].head(10).copy()
            cpdf = s.load_chain_preds()
            if cpdf is not None:
                for col in cpdf.columns:
                    vals = cpdf[col].astype(float).iloc[:10]
                    base[col] = np.nan
                    base.loc[vals.index[vals.index < 10], col] = vals.values
            pred = aml.predict(base)
            check(f"pickled {t} predicts", pred is not None and len(pred) == 10,
                  str(None if pred is None else pred.shape))

    print("== 6. resume (re-run same session) ==")
    s2 = sc.load_session(s.dir)
    s2._df = df
    ok2 = run_chain(s2, log=lambda m: None, stop_check=None, progress=lambda t, m: None)
    if ok2:
        s2.mark_finished()
    check("resumed run ok", ok2 is True)
    check("resumed session finished", s2.status == "finished", s2.status)

    print("== 6b. pause / resume semantics ==")
    from dashboard.core import session as _sc
    s3 = _sc.load_session(s.dir)
    s3._df = df
    s3._finished_targets = {t for t in ["target_a", "target_b", "target_c"]
                            if s3.is_finished_on_disk(t)}
    s3.resume_mode = True
    logs3 = []
    ok3 = run_chain(s3, log=logs3.append, stop_check=None, progress=lambda t, m: None)
    check("resume skips finished targets", ok3 is True, str(ok3))
    check("resume reused from disk",
          any("already finished" in m for m in logs3), str(logs3[:2]))

    s4 = _sc.load_session(s.dir)
    s4._df = df
    s4.resume_mode = True
    s4._finished_targets = set()
    paused = {"flag": False}
    def stop_check():
        return "pause"
    ok4 = run_chain(s4, log=lambda m: None, stop_check=stop_check,
                    progress=lambda t, m: None)
    check("pause returns False", ok4 is False)
    check("pause marks status", s4.status == "paused", s4.status)

    print("== 6c. collective live figures ==")
    from dashboard.core.reports import (performance_scatter_figure,
                                        leaderboard_boxplot_figure, training_time_figure,
                                        features_importance_heatmap, models_correlation_figure)
    tgt = s2.last_finished_target()
    rp = s2.target_results_path(tgt)
    recs = [{"name": r["name"], "metric": float(r["metric_value"]),
             "time": float(r["train_time"]), "model_type": r["model_type"]}
            for r in s2.load_leaderboard(rp, 40)]
    for name, fig in [("perf scatter", performance_scatter_figure(recs, "rmse")),
                      ("boxplot", leaderboard_boxplot_figure(recs, "rmse")),
                      ("train time", training_time_figure(recs)),
                      ("feat heatmap", features_importance_heatmap(rp)),
                      ("corr heatmap", models_correlation_figure(rp))]:
        check(f"plotly fig {name} has traces", len(fig.data) > 0,
              f"traces={len(fig.data)}")
    from dashboard.core.summarizer import summarize
    bullets = summarize(s2, s2.summary())
    check("summarizer produces bullets", len(bullets) >= 2, str(bullets))
    check("summary mentions best model",
          any("Best model" in b[1] for b in bullets), str(bullets))

    print("== 7. HTML report ==")
    from dashboard.core import reports as rep
    report = rep.build_html_report(s2)
    check("report.html generated", os.path.exists(report))
    check("report non-trivial size", os.path.getsize(report) > 50_000,
          f"{os.path.getsize(report)} bytes")
    from dashboard.core.reports import leaderboard_figure, learning_curves_grid, \
        importance_figure, predictions_figure, residuals_figure, features_correlation_figure
    f1 = leaderboard_figure([{"name": "A", "metric": 1.0, "time": 2.0, "model_type": "X"}], "rmse")
    f2 = learning_curves_grid(s.target_results_path("target_a"))
    f3 = importance_figure(s.target_results_path("target_a"))
    f4 = predictions_figure([1.0, 2.0, 3.0], [1.2, 1.8, 3.1])
    f5 = residuals_figure([1.0, 2.0, 3.0], [1.2, 1.8, 3.1])
    f6 = features_correlation_figure(df, ["x1", "x2"], "target_a")
    for name, fig in [("leaderboard", f1), ("lcurves", f2), ("importance", f3),
                      ("pred", f4), ("resid", f5), ("corr", f6)]:
        check(f"plotly fig {name} has traces", len(fig.data) > 0)

    print("== 8. Dash app boot ==")
    from dashboard.app import create_app
    from dashboard.ui.state import get_state
    get_state().set_workdir(os.path.dirname(os.path.dirname(s2.dir)))
    app = create_app()
    client = app.server.test_client()
    r = client.get("/")
    check("index page 200", r.status_code == 200, str(r.status_code))
    r2 = client.get(f"/reports/{s2.id}/report.html")
    check("report route 200", r2.status_code == 200, str(r2.status_code))
    check("report content served", r2.data[:100].count(b"<") > 0)

    from dash._utils import split_callback_id
    import json as _json

    def _post(name, overrides=None):
        for key, entry in app.callback_map.items():
            if entry["callback"].__name__ != name:
                continue
            out = [{"id": i.get("id"), "property": i.get("property"),
                    "value": (overrides or {}).get(i.get("id"), i.get("value"))}
                   for i in entry.get("inputs", [])]
            state = [{"id": s.get("id"), "property": s.get("property"),
                      "value": s.get("value")} for s in entry.get("state", [])]
            flat = []
            spec = split_callback_id(key)

            def _fl(x):
                for i in (x if isinstance(x, list) else [x]):
                    if isinstance(i, list):
                        _fl(i)
                    else:
                        flat.append(i)
            _fl(spec)
            payload = {"output": key,
                       "outputs": flat[0] if len(flat) == 1 else flat,
                       "changedPropIds": [i["id"] + "." + i["property"] for i in out],
                       "inputs": out, "state": state}
            r = client.post("/_dash-update-component",
                            data=_json.dumps(payload), content_type="application/json")
            return r.status_code
        return None

    get_state().session = s2
    st0 = _post("poll_all", {"poll-interval": 1})
    check("callback poll_all", st0 == 200, str(st0))
    cbx = app.callback_map[next(k for k in app.callback_map
                                if "console-progress" in k)]
    poll_out = cbx["callback"].__wrapped__(1, None)
    check("poll returns 34 outputs", len(poll_out) == 34, str(len(poll_out)))
    check("console progress label", "/" in str(poll_out[-2]), str(poll_out[-2]))
    st1 = _post("upload_data")
    check("callback upload_data", st1 == 200, str(st1))
    st2 = _post("new_session", {"btn-new-session": 1})
    check("callback new_session", st2 == 200, str(st2))
    st3 = _post("fe_preview")
    check("callback fe_preview", st3 in (200, 204), str(st3))
    tgt = s2.last_finished_target()
    lb = s2.load_leaderboard(s2.target_results_path(tgt), 1)
    st4 = _post("model_detail", {"model-select": lb[0]["name"], "poll-interval": 1})
    check("callback model_detail", st4 == 200, str(st4))
    from dashboard.core.reports import model_predictions_figure
    mdir = os.path.join(s2.target_results_path(tgt), str(lb[0]["name"]))
    fmp = model_predictions_figure(mdir, str(lb[0]["name"]))
    check("oof predictions figure has data", len(fmp.data) > 0)

    print(f"\nPASS={PASS} FAIL={FAIL}  elapsed={time.time()-t0:.0f}s")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
