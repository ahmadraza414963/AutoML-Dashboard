# -*- coding: utf-8 -*-
"""Plotly figure builders for live/interactive reporting inside the dashboard."""
import glob
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRIC_TITLES = {
    "rmse": "RMSE", "mse": "MSE", "mae": "MAE", "r2": "R²", "mape": "MAPE",
    "spearman": "Spearman", "pearson": "Pearson",
}
_BADGE_LAYOUT = dict(margin=dict(t=50, b=40, l=60, r=20),
                     template="plotly_white", height=420)


def _metric_title(name):
    return METRIC_TITLES.get(name, name)


def leaderboard_figure(records, metric="rmse", title=None):
    """records: list of dicts (name, metric, time, model_type) sorted by metric.
    Accepts both the fork's leaderboard.csv schema (metric_value, train_time)
    and the dashboard's normalized schema."""
    if not records:
        return go.Figure(go.Scatter(x=[], y=[])).update_layout(title="No models yet")
    recs = []
    for r in records:
        recs.append({
            "name": r.get("name", "?"),
            "metric": float(r.get("metric_value", r.get("metric", 0))),
            "time": float(r.get("train_time", r.get("elapsed_time", 0))),
            "model_type": r.get("model_type", "?"),
        })
    recs = [r for r in recs if r["metric"] == r["metric"]]  # drop NaN
    if not recs:
        return go.Figure(go.Scatter(x=[], y=[])).update_layout(title="No models yet")
    opt_neg = metric in ("r2", "spearman", "pearson")
    best = max(r["metric"] for r in recs) if opt_neg else min(r["metric"] for r in recs)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2],
                        specs=[[{}, {"type": "domain"}]],
                        subplot_titles=(_metric_title(metric), "Model type"))
    fig.add_trace(go.Bar(
        x=[r["name"] for r in recs], y=[r["metric"] for r in recs],
        marker_color=["#2E7D32" if r["metric"] == best else "#1E88E5"
                      for r in recs],
        hovertemplate="%{x}<br>" + _metric_title(metric) + " = %{y:.4g}<extra></extra>",
        showlegend=False), 1, 1)
    types = sorted({r["model_type"] for r in recs})
    counts = {t: sum(1 for r in recs if r["model_type"] == t) for t in types}
    fig.add_trace(go.Pie(labels=list(counts.keys()), values=list(counts.values()),
                         hole=0.4, showlegend=False), 1, 2)
    fig.update_layout(**_BADGE_LAYOUT,
                      title=title or f"Leaderboard ({_metric_title(metric)})")
    fig.update_xaxes(tickangle=-25)
    return fig


def _norm_records(records):
    """Normalize leaderboard records to (name, metric, time, model_type)."""
    recs = []
    for r in records or []:
        m = float(r.get("metric_value", r.get("metric", float("nan"))))
        if m != m:
            continue
        recs.append({
            "name": r.get("name", "?"),
            "metric": m,
            "time": float(r.get("train_time", r.get("elapsed_time",
                       r.get("time", 0))) or 0),
            "model_type": r.get("model_type", "?"),
        })
    return recs


def performance_scatter_figure(records, metric="rmse"):
    """Live re-implementation of the fork's ldb_performance.png."""
    recs = _norm_records(records)
    if not recs:
        return go.Figure(go.Scatter(x=[], y=[])).update_layout(title="No models yet")
    types = sorted({r["model_type"] for r in recs})
    fig = go.Figure()
    for t in types:
        sub = [r for r in recs if r["model_type"] == t]
        fig.add_trace(go.Scatter(
            x=list(range(len(sub))), y=[r["metric"] for r in sub], mode="lines+markers",
            name=t, hovertemplate="%{text}<br>%{y:.4g}<extra>" + t + "</extra>",
            text=[r["name"] for r in sub]))
    fig.update_layout(**_BADGE_LAYOUT,
                      title=f"AutoML performance — {_metric_title(metric)} per model (live)",
                      xaxis_title="Model (per type)", yaxis_title=_metric_title(metric))
    return fig


def leaderboard_boxplot_figure(records, metric="rmse"):
    """Live re-implementation of the fork's ldb_performance_boxplot.png."""
    recs = _norm_records(records)
    if not recs:
        return go.Figure(go.Scatter(x=[], y=[])).update_layout(title="No models yet")
    opt_neg = metric in ("r2", "spearman", "pearson")
    by_type = {}
    for r in recs:
        by_type.setdefault(r["model_type"], []).append(r["metric"])
    order = sorted(by_type, key=lambda t: min(by_type[t]), reverse=opt_neg)
    fig = go.Figure()
    for t in order:
        fig.add_trace(go.Box(y=by_type[t], name=t, boxpoints="all",
                             jitter=0.3, pointpos=0,
                             hovertemplate=t + "<br>%{y:.4g}<extra></extra>"))
    fig.update_layout(**_BADGE_LAYOUT,
                      title=f"Performance boxplot by model type — {_metric_title(metric)} (live)",
                      yaxis_title=_metric_title(metric))
    return fig


def training_time_figure(records):
    """Training time per model (live)."""
    recs = _norm_records(records)
    if not recs:
        return go.Figure(go.Scatter(x=[], y=[])).update_layout(title="No models yet")
    recs = sorted(recs, key=lambda r: r["time"])
    fig = go.Figure(go.Bar(x=[r["name"] for r in recs], y=[r["time"] for r in recs],
                           marker_color="#00897B",
                           hovertemplate="%{x}: %{y:.1f} s<extra></extra>"))
    fig.update_layout(**_BADGE_LAYOUT, title="Training time per model (live)",
                      xaxis_title="Model", yaxis_title="Seconds")
    fig.update_xaxes(tickangle=-30)
    return fig


def features_importance_heatmap(results_path, max_models=30, top_features=25):
    """Live re-implementation of the fork's features_heatmap.png: per-model
    permutation importance (mean across folds) as a heatmap."""
    import glob as _glob
    if not os.path.isdir(results_path):
        return go.Figure().update_layout(title="No results yet")
    model_imp = {}
    for mdir in sorted(os.listdir(results_path)):
        p = os.path.join(results_path, mdir)
        if not os.path.isdir(p) or mdir.startswith("_"):
            continue
        files = [f for f in _glob.glob(os.path.join(p, "*_importance.csv"))
                 if "shap" not in os.path.basename(f)]
        if not files:
            continue
        frames = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
        if not frames:
            continue
        key = "feature" if "feature" in frames[0].columns else frames[0].columns[0]
        val = [c for c in frames[0].columns if c != key][0]
        df = pd.concat([f[[key, val]] for f in frames], ignore_index=True)
        model_imp[mdir] = df.groupby(key, as_index=False)[val].mean().set_index(key)[val]
        if len(model_imp) >= max_models:
            break
    if len(model_imp) < 2:
        return go.Figure().update_layout(title="Feature heatmap — need ≥ 2 models with importance data")
    mfi = pd.DataFrame(model_imp).fillna(0.0)
    mfi = mfi.loc[mfi.mean(axis=1).sort_values(ascending=False).index]
    if len(mfi) > top_features:
        mfi = mfi.head(top_features)
    fig = go.Figure(go.Heatmap(
        z=mfi.values, x=mfi.columns, y=mfi.index, colorscale="Blues",
        zmin=0, zmax=float(mfi.values.max()) if mfi.values.size else 1,
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>"))
    kw = dict(_BADGE_LAYOUT)
    kw["height"] = min(900, 120 + 18 * len(mfi))
    fig.update_layout(**kw, title="Feature importance across models (live)")
    fig.update_xaxes(tickangle=-30)
    return fig


def models_correlation_figure(results_path, max_models=30):
    """Live re-implementation of the fork's correlation_heatmap.png: spearman
    correlation of out-of-fold predictions between every pair of models."""
    from scipy.stats import spearmanr
    if not os.path.isdir(results_path):
        return go.Figure().update_layout(title="No results yet")
    names, preds = [], []
    for mdir in sorted(os.listdir(results_path)):
        p = os.path.join(results_path, mdir)
        f = os.path.join(p, "predictions_out_of_folds.csv")
        if not os.path.exists(f):
            f = os.path.join(p, "predictions_validation.csv")
        if not os.path.isfile(f):
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        cols = [c for c in df.columns if "prediction" in c]
        if not cols:
            cols = list(df.columns)
        ser = df[cols[0]].astype(float).reset_index(drop=True)
        if len(ser) < 10:
            continue
        names.append(mdir)
        preds.append(ser)
        if len(names) >= max_models:
            break
    if len(names) < 2:
        return go.Figure().update_layout(title="Model correlation — need ≥ 2 models with predictions")
    n = len(names)
    corr = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = np.nan
            try:
                with np.errstate(invalid="ignore"):
                    res = spearmanr(preds[i], preds[j])
                    v = res.statistic if hasattr(res, "statistic") else res[0]
            except Exception:
                v = np.nan
            corr[i, j] = corr[j, i] = v
    fig = go.Figure(go.Heatmap(
        z=corr, x=names, y=names, colorscale="Blues", zmin=0, zmax=1,
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>"))
    kw = dict(_BADGE_LAYOUT)
    kw["height"] = min(900, 150 + 18 * n)
    fig.update_layout(**kw, title="Spearman correlation of model predictions (live)")
    fig.update_xaxes(tickangle=-45)
    return fig


def predictions_figure(obs, pred, metric_str=""):
    obs = np.asarray(obs, dtype=float); pred = np.asarray(pred, dtype=float)
    m = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[m], pred[m]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obs, y=pred, mode="markers", name="samples",
                             marker=dict(size=6, color="#1E88E5", opacity=0.6),
                             hovertemplate="Actual=%{x:.4g}<br>Pred=%{y:.4g}<extra></extra>"))
    lo, hi = float(np.min([obs.min(), pred.min()])), float(np.max([obs.max(), pred.max()]))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = ŷ",
                             line=dict(color="black", dash="dash")))
    fig.update_layout(**_BADGE_LAYOUT,
                      title="Predictions vs Actuals" + (f" ({metric_str})" if metric_str else ""),
                      xaxis_title="Actual", yaxis_title="Predicted")
    return fig


def residuals_figure(obs, pred):
    obs = np.asarray(obs, dtype=float); pred = np.asarray(pred, dtype=float)
    m = np.isfinite(obs) & np.isfinite(pred)
    obs, pred, res = obs[m], pred[m], pred[m] - obs[m]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3],
                        subplot_titles=("Residuals vs Actual", "Residuals histogram"))
    fig.add_trace(go.Scatter(x=obs, y=res, mode="markers", showlegend=False,
                             marker=dict(size=6, color="#00897B", opacity=0.6),
                             hovertemplate="Actual=%{x:.4g}<br>Residual=%{y:.4g}<extra></extra>"), 1, 1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_trace(go.Histogram(x=res, nbinsx=min(60, max(10, len(res) // 5)),
                               showlegend=False, marker_color="#00897B"), 1, 2)
    fig.update_layout(**_BADGE_LAYOUT, title="Residuals",
                      xaxis_title="Actual", yaxis_title="Residual")
    return fig


def _parse_training_log(log_path):
    it, tr, va = [], [], []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = re.split(r"[\s,]+", line.strip())
                if len(parts) >= 3 and parts[0].replace(".", "", 1).replace("-", "").isdigit():
                    try:
                        it.append(float(parts[0]))
                        tr.append(float(parts[1]))
                        va.append(float(parts[2]))
                    except ValueError:
                        continue
    except Exception:
        pass
    return it, tr, va


def learning_curve_figure(log_path):
    """Parse mljar {learner}_training.log into a figure."""
    iter_no, train, val = _parse_training_log(log_path)
    if not iter_no:
        return go.Figure(go.Scatter(x=[], y=[])).update_layout(title="No learning curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iter_no, y=train, name="train",
                             line=dict(color="#1E88E5", width=2)))
    fig.add_trace(go.Scatter(x=iter_no, y=val, name="validation",
                             line=dict(color="#E53935", width=2)))
    fig.update_layout(**_BADGE_LAYOUT,
                      title=os.path.basename(log_path).replace("_training.log", ""),
                      xaxis_title="Iteration", yaxis_title="Loss")
    return fig


def learning_curves_grid(results_path, max_rows=6, title="Learning curves"):
    logs = sorted(glob.glob(os.path.join(results_path, "*_training.log")))
    if not logs:
        return go.Figure(go.Scatter(x=[], y=[])).update_layout(title="No training logs")
    logs = logs[:max_rows]
    cols = 2
    rows = int(np.ceil(len(logs) / cols))
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[os.path.basename(l).replace("_training.log", "")[:26]
                                        for l in logs])
    for i, log in enumerate(logs):
        it, tr, va = _parse_training_log(log)
        r, c = i // cols + 1, i % cols + 1
        if it:
            fig.add_trace(go.Scatter(x=it, y=tr, name="train", legendgroup="train", showlegend=False,
                                     line=dict(color="#1E88E5", width=1.5)), r, c)
            fig.add_trace(go.Scatter(x=it, y=va, name="validation", legendgroup="val", showlegend=False,
                                     line=dict(color="#E53935", width=1.5)), r, c)
    fig.update_layout(margin=dict(t=60, b=40, l=60, r=20), template="plotly_white",
                      height=rows * 260, title=title)
    return fig


def _importance_df(results_path, model_name, shap=False):
    if shap:
        files = sorted(glob.glob(os.path.join(results_path, "*_shap_importance.csv")))
        frames = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
        if not frames:
            return None, None
        key = "feature" if "feature" in frames[0].columns else frames[0].columns[0]
        if len(frames) > 1:
            val = [c for c in frames[0].columns if c != key][0]
            data = pd.concat([f[[key, val]] for f in frames], ignore_index=True)
            return data.groupby(key, as_index=False)[val].mean(), "aggregate"
        return frames[0], files[0]
    files = [f for f in sorted(glob.glob(os.path.join(results_path, "*_importance.csv")))
             if "shap" not in os.path.basename(f)]
    frames = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
    if not frames:
        return None, None
    key = "feature" if "feature" in frames[0].columns else frames[0].columns[0]
    if len(frames) > 1:
        num_cols = [c for c in frames[0].columns if c != key]
        if num_cols:
            val = num_cols[0]
            data = pd.concat([f[[key, val]] for f in frames], ignore_index=True)
            return data.groupby(key, as_index=False)[val].mean(), "aggregate"
    return frames[0], files[0]


def importance_figure(results_path, model_name=None, shap=False):
    df, fname = _importance_df(results_path, model_name, shap)
    if df is None:
        title = ("SHAP not available for this model type"
                 if shap else "No importance data")
        return go.Figure(go.Scatter(x=[], y=[])).update_layout(title=title)
    cols = [c for c in df.columns if c not in ("feature",)]
    if not cols:
        cols = [df.columns[1]]
    val_col = cols[0]
    d = df.sort_values(val_col, ascending=True)
    d = d.iloc[-50:]
    fig = go.Figure(go.Bar(x=d[val_col], y=d["feature"], orientation="h",
                           marker_color="#8E24AA" if shap else "#00897B",
                           hovertemplate="%{y}: %{x:.4g}<extra></extra>"))
    kw = dict(_BADGE_LAYOUT)
    kw["height"] = min(1100, 180 + len(d) * 16)
    name = model_name or os.path.basename(results_path)
    fig.update_layout(**kw, title=("SHAP importance" if shap else "Permutation importance")
                      + f" — {name}")
    return fig


def target_distribution_figure(df, target):
    y = pd.to_numeric(df[target], errors="coerce").dropna()
    fig = go.Figure(go.Histogram(x=y, nbinsx=min(80, max(10, int(len(y) ** 0.5) * 4)),
                                 marker_color="#FB8C00"))
    fig.update_layout(**_BADGE_LAYOUT, title=f"Distribution of target '{target}'",
                      xaxis_title=target, yaxis_title="count")
    return fig


def features_correlation_figure(df, features, target=None, max_cols=45):
    cols = [c for c in features if c in df.columns]
    if target and target in df.columns:
        cols = cols + [target]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    if sub.shape[1] == 0:
        return go.Figure().update_layout(title="No numeric columns")
    corr = sub.corr(method="spearman").iloc[:max_cols, :max_cols]
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                               colorscale="RdBu", zmid=0,
                               text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
                               hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>"))
    kw = dict(_BADGE_LAYOUT)
    kw["height"] = min(900, 90 + 16 * len(corr))
    fig.update_layout(**kw, title="Spearman correlation (numeric columns)")
    return fig


def model_predictions_figure(results_path, model_name=None):
    """Scatter of validation predictions for one model (predictions_out_of_folds.csv)."""
    f = os.path.join(results_path, "predictions_out_of_folds.csv")
    if not os.path.exists(f):
        return go.Figure().update_layout(title="No OOF predictions for this model")
    try:
        df = pd.read_csv(f)
        cols = list(df.columns)
        obs_col = next((c for c in cols if c.lower().startswith("target")), cols[0])
        pred_col = next((c for c in cols if c != obs_col), cols[-1])
        return predictions_figure(df[obs_col], df[pred_col],
                                  model_name or os.path.basename(results_path))
    except Exception:
        return go.Figure().update_layout(title="No predictions file")


def chain_progress_figure(records):
    """records: list of (target, metric, time_sec)."""
    if not records:
        return go.Figure().update_layout(title="No targets finished yet")
    targets = [r[0] for r in records]
    metrics = [r[1] for r in records]
    times = [r[2] for r in records]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Best metric per target", "Training time per target"))
    fig.add_trace(go.Bar(x=targets, y=metrics, marker_color="#1E88E5",
                         hovertemplate="%{x}: %{y:.4g}<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(x=targets, y=times, marker_color="#00897B",
                         hovertemplate="%{x}: %{y:.1f} s<extra></extra>"), 1, 2)
    fig.update_layout(**_BADGE_LAYOUT, title="Chain run progress")
    return fig


def build_html_report(session):
    """Assemble all figures + tables into a single standalone HTML report."""
    from plotly.offline import plot as offline_plot
    parts = []
    parts.append("<h1>AutoML Dashboard report — session " + session.id + "</h1>")
    parts.append(pd.DataFrame([session.summary()]).to_html())
    target = session.last_finished_target()
    if target:
        results_path = session.target_results_path(target)
        ldb_path = os.path.join(results_path, "leaderboard.csv")
        ldb = pd.read_csv(ldb_path) if os.path.exists(ldb_path) else None
        if ldb is not None:
            parts.append("<h2>Leaderboard</h2>")
            parts.append(ldb.head(50).to_html())
            fig = leaderboard_figure(ldb.to_dict("records"),
                                     metric=str(session.config.get("automl", {}).get("eval_metric", "rmse")))
            parts.append(offline_plot(fig, output_type="div"))
            best = str(ldb.iloc[0]["name"]) if len(ldb) else None
            mdir = os.path.join(results_path, best) if best and os.path.isdir(
                os.path.join(results_path, best)) else results_path
            parts.append(offline_plot(learning_curves_grid(mdir), output_type="div"))
            parts.append(offline_plot(importance_figure(mdir), output_type="div"))
            oof = os.path.join(mdir, "predictions_out_of_folds.csv")
            if os.path.exists(oof):
                dfo = pd.read_csv(oof)
                oc = next((c for c in dfo.columns if c.lower().startswith("target")), dfo.columns[0])
                pc = next((c for c in dfo.columns if c != oc), dfo.columns[-1])
                parts.append(offline_plot(predictions_figure(dfo[oc], dfo[pc], best or ""),
                                          output_type="div"))
                parts.append(offline_plot(residuals_figure(dfo[oc], dfo[pc]), output_type="div"))
    out = os.path.join(session.dir, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>Session report</title></head><body>")
        f.write("".join(parts))
        f.write("</body></html>")
    return out