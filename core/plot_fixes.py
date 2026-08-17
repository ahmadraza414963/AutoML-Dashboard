# -*- coding: utf-8 -*-
"""
Fixes the poorly-rendered matplotlib plots produced by the mlsuper
(mljar-supervised fork) package:

  * every `fig.savefig(...)` defaults to dpi=150 + bbox_inches='tight'
  * learning-curve plots: bigger figures, rotated/short tick labels,
    legend outside the axes, no label collisions
  * leaderboard plots: model names on the x-axis (rotated), dynamic boxplot
  * feature-importance heatmaps / bar charts: dynamic sizes, truncated &
    rotated labels
  * regression predictions plot: identity line, fixed residual histogram
  * SHAP summary figure gets an explicit big figure so y labels don't merge
"""
import os
from dashboard import _vendored  # noqa: F401  (vendored fork must be importable first)

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def _truncate(name, n=38):
    name = str(name)
    return name if len(name) <= n else name[: n - 3] + "..."


def _patch_figure_savefig():
    from matplotlib.figure import Figure

    _orig = Figure.savefig

    def _savefig(self, *args, **kwargs):
        kwargs.setdefault("dpi", 150)
        kwargs.setdefault("bbox_inches", "tight")
        return _orig(self, *args, **kwargs)

    Figure.savefig = _savefig


def _patch_learning_curves():
    import supervised.utils.learning_curves as lc

    _orig_single = lc.LearningCurves.plot_single_iter
    _orig_iters = lc.LearningCurves.plot_iterations
    _orig_ens = lc.LearningCurves.plot_for_ensemble

    def plot_single_iter(learner_names, metric_name, model_path, colors):
        model_path = str(model_path)
        my_colors = list(colors)
        n = len(learner_names)
        while len(my_colors) < n:
            my_colors += colors
        fig, ax = plt.subplots(figsize=(max(11, 0.55 * n), 8))
        names = []
        for i, learner_name in enumerate(learner_names):
            fname = os.path.join(model_path, f"{learner_name}_training.log")
            if not os.path.exists(fname):
                continue
            df = pd.read_csv(fname, names=["iteration", "train", "test"])
            if df.shape[0] != 1:
                continue
            names.append(learner_name)
            ax.bar(f"F{learner_name}", df["test"].values[0], color=my_colors[i], edgecolor="white")
            ax.bar(f"F{learner_name}_tr", df["train"].values[0], color=my_colors[i], alpha=0.4, edgecolor="white")
        if not names:
            plt.close(fig)
            return
        ax.set_title(f"Learning curve ({metric_name}) - single iteration")
        ax.set_ylabel(metric_name)
        ax.set_xticks(range(0, 2 * len(names), 2))
        ax.set_xticklabels([_truncate(ln, 22) for ln in names], rotation=45, ha="right", fontsize=9)
        ax.legend(["validation", "train"], fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(model_path, "learning_curves.png"))
        plt.close(fig)

    def plot_iterations(learner_names, metric_name, model_path, colors, trees_in_iteration=None):
        model_path = str(model_path)
        my_colors = list(colors)
        n = len(learner_names)
        while len(my_colors) < n:
            my_colors += colors
        fig, ax = plt.subplots(figsize=(max(12, 0.9 * n), 8))
        for i, learner_name in enumerate(learner_names):
            fname = os.path.join(model_path, f"{learner_name}_training.log")
            if not os.path.exists(fname):
                continue
            df = pd.read_csv(fname, names=["iteration", "train", "test"])
            if df.shape[0] <= 1:
                continue
            df = df.apply(pd.to_numeric, errors="coerce")
            if df["test"].isnull().all():
                continue
            if trees_in_iteration is not None:
                df["iteration"] = df["iteration"] * trees_in_iteration + trees_in_iteration
            ax.plot(df["iteration"], df["train"], color=my_colors[i], linestyle="--", linewidth=1.2)
            ax.plot(df["iteration"], df["test"], color=my_colors[i], label=_truncate(learner_name, 26), linewidth=1.4)
            try:
                best_idx = df["test"].idxmax() if lc.Metric.optimize_negative(metric_name) else df["test"].idxmin()
                ax.axvline(df["iteration"].iloc[best_idx], color=my_colors[i], alpha=0.25, linewidth=0.8)
            except Exception:
                pass
        if n <= 18:
            ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
        ax.set_title(f"Learning curve ({metric_name})")
        ax.set_xlabel("iteration")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(model_path, "learning_curves.png"))
        plt.close(fig)

    def plot_for_ensemble(scores, metric_name, model_path):
        model_path = str(model_path)
        if not scores:
            return
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(range(1, len(scores) + 1), scores, marker="o", markersize=4, linewidth=1.4)
        ax.set_title(f"Ensemble learning curve ({metric_name})")
        ax.set_xlabel("step")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(model_path, "learning_curves.png"))
        plt.close(fig)

    lc.LearningCurves.plot_single_iter = staticmethod(plot_single_iter)
    lc.LearningCurves.plot_iterations = staticmethod(plot_iterations)
    lc.LearningCurves.plot_for_ensemble = staticmethod(plot_for_ensemble)


def _patch_leaderboard_plots():
    import supervised.utils.leaderboard_plots as lbp

    _orig = lbp.LeaderboardPlots.compute

    def compute(ldb, model_path, fout):
        model_path = str(model_path)
        if ldb.shape[0] < 2:
            return
        ldb = ldb.reset_index(drop=True)
        try:
            fig, ax = plt.subplots(figsize=(max(10, 0.35 * ldb.shape[0]), 7))
            ax.plot(range(1, ldb.shape[0] + 1), ldb.metric_value, marker="*", markersize=9, ls="none", alpha=0.85)
            ax.set_xlabel("Model (leaderboard order)")
            ax.set_ylabel(ldb.metric_type.iloc[0])
            ax.set_title("AutoML performance")
            ax.set_xticks(range(1, ldb.shape[0] + 1))
            ax.set_xticklabels([_truncate(m, 20) for m in ldb.name], rotation=45, ha="right", fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(model_path, "ldb_performance.png"))
            plt.close(fig)
        except Exception as e:
            print(f"leaderboard star plot failed: {e}")

        try:
            df2 = ldb[["model_type", "metric_value"]].copy()
            mins = df2.groupby("model_type").metric_value.transform("min")
            df2["m"] = df2.metric_value - mins
            df2 = df2[df2.m == 0]
            order = df2.groupby("model_type").metric_value.min().sort_values().index
            fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(order)), 7))
            data = [ldb[ldb.model_type == t].metric_value.values for t in order]
            ax.boxplot(data, labels=[_truncate(t, 24) for t in order])
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            ax.set_ylabel(ldb.metric_type.iloc[0])
            ax.set_title("AutoML performance by model type")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(model_path, "ldb_performance_boxplot.png"))
            plt.close(fig)
        except Exception as e:
            print(f"leaderboard boxplot failed: {e}")

    lbp.LeaderboardPlots.compute = staticmethod(compute)


def _patch_automl_plots():
    import supervised.utils.automl_plots as ap

    _orig_mfi = ap.AutoMLPlots.models_feature_importance
    _orig_corr = ap.AutoMLPlots.models_correlation

    def models_feature_importance(results_path, models):
        results_path = str(results_path)
        mfi = {}
        for m in models:
            model_path = os.path.join(results_path, m.get_name())
            if not os.path.isdir(model_path):
                continue
            for f in os.listdir(model_path):
                if f.endswith("_importance.csv") and "shap" not in f:
                    try:
                        imp = pd.read_csv(os.path.join(model_path, f), index_col=0)
                        imp = imp.iloc[:, 0]
                        mfi[m.get_name()] = imp
                    except Exception:
                        pass
        if not mfi:
            return
        dfm = pd.DataFrame(mfi).dropna(how="all")
        if dfm.empty:
            return
        mean_imp = dfm.mean(axis=1).sort_values(ascending=False).head(25)
        features = mean_imp.index.tolist()
        models_names = dfm.columns.tolist()
        heat = dfm.loc[features].T
        fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(models_names) + 3), max(8, 0.32 * len(features) + 2)))
        ax.imshow(heat.values, cmap="Blues", aspect="auto")
        ax.set_yticks(range(len(models_names)))
        ax.set_yticklabels([_truncate(m, 22) for m in models_names], fontsize=8)
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([_truncate(f, 34) for f in features], rotation=45, ha="right", fontsize=8)
        ax.set_title("Permutation feature importance (mean across folds)")
        ax.figure.colorbar(ax.images[0], ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(results_path, "features_heatmap.png"))
        plt.close(fig)

    def models_correlation(results_path, models):
        results_path = str(results_path)
        try:
            preds = []
            names = []
            for m in models:
                oof = m.get_out_of_folds()
                if oof is None:
                    continue
                cols = [c for c in oof.columns if "prediction" in c]
                if not cols:
                    continue
                preds.append(oof[cols].mean(axis=1).rename(m.get_name()))
                names.append(m.get_name())
            if len(preds) < 2:
                return
            df = pd.concat(preds, axis=1)
            corr = df.corr(method="spearman")
            n = corr.shape[0]
            fig, ax = plt.subplots(figsize=(max(10, 0.45 * n), max(9, 0.45 * n)))
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(n))
            ax.set_xticklabels([_truncate(c, 18) for c in corr.columns], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels([_truncate(c, 18) for c in corr.index], fontsize=8)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
            ax.set_title("Spearman correlation of models' out-of-fold predictions")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(os.path.join(results_path, "correlation_heatmap.png"))
            plt.close(fig)
        except Exception as e:
            print(f"models_correlation failed: {e}")

    ap.AutoMLPlots.models_feature_importance = staticmethod(models_feature_importance)
    ap.AutoMLPlots.models_correlation = staticmethod(models_correlation)


def _patch_additional_metrics():
    import supervised.utils.additional_metrics as am

    def _barh_plot(fout, model_path, fold_cnt, repeat_cnt, fname_src, out_fname, xlabel, title):
        model_path = str(model_path)
        all_importance = []
        for f in os.listdir(model_path):
            if f.endswith(fname_src):
                df = pd.read_csv(os.path.join(model_path, f), index_col=0)
                df.columns = [f]
                all_importance += [df]
        if not all_importance:
            return
        df_importance = pd.concat(all_importance, axis=1)
        df_importance = df_importance.mean(axis=1).sort_values()
        df_importance = df_importance.tail(25)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(df_importance) + 2)))
        ax.barh([_truncate(i, 40) for i in df_importance.index], df_importance.values, height=0.7)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(model_path, out_fname))
        plt.close(fig)

    def add_permutation_importance(fout, model_path, fold_cnt, repeat_cnt):
        _barh_plot(fout, model_path, fold_cnt, repeat_cnt, "_importance.csv", "permutation_importance.png",
                   "Mean of feature importance", "Top-25 important features")

    def add_shap_importance(fout, model_path, fold_cnt, repeat_cnt):
        _barh_plot(fout, model_path, fold_cnt, repeat_cnt, "_shap_importance.csv", "shap_importance.png",
                   "mean(|SHAP value|)", "Top-25 features by mean |SHAP value|")

    am.AdditionalMetrics.add_permutation_importance = staticmethod(add_permutation_importance)
    am.AdditionalMetrics.add_shap_importance = staticmethod(add_shap_importance)


def _patch_additional_plots():
    import supervised.utils.additional_plots as aps

    _orig_append = aps.AdditionalPlots.append

    def append(fout, model_path, plots):
        model_path = str(model_path)
        for p in plots or []:
            fig = p.get("figure")
            if fig is None:
                continue
            try:
                fig.tight_layout()
            except Exception:
                pass
            fig.savefig(os.path.join(model_path, p["fname"]))
            plt.close(fig)
            fout.write(f"## {p['title']}\n\n![](./{p['fname']})\n\n")

    aps.AdditionalPlots.append = staticmethod(append)

    _orig_reg = aps.AdditionalPlots.plots_regression

    def plots_regression(target, predictions):
        plots = []
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions)
        target = pd.Series(np.asarray(target).ravel()).reset_index(drop=True)
        pred = pd.Series(np.asarray(predictions).ravel()).reset_index(drop=True)
        n = min(len(target), 5000)
        idx = np.random.RandomState(7).choice(len(target), n, replace=False) if len(target) > n else np.arange(len(target))
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(target.iloc[idx], pred.iloc[idx], alpha=0.25, s=18)
        lo = min(target.min(), pred.min())
        hi = max(target.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y = x")
        ax.set_xlabel("target")
        ax.set_ylabel("prediction")
        ax.set_title("Target values vs predicted values")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout(pad=3.0)
        plots.append({"title": "Target values vs predicted values", "fname": "prediction_scatter.png", "figure": fig})

        fig, ax1 = plt.subplots(figsize=(10, 7))
        resid = pred - target
        ax1.scatter(pred.iloc[idx], resid.iloc[idx], alpha=0.25, s=18)
        ax1.axhline(0, color="r", linestyle="--", linewidth=1.2)
        ax1.set_xlabel("prediction")
        ax1.set_ylabel("residual")
        ax1.set_title("Predicted values vs residuals")
        ax1.grid(alpha=0.3)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        div = make_axes_locatable(ax1)
        axh = div.append_axes("right", size="18%", pad=0.08)
        axh.hist(resid, bins=40, orientation="horizontal", color="#007cf2", alpha=0.7)
        axh.set_xlabel("count", fontsize=8)
        axh.tick_params(labelsize=7)
        fig.tight_layout(pad=3.0)
        plots.append({"title": "Predicted values vs residuals", "fname": "prediction_residuals.png", "figure": fig})
        return plots

    aps.AdditionalPlots.plots_regression = staticmethod(plots_regression)


def _patch_shap():
    import supervised.utils.shap as shap_mod

    _orig_summary = shap_mod.PlotSHAP.summary
    _orig_dependence = shap_mod.PlotSHAP.dependence
    _orig_is_available = shap_mod.PlotSHAP.is_available
    _orig_get_explainer = shap_mod.PlotSHAP.get_explainer

    EXPLAINABLE = {"Xgboost", "Decision Tree", "Random Forest", "LightGBM",
                   "Extra Trees", "CatBoost", "Linear"}

    def is_available(algorithm, X_train, y_train, ml_task):
        if algorithm.algorithm_short_name not in EXPLAINABLE:
            return False
        if hasattr(X_train, "dtypes"):
            num = (pd.api.types.is_numeric_dtype(X_train[c]) or
                   pd.api.types.is_bool_dtype(X_train[c]) for c in X_train.columns)
            if not all(num):
                return False
        return _orig_is_available(algorithm, X_train, y_train, ml_task)

    def summary(shap_values, X_vald, model_file_path, learner_name, class_names):
        n_feat = X_vald.shape[1] if hasattr(X_vald, "shape") else 10
        fig, ax = plt.subplots(figsize=(max(8, 6), max(4.5, 0.28 * n_feat + 1.5)))
        try:
            _orig_summary(shap_values, X_vald, model_file_path, learner_name, class_names)
        except Exception:
            pass
        try:
            fig.tight_layout()
        except Exception:
            pass

    def dependence(shap_values, X_vald, model_file_path, learner_name, file_postfix=""):
        n_feat = X_vald.shape[1] if hasattr(X_vald, "shape") else 9
        plots_cnt = min(9, n_feat)
        cols = 3 if plots_cnt > 4 else 2
        rows = int(np.ceil(plots_cnt / cols))
        fig = plt.figure(figsize=(6.5 * cols, 4.5 * rows))
        try:
            _orig_dependence(shap_values, X_vald, model_file_path, learner_name, file_postfix)
        except Exception:
            pass

    shap_mod.PlotSHAP.summary = staticmethod(summary)
    shap_mod.PlotSHAP.dependence = staticmethod(dependence)
    shap_mod.PlotSHAP.is_available = staticmethod(is_available)
    shap_mod.PlotSHAP.get_explainer = staticmethod(
        lambda algorithm, X_train: _orig_get_explainer(algorithm, X_train)
        if algorithm.algorithm_short_name in EXPLAINABLE else None)


_APPLIED = False


def apply_plot_fixes():
    global _APPLIED
    if _APPLIED:
        return
    _patch_figure_savefig()
    _patch_learning_curves()
    _patch_leaderboard_plots()
    _patch_automl_plots()
    _patch_additional_metrics()
    _patch_additional_plots()
    _patch_shap()
    _APPLIED = True
