# -*- coding: utf-8 -*-
"""Natural-language run summaries generated from a session's on-disk state."""
import os


def summarize(session, summary):
    """Return a list of (icon, text, color) summary bullets for the UI."""
    bullets = []
    status = summary.get("status", "idle")
    targets = summary.get("targets", [])
    targets_done = [t for t in targets if t.get("status") == "finished"]
    targets_running = [t for t in targets if t.get("status") in ("running", "fitting", "preparing")]
    n_total = len(targets)

    if n_total == 0:
        return [("info-circle", "No target columns configured yet — go to the Setup tab.", "secondary")]

    if status == "running":
        if targets_running:
            names = ", ".join(t["target"] for t in targets_running)
            bullets.append(("play-circle", f"Training in progress — {len(targets_done)} of {n_total} "
                                          f"targets finished, currently working on: {names}.", "primary"))
        else:
            bullets.append(("play-circle", f"Training started — {len(targets_done)} of {n_total} "
                                           "targets finished so far.", "primary"))
    elif status == "paused":
        bullets.append(("pause-circle", f"Run paused — {len(targets_done)} of {n_total} targets finished. "
                                        "Hit Resume to continue from where it stopped.", "warning"))
    elif status == "stopped":
        bullets.append(("stop-circle", f"Run stopped by user — {len(targets_done)} of {n_total} targets "
                                       "finished. Resume to continue from disk.", "warning"))
    elif status == "error":
        bullets.append(("exclamation-triangle", f"Run ended with an error: {summary.get('last_error', '?')}",
                        "danger"))
    elif status == "finished" or (status == "idle" and targets_done):
        best, mval, mt = _best_target(targets_done, session)
        if best:
            bullets.append(("check-circle", f"All {n_total} target(s) trained. Best model overall: "
                                            f"{best} ({mt}) — press the Model detail tab to explore it.",
                            "success"))
        else:
            bullets.append(("check-circle", f"Training finished for {len(targets_done)} of {n_total} targets.",
                            "success"))

    ldb = summary.get("leaderboard") or []
    if ldb:
        n_models = len(ldb)
        types = sorted({r.get("model_type", "?") for r in ldb})
        bullets.append(("grid-3x3-gap", f"{n_models} models trained across {len(types)} algorithm "
                                        f"families: {', '.join(types)}.", "info"))

    gpu = session.config.get("automl", {}).get("use_gpu", True)
    n_jobs = session.config.get("automl", {}).get("n_jobs", -1)
    if gpu:
        bullets.append(("gpu-card", "GPU acceleration enabled — LightGBM, XGBoost and CatBoost run on "
                                    "the GPU (auto-detected at runtime).", "primary"))
    else:
        bullets.append(("cpu", f"CPU-only training — n_jobs = {n_jobs} (all cores).", "secondary"))

    shap_models = _shap_count(session, targets_done)
    if shap_models:
        bullets.append(("lightning-charge-fill",
                        f"SHAP explanations generated for {len(shap_models)} of {len(targets_done) or len(ldb)} "
                        "trained models — see the SHAP tab in Model detail.", "info"))

    chain = session.config.get("chain", {}).get("enabled", False)
    if chain and n_total > 1:
        bullets.append(("link-45deg", "Multi-output regression chain enabled — each target also uses the "
                                      "out-of-fold predictions of the previous targets.", "info"))

    split = session.config.get("split", {})
    if split.get("enabled"):
        bullets.append(("arrow-right-square", f"Holdout evaluation enabled — {split.get('test_ratio', 0.15)} "
                                              "of rows are held out and scored on the Chain tab.", "info"))

    return bullets or [("info-circle", "No run data yet.", "secondary")]


def _best_target(targets_done, session):
    best_name, best_val, best_mt = None, None, None
    for t in targets_done:
        rp = t.get("results_path")
        try:
            lb = session.load_leaderboard(rp, 1)
        except Exception:
            lb = []
        if not lb:
            continue
        r = lb[0]
        val = float(r.get("metric_value", r.get("metric", 0)))
        mt = str(r.get("model_type", "?"))
        if best_val is None or val < best_val:
            best_name, best_val, best_mt = r.get("name"), val, mt
    return best_name, best_val, best_mt


def _shap_count(session, targets_done):
    import glob
    out = []
    for t in targets_done:
        rp = t.get("results_path")
        if not rp or not os.path.isdir(rp):
            continue
        for mdir in sorted(os.listdir(rp)):
            p = os.path.join(rp, mdir)
            if os.path.isdir(p) and glob.glob(os.path.join(p, "*_shap_importance.csv")):
                out.append(mdir)
    return out
