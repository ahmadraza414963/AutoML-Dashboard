# -*- coding: utf-8 -*-
"""All Dash callbacks. Registered by `register_callbacks(app)`."""
import json
import os
import urllib.parse

import pandas as pd
import plotly.graph_objects as go
from dash import Output, Input, State, callback, ctx, html, dcc, no_update, ALL

import dash_bootstrap_components as dbc

from . import layout as L
from .state import get_state
from ..core import session as sc
from ..core import reports as rep
from ..core.runner import RunnerRegistry

STATUS_COLORS = {"finished": "success", "running": "primary", "error": "danger",
                 "stopped": "warning", "paused": "warning", "idle": "secondary",
                 "pending": "secondary", "preparing": "secondary"}


def _empty_fig(title=""):
    return go.Figure(go.Scatter(x=[], y=[])).update_layout(
        template="plotly_white", title=title, height=400)


_HEAT_CACHE = {}


def _cached_heat(kind, key, builder):
    """Recompute expensive heatmaps only when results on disk changed."""
    if key is not None and _HEAT_CACHE.get(kind) == key:
        return no_update
    fig = builder()
    if key is not None:
        _HEAT_CACHE[kind] = key
    return fig


def _summary_cards(bullets):
    return html.Div([
        dbc.ListGroup([
            dbc.ListGroupItem([
                html.I(className=f"bi bi-{icon} me-2 text-{color}"),
                html.Span(text),
            ], className="border-0")
            for icon, text, color in bullets
        ], flush=True, className="summary-list"),
    ])


# --------------------------------------------------------------------------
# control -> config path mapping
# --------------------------------------------------------------------------
CONTROL_CFG = [
    (L.ID_MODE, "value", ("automl", "mode")),
    (L.ID_EVAL_METRIC, "value", ("automl", "eval_metric")),
    (L.ID_VALIDATION, "value", ("automl", "validation_type")),
    (L.ID_CFG_KFOLDS, "value", ("automl", "k_folds")),
    (L.ID_CFG_TRAIN_RATIO, "value", ("automl", "train_ratio")),
    (L.ID_CFG_TOTAL_TIME, "value", ("automl", "total_time_limit")),
    (L.ID_CFG_MODEL_TIME, "value", ("automl", "model_time_limit")),
    (L.ID_CFG_RANDOM_STATE, "value", ("automl", "random_state")),
    (L.ID_CFG_START_RANDOM, "value", ("automl", "start_random_models")),
    (L.ID_CFG_HILL, "value", ("automl", "hill_climbing_steps")),
    (L.ID_CFG_TOP, "value", ("automl", "top_models_to_improve")),
    (L.ID_SW_ENSEMBLE, "value", ("automl", "train_ensemble")),
    (L.ID_SW_STACK, "value", ("automl", "stack_models")),
    (L.ID_SW_BOOST, "value", ("automl", "boost_on_errors")),
    (L.ID_SW_FEAT_SEL, "value", ("automl", "features_selection")),
    (L.ID_SEL_EXPLAIN, "value", ("automl", "explain_level")),
    (L.ID_SW_GPU, "value", ("automl", "use_gpu")),
    (L.ID_IN_NJOBS, "value", ("automl", "n_jobs")),
    (L.ID_FEATURES, "value", ("columns", "features")),
    (L.ID_TARGETS, "value", ("columns", "targets")),
    (L.ID_BUILTIN_ALGS, "value", ("algorithms", "builtin")),
    (L.ID_SKLEARN_ALGS, "value", ("algorithms", "sklearn")),
    (L.ID_SW_GOLDEN, "value", ("feature_engineering", "golden")),
    (L.ID_IN_GOLDEN_COUNT, "value", ("feature_engineering", "golden_count")),
    (L.ID_SW_KMEANS, "value", ("feature_engineering", "kmeans")),
    (L.ID_SW_MIX, "value", ("feature_engineering", "mix_encoding")),
    (L.ID_SW_CHAIN, "value", ("chain", "enabled")),
    (L.ID_SW_SPLIT, "value", ("split", "enabled")),
    (L.ID_CFG_TEST_RATIO, "value", ("split", "test_ratio")),
    (L.ID_CFG_SPLIT_SEED, "value", ("split", "seed")),
]


def _nested_set(cfg, path, value):
    node = cfg
    for key in path[:-1]:
        node = node.setdefault(key, {})
    if value is None:
        node.pop(path[-1], None)
    else:
        node[path[-1]] = value


def _cfg_get(cfg, path):
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return no_update
        node = node[key]
    return node


def _col_options(session):
    st = get_state()
    cols = st.columns()
    if not cols and session is not None:
        df = session.load_data_snapshot()
        if df is not None:
            cols = list(df.columns)
    return [{"label": c, "value": c} for c in cols]


def _sessions_table_rows():
    st = get_state()
    try:
        metas = sc.list_sessions(st.workdir, include_meta=False)
    except Exception:
        metas = []
    rows = []
    for s in metas:
        status = s.get("status") or "idle"
        color = STATUS_COLORS.get(status, "secondary")
        cur = " *" if (st.session and st.session.id == s.get("id")) else ""
        rows.append(html.Tr(
            html.Td([
                dbc.Badge(status, color=color, className="me-2"),
                html.Span(str(s.get("name", "?")) + cur),
            ]),
            id={"type": "session-row", "id": s.get("id")},
            style={"cursor": "pointer"},
        ))
    return dbc.Table(
        html.Tbody(rows or html.Tr(html.Td("No sessions yet", colSpan=2,
                                           className="text-muted"))),
        size="sm", striped=True, hover=True)


def _targets_table(session):
    if session is None:
        return html.Div("No session selected.", className="text-muted small")
    summary = session.summary()
    rows = []
    for t in summary.get("targets", []):
        stt = t.get("status", "pending")
        color = STATUS_COLORS.get(stt, "secondary")
        rows.append(html.Tr([
            html.Td(t["target"]),
            html.Td(dbc.Badge(stt, color=color)),
            html.Td(html.Small(t.get("best_model", "") or "", className="text-muted")),
            html.Td(html.Small(t.get("results_path", ""), className="text-muted")),
        ]))
    return dbc.Table([html.Thead(html.Tr([html.Th("Target"), html.Th("Status"),
                                          html.Th("Best model"), html.Th("Results path")])),
                      html.Tbody(rows or html.Tr(html.Td("No targets configured",
                                                         colSpan=4)))],
                     size="sm", bordered=True)


def _progress_label(session, targets, done):
    n = len(targets)
    if not n:
        return "No targets configured"
    pct = int(round(done / n * 100)) if n else 0
    done_m = total_m = 0
    for t in targets or []:
        if t.get("status") in ("running", "preparing", "fitting"):
            try:
                dm, tm = session.target_model_progress(t.get("target", ""))
                done_m += dm
                total_m += tm
            except Exception:
                pass
    if total_m > 0:
        return f"{done}/{n} targets · {done_m}/{total_m} models — {pct}% overall"
    return f"{done}/{n} targets finished — {pct}%"


def _targets_progress_bars(session, targets):
    """Per-target summary bars + a per-model bar for every planned model."""
    blocks = []
    for t in targets or []:
        stt = t.get("status", "pending")
        name = t.get("target", "?")
        if stt == "finished":
            val, color, striped, animated = 100, "success", False, False
            detail = "done"
        elif stt in ("running", "preparing", "fitting"):
            done_m, total_m = (0, 0)
            try:
                done_m, total_m = session.target_model_progress(name)
            except Exception:
                done_m, total_m = 0, 0
            if total_m > 0:
                val = max(0, min(int(round(done_m / total_m * 100)), 100))
                color, striped, animated = "info", True, True
                detail = f"{done_m}/{total_m} models"
            else:
                val, color, striped, animated = 75, "info", True, True
                detail = "training…"
        else:
            val, color, striped, animated = 0, "secondary", False, False
            detail = "pending"
        block = [
            html.Div([
                html.Small(name, className="fw-semibold"),
                html.Small(f"{stt} · {detail}", className="text-muted float-end"),
            ], className="mb-1"),
            dbc.Progress(value=val, color=color, striped=striped, animated=animated,
                         className="mb-2"),
        ]
        try:
            models = session.target_models_detail(name)
        except Exception:
            models = []
        if models:
            model_rows = []
            for m in models:
                mst = m["status"]
                if mst == "done":
                    mval, mcolor = 100, "success"
                elif mst == "training":
                    mval, mcolor = 50, "info"
                else:
                    mval, mcolor = 0, "secondary"
                model_rows.append(html.Div([
                    html.Small(m["name"], className="me-2"),
                    dbc.Progress(value=mval, color=mcolor,
                                 striped=(mst == "training"),
                                 animated=(mst == "training"),
                                 className="flex-grow-1 align-self-center"),
                ], className="d-flex align-items-center mb-1"))
            block.append(html.Div(model_rows, className="models-progress-scroll"))
        blocks.append(html.Div(block, className="mb-3"))
    if not blocks:
        return html.Div("No targets configured.", className="text-muted small")
    return html.Div(blocks)


def _parse_params(text):
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            continue
        low = v.lower()
        if low in ("true", "false"):
            out[k] = low == "true"
        elif low in ("none", "null"):
            out[k] = None
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def register_callbacks(app):

    # ------------------------------------------------------------------ #
    # 0. persist every setup control into the session config
    # ------------------------------------------------------------------ #
    @app.callback([Output("dummy-out", "children")],
                  [Input(cid, prop) for cid, prop, _ in CONTROL_CFG],
                  prevent_initial_call=True)
    def save_config(*values):
        st = get_state()
        if st.session is None:
            return [None]
        for (_, _, path), value in zip(CONTROL_CFG, values):
            _nested_set(st.session.config, path, value)
        st.session.save()
        return [None]

    # ------------------------------------------------------------------ #
    # 1. workdir
    # ------------------------------------------------------------------ #
    @app.callback(Output(L.ID_MODAL_WORKDIR, "is_open"),
                  Input(L.ID_BTN_WORKDIR, "n_clicks"),
                  State(L.ID_MODAL_WORKDIR, "is_open"))
    def toggle_workdir(n, is_open):
        return True if n else no_update

    @app.callback([Output("workdir-badge", "value"),
                   Output("workdir-badge-top", "children"),
                   Output(L.ID_STORE_SESSION, "data")],
                  Input(L.ID_BTN_WORKDIR_SAVE, "n_clicks"),
                  State(L.ID_INPUT_WORKDIR, "value"),
                  prevent_initial_call=True)
    def apply_workdir(n, path):
        st = get_state()
        st.set_workdir(path)
        st.session = None
        st.set_df(None, None)
        return [path, path, None]

    # ------------------------------------------------------------------ #
    # 2. sessions table + create / load / delete
    # ------------------------------------------------------------------ #
    @app.callback(Output(L.ID_STORE_SESSION, "data", allow_duplicate=True),
                  Input(L.ID_BTN_NEW_SESSION, "n_clicks"),
                  State(L.ID_INPUT_SESSION_NAME, "value"),
                  prevent_initial_call=True)
    def new_session(n, name):
        st = get_state()
        s = sc.create_session(st.workdir, name)
        st.session = s
        return {"id": s.id}

    @app.callback(Output(L.ID_STORE_SESSION, "data", allow_duplicate=True),
                  Input({"type": "session-row", "id": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def select_session_row(_):
        trig = ctx.triggered_id
        if isinstance(trig, dict) and trig.get("id"):
            return {"id": trig.get("id")}
        from dash.exceptions import PreventUpdate
        raise PreventUpdate

    @app.callback(Output(L.ID_STORE_SESSION, "data", allow_duplicate=True),
                  Input(L.ID_BTN_DELETE_SESSION, "n_clicks"),
                  State(L.ID_STORE_SESSION, "data"),
                  prevent_initial_call=True)
    def delete_session(n, current):
        st = get_state()
        if current and current.get("id"):
            sc.delete_session(st.workdir, current["id"])
        if st.session and st.session.id == (current or {}).get("id"):
            st.session = None
        return None

    # ------------------------------------------------------------------ #
    # 3. session selected -> populate every control
    # ------------------------------------------------------------------ #
    @app.callback(
        [Output(cid, prop) for cid, prop, _ in CONTROL_CFG] +
        [Output(L.ID_FEATURES, "options"), Output(L.ID_TARGETS, "options"),
         Output(L.ID_PARAM_ALG_SELECT, "options"),
         Output(L.ID_PARAM_SUMMARY, "children"),
         Output(L.ID_EXTRA_TARGETS, "children")],
        Input(L.ID_STORE_SESSION, "data"),
    )
    def load_session(data):
        st = get_state()
        session = None
        if data and data.get("id"):
            session = sc.load_session(os.path.join(sc.sessions_dir(st.workdir), data["id"]))
        co = _col_options(session)
        if session is None:
            return [no_update] * len(CONTROL_CFG) + [co, co, [], "", ""]
        st.session = session
        cfg = session.config
        controls = [_cfg_get(cfg, p) for _, _, p in CONTROL_CFG]
        algs = cfg.get("algorithms", {}).get("builtin", []) + \
            cfg.get("algorithms", {}).get("sklearn", [])
        alg_opts = [{"label": a, "value": a} for a in algs]
        params = cfg.get("algorithms", {}).get("params", {})
        if params:
            summary = html.Div([html.Small(f"Overrides for {len(params)} algorithm(s):",
                                           className="fw-bold")] +
                               [html.Div(html.Small(f"{a}: {ov}",
                                                    className="text-muted"))
                                for a, ov in params.items()])
        else:
            summary = html.Div()
        n_chain = len(cfg.get("columns", {}).get("targets", []))
        if n_chain > 1:
            extra = html.Span(f"{n_chain} targets - chained multi-output training",
                              className="text-warning")
        elif n_chain == 1:
            extra = html.Span("1 target - plain AutoML", className="text-success")
        else:
            extra = ""
        return controls + [co, co, alg_opts, summary, extra]

    # ------------------------------------------------------------------ #
    # 4. data upload
    # ------------------------------------------------------------------ #
    @app.callback([Output(L.ID_DATA_INFO, "children"),
                   Output(L.ID_SELECT_SHEET, "options"),
                   Output(L.ID_SELECT_SHEET, "value"),
                   Output(L.ID_BTN_LOAD, "disabled")],
                  Input(L.ID_UPLOAD, "contents"),
                  State(L.ID_UPLOAD, "filename"))
    def upload_data(contents, filename):
        if not contents:
            return [no_update] * 4
        from ..core.data_io import parse_contents, excel_sheets
        st = get_state()
        df, err = parse_contents(contents, filename, sheet=None)
        if df is None:
            return [html.Span(f"Error: {err}", className="text-danger"), [], None, True]
        st.set_df(df, filename)
        sheets = excel_sheets(df) if filename.lower().endswith((".xlsx", ".xls")) else []
        info = html.Div([
            html.Div(f"File: {filename}"),
            html.Div(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns"),
        ])
        return [info, [{"label": s, "value": s} for s in sheets],
                sheets[0] if sheets else None, False]

    @app.callback([Output(L.ID_FEATURES, "options", allow_duplicate=True),
                   Output(L.ID_TARGETS, "options", allow_duplicate=True),
                   Output(L.ID_FEATURES, "value", allow_duplicate=True),
                   Output(L.ID_TARGETS, "value", allow_duplicate=True),
                   Output(L.ID_PREVIEW_TABLE, "children")],
                  Input(L.ID_BTN_LOAD, "n_clicks"),
                  State(L.ID_SELECT_SHEET, "value"),
                  prevent_initial_call=True)
    def load_data(n, sheet):
        st = get_state()
        if not n or st.df is None:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        if sheet and st.df_sheet != sheet:
            from ..core.data_io import load_file
            path = None
            if st.session and st.session.config.get("data", {}).get("path"):
                path = st.session.config["data"]["path"]
            if path:
                st.df = load_file(path, sheet=sheet)
                st.df_sheet = sheet
        opts = [{"label": c, "value": c} for c in st.columns()]
        prev = dbc.Table.from_dataframe(st.df.head(100), striped=True, hover=True,
                                        size="sm")
        if st.session is not None:
            st.session.config.setdefault("data", {})["filename"] = st.df_name
            st.session.save()
        return [opts, opts, no_update, no_update, prev]

    # ------------------------------------------------------------------ #
    # 5. parameter overrides editor
    # ------------------------------------------------------------------ #
    @app.callback(Output(L.ID_PARAM_EDITOR, "value"),
                  Input(L.ID_PARAM_ALG_SELECT, "value"),
                  State(L.ID_STORE_SESSION, "data"))
    def edit_param(alg, data):
        st = get_state()
        if st.session is None or not alg:
            return ""
        params = st.session.config.get("algorithms", {}).get("params", {})
        ov = params.get(alg, {})
        return "\n".join(f"{k} = {v}" for k, v in ov.items())

    @app.callback(Output(L.ID_PARAM_SUMMARY, "children", allow_duplicate=True),
                  Input(L.ID_BTN_PARAM_APPLY, "n_clicks"),
                  [State(L.ID_PARAM_ALG_SELECT, "value"),
                   State(L.ID_PARAM_EDITOR, "value")],
                  prevent_initial_call=True)
    def apply_param(n, alg, text):
        st = get_state()
        if not n or st.session is None or not alg:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        ov = _parse_params(text)
        params = st.session.config["algorithms"]["params"]
        if ov:
            params[alg] = ov
        else:
            params.pop(alg, None)
        st.session.save()
        return html.Div(html.Small(f"Overrides for {alg}: {ov}",
                                   className="text-success"))

    @app.callback(Output(L.ID_PARAM_SUMMARY, "children", allow_duplicate=True),
                  Input(L.ID_BTN_PARAM_CLEAR, "n_clicks"),
                  prevent_initial_call=True)
    def clear_params(n):
        st = get_state()
        if not n or st.session is None:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        st.session.config["algorithms"]["params"] = {}
        st.session.save()
        return html.Div(html.Small("All overrides cleared", className="text-muted"))

    # ------------------------------------------------------------------ #
    # 6. run / pause / resume / stop
    # ------------------------------------------------------------------ #
    @app.callback([Output(L.ID_BTN_RUN, "disabled"),
                   Output(L.ID_BTN_STOP, "disabled"),
                   Output(L.ID_BTN_PAUSE, "disabled"),
                   Output(L.ID_BTN_RESUME, "disabled"),
                   Output(L.ID_STATUS_BADGE, "children"),
                   Output(L.ID_STATUS_BADGE, "color")],
                  [Input(L.ID_BTN_RUN, "n_clicks"),
                   Input(L.ID_BTN_STOP, "n_clicks"),
                   Input(L.ID_BTN_PAUSE, "n_clicks"),
                   Input(L.ID_BTN_RESUME, "n_clicks")],
                  State(L.ID_STORE_SESSION, "data"),
                  prevent_initial_call=True)
    def run_pause_resume(run_clicks, stop_clicks, pause_clicks, resume_clicks, data):
        st = get_state()
        trig = ctx.triggered_id
        session = st.session
        if session is None and data and data.get("id"):
            session = sc.load_session(os.path.join(sc.sessions_dir(st.workdir), data["id"]))
            st.session = session
        if session is None:
            return [True, True, True, True, "no session", "secondary"]
        if trig == L.ID_BTN_STOP:
            RunnerRegistry.stop(session.id)
            return [True, True, True, False, "stopping...", "warning"]
        if trig == L.ID_BTN_PAUSE:
            RunnerRegistry.pause(session.id)
            return [True, True, True, False, "pausing...", "warning"]
        if trig == L.ID_BTN_RESUME:
            try:
                if st.df is not None:
                    session._df = st.df.copy()
                else:
                    from ..core.chain import build_working_data
                    build_working_data(session)
                ok = RunnerRegistry.resume(session)
            except Exception as e:
                return [False, True, True, False, f"error: {e}", "danger"]
            if not ok:
                return [True, True, True, False, "already running", "primary"]
            return [True, True, True, False, "resuming...", "primary"]
        try:
            if st.df is not None:
                session._df = st.df.copy()
            else:
                from ..core.chain import build_working_data
                build_working_data(session)
            ok = RunnerRegistry.start(session)
        except Exception as e:
            return [False, True, True, False, f"error: {e}", "danger"]
        if not ok:
            return [True, True, True, False, "already running", "primary"]
        return [True, True, True, False, "running...", "primary"]

    # ------------------------------------------------------------------ #
    # 7. polling: progress, console, leaderboard, collective figures,
    #    correlations, sessions, files
    # ------------------------------------------------------------------ #
    @app.callback(
        [Output(L.ID_STATUS_BADGE, "children", allow_duplicate=True),
         Output(L.ID_STATUS_BADGE, "color", allow_duplicate=True),
         Output("status-badge-top", "children"),
         Output("status-badge-top", "color"),
         Output(L.ID_PROGRESS, "value"),
         Output(L.ID_TARGETS_TABLE, "children", allow_duplicate=True),
         Output(L.ID_CONSOLE, "children"),
         Output(L.ID_SUMMARY_OUT, "children"),
         Output(L.ID_LDB_TABLE, "data"), Output(L.ID_LDB_TABLE, "columns"),
         Output(L.ID_LDB_FIG, "figure"),
         Output(L.ID_PERF_FIG, "figure"), Output(L.ID_BOXPLOT_FIG, "figure"),
         Output(L.ID_TIME_FIG, "figure"),
         Output(L.ID_FEAT_HEAT_FIG, "figure"), Output(L.ID_CORR_HEAT_FIG, "figure"),
         Output(L.ID_MODEL_SELECT, "options"), Output(L.ID_MODEL_SELECT, "value"),
         Output(L.ID_CHAIN_FIG, "figure"), Output(L.ID_CHAIN_TABLE, "data"),
         Output(L.ID_CHAIN_TABLE, "columns"),
         Output(L.ID_TEST_METRICS_TABLE, "data"),
         Output(L.ID_TEST_METRICS_TABLE, "columns"),
         Output(L.ID_FILES_TABLE, "data"), Output(L.ID_FILES_TABLE, "columns"),
         Output(L.ID_TABLE_SESSIONS, "children"),
         Output(L.ID_BTN_RUN, "disabled", allow_duplicate=True),
         Output(L.ID_BTN_STOP, "disabled", allow_duplicate=True),
         Output(L.ID_BTN_PAUSE, "disabled", allow_duplicate=True),
         Output(L.ID_BTN_RESUME, "disabled", allow_duplicate=True),
         Output("report-link", "children"),
         Output(L.ID_CONSOLE_PROGRESS, "value"),
         Output(L.ID_CONSOLE_PROGRESS_LABEL, "children"),
         Output(L.ID_CONSOLE_TARGETS_PROGRESS, "children")],
        Input(L.ID_POLL, "n_intervals"),
        State(L.ID_MODEL_SELECT, "value"),
        prevent_initial_call=True,
    )
    def poll_all(_, model_value):
        st = get_state()
        session = st.session
        if session is None:
            return (["idle", "secondary", "idle", "secondary", 0, _targets_table(None), "",
                     [], [], [], _empty_fig(), _empty_fig(), _empty_fig(), _empty_fig(),
                     _empty_fig(), _empty_fig(), [], None, _empty_fig(), [], [],
                     [], [], [], [], _sessions_table_rows(), True, True, True, True, "",
                     0, "", []])
        summary = session.summary()
        status = summary["status"]
        color = STATUS_COLORS.get(status, "secondary")
        badge = html.Div([status.upper(),
                          html.Small(f" {summary['updated_at']}", className="ms-1 text-muted")])
        targets = summary.get("targets", [])
        done = sum(1 for t in targets if t.get("status") == "finished")
        if targets:
            pct = int((done + 0.5 * sum(1 for t in targets
                                        if t.get("status") in ("running", "preparing", "fitting"))) / len(targets) * 100)
        else:
            pct = 0

        logs = session.get_logs(700)
        ldb_data, ldb_cols, ldb_fig = [], [], _empty_fig()
        recs = []
        ldb_recs = summary.get("leaderboard") or []
        metric = str(session.config.get("automl", {}).get("eval_metric", "rmse"))
        results_path = None
        if ldb_recs:
            df = pd.DataFrame(ldb_recs).dropna(subset=["name"])
            ldb_data = df.to_dict("records")
            ldb_cols = [{"name": c, "id": c} for c in df.columns]
            recs = [{"name": r["name"],
                     "metric": float(r.get("metric_value", r.get("metric", 0))),
                     "time": float(r.get("train_time", r.get("elapsed_time", 0))),
                     "model_type": r.get("model_type", "")} for r in ldb_data]
            ldb_fig = rep.leaderboard_figure(recs, metric)
        else:
            tgt = session.last_finished_target()
            if tgt:
                results_path = session.target_results_path(tgt)
        for t in targets:
            if t.get("status") == "finished" and t.get("results_path"):
                results_path = t.get("results_path")
                break
        perf_fig = rep.performance_scatter_figure(recs, metric) if recs else _empty_fig()
        box_fig = rep.leaderboard_boxplot_figure(recs, metric) if recs else _empty_fig()
        time_fig = rep.training_time_figure(recs) if recs else _empty_fig()
        heat_key = None
        if results_path and os.path.isdir(results_path):
            lb_mtime = os.path.getmtime(os.path.join(results_path, "leaderboard.csv")) \
                if os.path.exists(os.path.join(results_path, "leaderboard.csv")) else 0
            dirs = sorted(os.listdir(results_path))
            heat_key = (results_path, lb_mtime, len(dirs))
        feat_fig = _cached_heat("feat", heat_key,
                                lambda: rep.features_importance_heatmap(results_path)
                                if results_path else _empty_fig())
        corr_fig = _cached_heat("corr", heat_key,
                                lambda: rep.models_correlation_figure(results_path)
                                if results_path else _empty_fig())

        model_opts = [{"label": r["name"], "value": r["name"]} for r in ldb_data]
        model_val = None
        if model_opts:
            model_val = model_value if model_value in [o["value"] for o in model_opts] \
                else model_opts[0]["value"]

        chain_fig, chain_data, chain_cols = _chain_view(session, targets)
        tmt_data, tmt_cols = _test_metrics(session)
        files_data, files_cols = _files_view(session)
        running = status == "running"
        paused = status in ("paused", "stopped")
        from ..core.summarizer import summarize
        summary_children = _summary_cards(summarize(session, summary))
        return (badge, color, status.upper(), color, pct, _targets_table(session), logs,
                summary_children,
                ldb_data, ldb_cols, ldb_fig, perf_fig, box_fig, time_fig,
                feat_fig, corr_fig,
                model_opts, model_val,
                chain_fig, chain_data, chain_cols, tmt_data, tmt_cols,
                files_data, files_cols, _sessions_table_rows(),
                running, not running, not running, paused, _report_link(session),
                pct, _progress_label(session, targets, done),
                _targets_progress_bars(session, targets))

    # ------------------------------------------------------------------ #
    # 8. model detail figures
    # ------------------------------------------------------------------ #
    @app.callback(
        [Output(L.ID_LC_FIG, "figure"), Output(L.ID_PRED_FIG, "figure"),
         Output(L.ID_RESID_FIG, "figure"), Output(L.ID_IMP_FIG, "figure"),
         Output(L.ID_SHAP_FIG, "figure"), Output(L.ID_MODEL_PARAMS_TABLE, "data"),
         Output(L.ID_MODEL_PARAMS_TABLE, "columns")],
        [Input(L.ID_MODEL_SELECT, "value"), Input(L.ID_POLL, "n_intervals")],
    )
    def model_detail(model_name, _):
        st = get_state()
        if st.session is None or not model_name:
            return (_empty_fig(), _empty_fig(), _empty_fig(), _empty_fig(),
                    _empty_fig(), [], [])
        target = st.session.last_finished_target()
        if target is None:
            return (_empty_fig("No results on disk yet"), _empty_fig(), _empty_fig(),
                    _empty_fig(), _empty_fig(), [], [])
        rp = st.session.target_results_path(target)
        mdir = os.path.join(rp, model_name)
        if not os.path.isdir(mdir):
            cands = [d for d in os.listdir(rp)
                     if os.path.isdir(os.path.join(rp, d)) and d.startswith(model_name.split("_", 1)[-1])]
            mdir = os.path.join(rp, cands[0]) if cands else rp

        def _safe(fn, title=None):
            try:
                return fn()
            except Exception:
                return _empty_fig(title)

        lc = _safe(lambda: rep.learning_curves_grid(mdir, title=f"Learning curves — {model_name}"))
        pr = _safe(lambda: _oof_figure(mdir, "pred"))
        rs = _safe(lambda: _oof_figure(mdir, "resid"))
        imp = _safe(lambda: rep.importance_figure(mdir, model_name, False))
        shp = _safe(lambda: rep.importance_figure(mdir, model_name, True))
        try:
            pdata, pcols = _model_params(mdir)
        except Exception:
            pdata, pcols = [], []
        return (lc, pr, rs, imp, shp, pdata, pcols)

    # ------------------------------------------------------------------ #
    # 9. feature engineering preview
    # ------------------------------------------------------------------ #
    @app.callback(Output(L.ID_FE_PREVIEW_OUT, "children"),
                  Input(L.ID_BTN_FE_PREVIEW, "n_clicks"),
                  State(L.ID_STORE_SESSION, "data"),
                  prevent_initial_call=True)
    def fe_preview(n, data):
        st = get_state()
        session = st.session
        if not n or session is None:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        fe_cfg = session.config.get("feature_engineering", {})
        features = session.config.get("columns", {}).get("features", [])
        targets = session.config.get("columns", {}).get("targets", [])
        if not targets or not features:
            return html.Div("Select features and at least one target first.",
                            className="text-warning")
        df = st.df if st.df is not None else session.load_data_snapshot()
        if df is None:
            return html.Div("Load data first.", className="text-warning")
        from ..core.fe_pipeline import run_fe_preview
        os.makedirs(session.fe_report_dir, exist_ok=True)
        res = run_fe_preview(df, features, targets[0], fe_cfg,
                             results_path=session.fe_report_dir)
        chips = [dbc.Badge(f"removed: {c}", color="danger", className="me-1")
                 for c in res["removed_columns"]] + \
                [dbc.Badge(f"added: {c}", color="success", className="me-1")
                 for c in res["added_columns"][:20]]
        golden = html.Div([html.H6("Golden features", className="mt-2"),
                           html.Div(", ".join(res["golden_features"][:50]) or "none",
                                    className="small text-muted")]) \
            if res["golden_features"] else html.Div()
        return html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    html.H6("Feature count"),
                    html.H4(f"{res['n_features_before']} -> {res['n_features_after']}"),
                ], className="border rounded p-2 text-center"), width=3),
                dbc.Col(html.Div([
                    html.H6("Target preprocessing"),
                    html.Small(" + ".join(res["target_preprocessing"]) or "none",
                               className="text-muted"),
                ], className="border rounded p-2 text-center"), width=3),
                dbc.Col(html.Div([
                    html.H6("Task detected"),
                    html.Small(str(res["ml_task"]).replace("_", " "),
                               className="text-muted"),
                ], className="border rounded p-2 text-center"), width=3),
                dbc.Col(html.Div([
                    html.H6("Removed / added"),
                    html.Small(f"{len(res['removed_columns'])} / {len(res['added_columns'])}",
                               className="text-muted"),
                ], className="border rounded p-2 text-center"), width=3),
            ], className="mb-2"),
            html.Div(chips),
            golden,
            dcc.Graph(figure=rep.features_correlation_figure(
                res["X"], list(res["X"].columns)[:30]),
                config={"displayModeBar": False}),
            dbc.Collapse([
                html.H6("Pipeline log", className="mt-2"),
                html.Pre(res["log"][-2500:], className="console-pre small"),
            ], is_open=False),
        ])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _chain_view(session, targets):
    records = []
    for t in targets:
        if t.get("status") == "finished":
            try:
                lb = session.load_leaderboard(t.get("results_path"), 1)
                m = float(lb[0].get("metric_value", lb[0].get("metric", 0)))
            except Exception:
                m = 0.0
            records.append((t["target"], m, 0.0))
    fig = rep.chain_progress_figure(records) if records else _empty_fig()
    data = [{"target": t["target"], "status": t.get("status", "?"),
             "best_model": t.get("best_model", ""),
             "results_path": t.get("results_path", "")} for t in targets]
    cols = [{"name": c, "id": c} for c in
            ["target", "status", "best_model", "results_path"]]
    return fig, data, cols


def _test_metrics(session):
    metrics = session.config.get("results", {}).get("test_metrics")
    if not metrics:
        return [], []
    keys = list(next(iter(metrics.values())).keys())
    data = [{"target": t, **m} for t, m in metrics.items()]
    cols = [{"name": c, "id": c} for c in ["target"] + keys]
    return data, cols


def _files_view(session):
    out = []
    root = session.dir
    for d in [session.results_root, session.models_root, session.data_root, session.dir]:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                out.append({"path": os.path.relpath(p, root),
                            "size_kb": round(os.path.getsize(p) / 1024, 1)})
    return out, [{"name": "path", "id": "path"}, {"name": "size_kb", "id": "size_kb"}]


def _report_link(session):
    report = os.path.join(session.dir, "report.html")
    if os.path.exists(report):
        size = os.path.getsize(report) / 1024
        if size >= 1024:
            label = f"{round(size / 1024, 1)} MB"
        else:
            label = f"{round(size)} KB"
        href = f"/reports/{urllib.parse.quote(session.id)}/report.html"
        return html.A(f"Open generated report.html ({label})", href=href,
                      target="_blank")
    return ""


def _oof_figure(mdir, kind):
    f = os.path.join(mdir, "predictions_out_of_folds.csv")
    if not os.path.exists(f):
        f = os.path.join(mdir, "predictions_validation.csv")
    if os.path.exists(f):
        try:
            df = pd.read_csv(f)
            obs_col = next((c for c in df.columns if c.lower().startswith("target")),
                           df.columns[0])
            pred_col = next((c for c in df.columns if c != obs_col), df.columns[-1])
            obs, pred = df[obs_col], df[pred_col]
            if kind == "pred":
                return rep.predictions_figure(obs, pred, os.path.basename(mdir))
            return rep.residuals_figure(obs, pred)
        except Exception:
            pass
    return _empty_fig()


def _model_params(mdir):
    fw = os.path.join(mdir, "framework.json")
    if not os.path.exists(fw):
        return [], []
    try:
        with open(fw, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        rows = []

        def _add(k, v):
            if isinstance(v, (dict, list)):
                rows.append({"param": k, "value": json.dumps(v, ensure_ascii=False)[:400]})
            else:
                rows.append({"param": k, "value": str(v)})

        for k in ("name", "metric_name", "final_loss", "train_time", "is_stacked", "saved"):
            if k in data:
                _add(k, data[k])
        learners = data.get("learners") or []
        if learners:
            lp = learners[0].get("params") or {}
            for k, v in lp.items():
                _add(f"learner.{k}", v)
        params = data.get("params") or {}
        pp = params.get("preprocessing") or {}
        for cat, spec in pp.items():
            if cat == "columns_preprocessing" and isinstance(spec, dict):
                _add("preprocessing.columns_preprocessing", spec)
            elif cat == "target_preprocessing":
                _add("preprocessing.target", spec)
            else:
                _add(f"preprocessing.{cat}", spec)
        return rows, [{"name": "param", "id": "param"}, {"name": "value", "id": "value"}]
    except Exception:
        return [], []