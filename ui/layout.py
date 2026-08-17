# -*- coding: utf-8 -*-
"""Dash layout (single page). All component IDs are exported as module constants."""
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

from .state import get_state

# --------------------------------------------------------------------------
# IDs
# --------------------------------------------------------------------------
ID_BTN_WORKDIR = "btn-workdir"
ID_MODAL_WORKDIR = "modal-workdir"
ID_INPUT_WORKDIR = "input-workdir"
ID_BTN_WORKDIR_SAVE = "btn-workdir-save"

ID_BTN_NEW_SESSION = "btn-new-session"
ID_INPUT_SESSION_NAME = "input-session-name"
ID_TABLE_SESSIONS = "table-sessions"
ID_BTN_DELETE_SESSION = "btn-delete-session"

ID_UPLOAD = "upload-data"
ID_SELECT_SHEET = "select-sheet"
ID_BTN_LOAD = "btn-load-data"
ID_DATA_INFO = "data-info"
ID_PREVIEW_TABLE = "preview-table"

ID_FEATURES = "sel-features"
ID_TARGETS = "sel-targets"
ID_EXTRA_TARGETS = "extra-targets-info"

ID_EVAL_METRIC = "sel-eval-metric"
ID_VALIDATION = "sel-validation"
ID_CFG_KFOLDS = "in-kfolds"
ID_CFG_TRAIN_RATIO = "in-train-ratio"
ID_MODE = "sel-mode"
ID_CFG_TOTAL_TIME = "in-total-time"
ID_CFG_MODEL_TIME = "in-model-time"
ID_CFG_RANDOM_STATE = "in-random-state"
ID_SW_ENSEMBLE = "sw-ensemble"
ID_SW_STACK = "sw-stack"
ID_SW_BOOST = "sw-boost"
ID_SW_FEAT_SEL = "sw-feat-sel"
ID_SEL_EXPLAIN = "sel-explain"
ID_SW_GPU = "sw-gpu"
ID_IN_NJOBS = "in-njobs"
ID_CFG_START_RANDOM = "in-start-random"
ID_CFG_HILL = "in-hill"
ID_CFG_TOP = "in-top"

ID_BUILTIN_ALGS = "sel-builtin"
ID_SKLEARN_ALGS = "sel-sklearn"
ID_PARAM_ALG_SELECT = "sel-param-alg"
ID_PARAM_EDITOR = "ta-param-editor"
ID_BTN_PARAM_APPLY = "btn-param-apply"
ID_PARAM_SUMMARY = "param-summary"
ID_BTN_PARAM_CLEAR = "btn-param-clear"

ID_SW_GOLDEN = "sw-golden"
ID_IN_GOLDEN_COUNT = "in-golden-count"
ID_SW_KMEANS = "sw-kmeans"
ID_SW_MIX = "sw-mix"
ID_BTN_FE_PREVIEW = "btn-fe-preview"
ID_FE_PREVIEW_OUT = "fe-preview-out"

ID_SW_CHAIN = "sw-chain"
ID_SW_SPLIT = "sw-split"
ID_CFG_TEST_RATIO = "in-test-ratio"
ID_CFG_SPLIT_SEED = "in-split-seed"

ID_BTN_RUN = "btn-run"
ID_BTN_STOP = "btn-stop"
ID_BTN_PAUSE = "btn-pause"
ID_BTN_RESUME = "btn-resume"
ID_STATUS_BADGE = "status-badge"
ID_PROGRESS = "progress-overall"
ID_TARGETS_TABLE = "targets-table"
ID_CONSOLE = "console"
ID_CONSOLE_PROGRESS = "console-progress"
ID_CONSOLE_PROGRESS_LABEL = "console-progress-label"
ID_CONSOLE_TARGETS_PROGRESS = "console-targets-progress"
ID_SUMMARY_OUT = "summary-out"

ID_TABS_MAIN = "tabs-main"
ID_TABS_RESULTS = "tabs-results"
ID_LDB_TABLE = "ldb-table"
ID_LDB_FIG = "ldb-fig"
ID_PERF_FIG = "perf-fig"
ID_BOXPLOT_FIG = "boxplot-fig"
ID_TIME_FIG = "time-fig"
ID_FEAT_HEAT_FIG = "feat-heat-fig"
ID_CORR_HEAT_FIG = "corr-heat-fig"
ID_MODEL_SELECT = "sel-model"
ID_LC_FIG = "lc-fig"
ID_PRED_FIG = "pred-fig"
ID_RESID_FIG = "resid-fig"
ID_IMP_FIG = "imp-fig"
ID_SHAP_FIG = "shap-fig"
ID_MODEL_PARAMS_TABLE = "model-params-table"
ID_CHAIN_FIG = "chain-fig"
ID_CHAIN_TABLE = "chain-table"
ID_TEST_METRICS_TABLE = "test-metrics-table"
ID_FILES_TABLE = "files-table"

ID_POLL = "poll-interval"
ID_STORE_SESSION = "store-session"


def _card(title, children, id_=None, class_name="mb-3", icon="gear"):
    return dbc.Card(
        [dbc.CardHeader(html.B([html.I(className=f"bi bi-{icon} me-1"), title]))] +
        ([html.Div(children, className="card-body")]),
        className="shadow-sm " + (class_name or ""),
    )


# --------------------------------------------------------------------------
# Data tab
# --------------------------------------------------------------------------
def data_tab():
    return html.Div([
        dbc.Row([
            dbc.Col(_card("Upload data", [
                dcc.Upload(id=ID_UPLOAD, className="upload-box",
                           children=html.Div(["Drag & drop or ", html.A("browse", href="#"),
                                              " CSV / Excel file"])),
                html.Div(id=ID_DATA_INFO, className="small text-muted mt-2"),
                html.Div([
                    dbc.Select(id=ID_SELECT_SHEET, options=[], className="mt-2"),
                    dbc.Button("Load & inspect", id=ID_BTN_LOAD, color="primary",
                               size="sm", className="mt-2", disabled=True),
                ]),
            ], icon="file-earmark-arrow-up"), width=4),
            dbc.Col(_card("Data preview", [
                html.Div("First 100 rows — scrollable", className="small text-muted mb-2"),
                html.Div(id=ID_PREVIEW_TABLE, className="preview-scroll"),
            ], icon="table"), width=8),
        ]),
        _card("What happens next", [
            html.Small("1. Upload a CSV / Excel file → 2. inspect the preview → "
                       "3. go to the Setup tab and pick feature / target columns and "
                       "algorithms → 4. press Start training.", className="text-muted"),
        ], icon="info-circle"),
    ])


# --------------------------------------------------------------------------
# Session tab
# --------------------------------------------------------------------------
def session_tab():
    st = get_state()
    return html.Div([
        _card("Session management", [
            dbc.Row([
                dbc.Col([
                    dbc.Label("New session name", className="small"),
                    dbc.InputGroup([
                        dbc.Input(id=ID_INPUT_SESSION_NAME, placeholder="Session name",
                                  value=f"run-{__import__('time').strftime('%m%d-%H%M')}"),
                        dbc.Button("Create", id=ID_BTN_NEW_SESSION, color="primary"),
                    ]),
                ], width=5),
                dbc.Col([
                    dbc.Label("Working directory", className="small"),
                    dbc.InputGroup([
                        dbc.Input(id="workdir-badge", readonly=True),
                        dbc.Button("Change", id=ID_BTN_WORKDIR, color="secondary",
                                   outline=True),
                    ]),
                ], width=7),
            ]),
            html.Hr(),
            html.Div("Click a row to load the session — its configuration and "
                     "results will appear in Setup / Results.", className="small text-muted mb-2"),
            html.Div(id=ID_TABLE_SESSIONS, className="session-list"),
            dbc.Button("Delete selected session", id=ID_BTN_DELETE_SESSION, color="danger",
                       outline=True, size="sm", className="mt-2"),
        ], icon="kanban"),
        _card("Tips", [
            html.Ul([
                html.Li(html.Small("Sessions store data, config, models and results "
                                   "in the working directory.")),
                html.Li(html.Small("A session can be interrupted (Pause / Stop) and "
                                   "Resumed later — finished targets are reused.")),
                html.Li(html.Small("Chained multi-target sessions train targets in the "
                                   "selected order and chain predictions between them.")),
            ], className="mb-0"),
        ], icon="lightbulb"),
    ])


# --------------------------------------------------------------------------
# Setup tab
# --------------------------------------------------------------------------
def setup_tab():
    return html.Div([
        _card("Data & columns", [
            dbc.Row([
                dbc.Col([
                    dbc.Label("Features (X)"),
                    dcc.Dropdown(id=ID_FEATURES, multi=True, placeholder="Select feature columns"),
                ], width=6),
                dbc.Col([
                    dbc.Label("Targets (Y) — order matters for chains"),
                    dcc.Dropdown(id=ID_TARGETS, multi=True,
                                 placeholder="Select one or more target columns"),
                    html.Div(id=ID_EXTRA_TARGETS, className="small text-muted mt-1"),
                ], width=6),
            ]),
        ], id_="card-cols", icon="columns-gap"),
        _card("AutoML settings", [
            dbc.Row([
                dbc.Col(dbc.Label("Mode"), width=2),
                dbc.Col(dbc.Select(id=ID_MODE,
                                   options=[{"label": m, "value": m}
                                            for m in ["Explain", "Perform", "Compete", "Optuna"]],
                                   value="Compete"), width=3),
                dbc.Col(dbc.Label("Eval metric", className="text-end"), width=2),
                dbc.Col(dbc.Select(id=ID_EVAL_METRIC,
                                   options=[{"label": m.upper(), "value": m}
                                            for m in ["rmse", "mse", "mae", "r2", "mape",
                                                      "spearman", "pearson"]],
                                   value="rmse"), width=3),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Label("Validation"), width=2),
                dbc.Col(dbc.Select(id=ID_VALIDATION,
                                   options=[{"label": "k-fold CV", "value": "kfold"},
                                            {"label": "train/test split", "value": "split"}],
                                   value="kfold"), width=3),
                dbc.Col(dbc.Label("k folds", className="text-end"), width=2),
                dbc.Col(dbc.Input(id=ID_CFG_KFOLDS, type="number", min=2, max=20, value=5), width=2),
                dbc.Col(dbc.Label("train ratio", className="text-end"), width=1),
                dbc.Col(dbc.Input(id=ID_CFG_TRAIN_RATIO, type="number", min=0.5, max=0.95,
                                  step=0.05, value=0.75), width=2),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Label("Total time limit (s)", className="small"), width=3),
                dbc.Col(dbc.Input(id=ID_CFG_TOTAL_TIME, type="number", value=3600), width=3),
                dbc.Col(dbc.Label("Model time limit (s)", className="small text-end"), width=3),
                dbc.Col(dbc.Input(id=ID_CFG_MODEL_TIME, type="number", placeholder="unlimited"), width=3),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Label("Random state", className="small"), width=3),
                dbc.Col(dbc.Input(id=ID_CFG_RANDOM_STATE, type="number", value=1234), width=3),
                dbc.Col(dbc.Label("Tuner: random models", className="small text-end"), width=3),
                dbc.Col(dbc.Input(id=ID_CFG_START_RANDOM, type="number", value=10), width=3),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Label("Hill climbing steps", className="small"), width=3),
                dbc.Col(dbc.Input(id=ID_CFG_HILL, type="number", value=3), width=3),
                dbc.Col(dbc.Label("Models to improve", className="small text-end"), width=3),
                dbc.Col(dbc.Input(id=ID_CFG_TOP, type="number", value=3), width=3),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Switch(id=ID_SW_ENSEMBLE, label="Train ensemble", value=True), width=3),
                dbc.Col(dbc.Switch(id=ID_SW_STACK, label="Stack models", value=True), width=3),
                dbc.Col(dbc.Switch(id=ID_SW_BOOST, label="Boost on errors", value=True), width=3),
                dbc.Col(dbc.Switch(id=ID_SW_FEAT_SEL, label="Feature selection", value=True), width=3),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Label("Explain level", className="small"), width=3),
                dbc.Col(dbc.Select(id=ID_SEL_EXPLAIN,
                                   options=[{"label": "0 — none", "value": 0},
                                            {"label": "1 — importance", "value": 1},
                                            {"label": "2 — + SHAP", "value": 2}],
                                   value=2), width=3),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Switch(id=ID_SW_GPU, label="Use GPU (auto-detect)", value=True), width=4),
                dbc.Col(dbc.Label("n_jobs (-1 = all cores)", className="small text-end"), width=4),
                dbc.Col(dbc.Input(id=ID_IN_NJOBS, type="number", min=-1, step=1, value=-1), width=2),
            ]),
        ], id_="card-automl", icon="sliders"),
        _card("Algorithms & hyper-parameters", [
            dbc.Row([
                dbc.Col([
                    dbc.Label("mlsuper built-in algorithms"),
                    dcc.Dropdown(id=ID_BUILTIN_ALGS, multi=True,
                                 options=[{"label": a, "value": a} for a in [
                                     "Baseline", "Linear", "Decision Tree", "Random Forest",
                                     "Extra Trees", "LightGBM", "Xgboost", "CatBoost",
                                     "Neural Network", "Nearest Neighbors"]]),
                ], width=6),
                dbc.Col([
                    dbc.Label("scikit-learn regressors (extras)"),
                    dcc.Dropdown(id=ID_SKLEARN_ALGS, multi=True,
                                 options=[{"label": a, "value": a}
                                          for a in __import__("dashboard.core.sklearn_regressors", fromlist=["x"]).SKLEARN_REGRESSORS]),
                ], width=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Per-algorithm parameter overrides", className="small"),
                    dbc.Select(id=ID_PARAM_ALG_SELECT, options=[], placeholder="Pick an algorithm"),
                    dcc.Textarea(id=ID_PARAM_EDITOR, rows=4, className="form-control mt-2",
                                 placeholder="param = value   (one per line, e.g.\nmax_depth = 12\nlearning_rate = 0.05\nn_estimators = 200)"),
                    html.Div([
                        dbc.Button("Apply override", id=ID_BTN_PARAM_APPLY, size="sm",
                                   color="primary", className="mt-2"),
                        dbc.Button("Clear all overrides", id=ID_BTN_PARAM_CLEAR, size="sm",
                                   color="danger", outline=True, className="mt-2 ms-1"),
                    ]),
                    html.Div(id=ID_PARAM_SUMMARY, className="small mt-2"),
                ], width=12),
            ]),
        ], id_="card-algs", icon="cpu"),
        _card("Feature engineering (mlsuper pipeline)", [
            dbc.Row([
                dbc.Col(dbc.Switch(id=ID_SW_GOLDEN, label="Golden features", value=True), width=3),
                dbc.Col(dbc.Input(id=ID_IN_GOLDEN_COUNT, type="number", min=1, step=1,
                                  placeholder="count (auto)", value=None), width=2),
                dbc.Col(dbc.Switch(id=ID_SW_KMEANS, label="K-Means features", value=True), width=3),
                dbc.Col(dbc.Switch(id=ID_SW_MIX, label="Mix encoding (categorical)", value=True), width=3),
            ], className="mb-2"),
            dbc.Button("Preview feature engineering on data", id=ID_BTN_FE_PREVIEW,
                       color="info", outline=True, size="sm"),
            html.Div(id=ID_FE_PREVIEW_OUT, className="mt-3"),
        ], id_="card-fe", icon="wand"),
        _card("Multi-output chain & holdout", [
            dbc.Row([
                dbc.Col(dbc.Switch(id=ID_SW_CHAIN,
                                   label="Enable regression chain (multi-output)",
                                   value=False), width=4),
                dbc.Col(dbc.Switch(id=ID_SW_SPLIT,
                                   label="Holdout test set evaluation", value=False), width=4),
                dbc.Col(dbc.Label("Test ratio", className="small"), width=2),
                dbc.Col(dbc.Input(id=ID_CFG_TEST_RATIO, type="number", min=0.05, max=0.5,
                                  step=0.05, value=0.15), width=2),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Label("Split seed", className="small"), width=3),
                dbc.Col(dbc.Input(id=ID_CFG_SPLIT_SEED, type="number", value=123), width=3),
            ]),
        ], id_="card-chain", icon="diagram-3"),
        _card("Run", [
            dbc.Row([
                dbc.Col([
                    dbc.Button("▶ Start training", id=ID_BTN_RUN, color="success",
                               size="lg", className="me-2"),
                    dbc.Button("⏸ Pause", id=ID_BTN_PAUSE, color="warning", size="lg",
                               outline=True, className="me-2", disabled=True),
                    dbc.Button("▶ Resume", id=ID_BTN_RESUME, color="info", size="lg",
                               outline=True, className="me-2", disabled=True),
                    dbc.Button("■ Stop", id=ID_BTN_STOP, color="danger", size="lg",
                               outline=True, disabled=True),
                    dbc.Badge("idle", id=ID_STATUS_BADGE, color="secondary",
                              className="ms-3 fs-6"),
                ]),
            ]),
            dbc.Progress(id=ID_PROGRESS, value=0, striped=True, className="mt-3 mb-1"),
            html.Div(id=ID_TARGETS_TABLE, className="mt-2"),
            html.Div("The verbose training log and a live natural-language summary are "
                     "in Results → Console.", className="small text-muted mt-2"),
        ], id_="card-run", icon="play-circle"),
    ])


# --------------------------------------------------------------------------
# Results tab
# --------------------------------------------------------------------------
def results_tab():
    return dbc.Tabs(id=ID_TABS_RESULTS, active_tab="tab-ldb", children=[
        dbc.Tab(label="Leaderboard", tab_id="tab-ldb", children=[
            _card("Leaderboard & collective performance (live)", [
                dash_table.DataTable(id=ID_LDB_TABLE, style_table={"overflowX": "auto"},
                                     style_cell={"fontSize": 12},
                                     style_header={"fontWeight": "bold"},
                                     page_size=10, sort_action="native"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id=ID_LDB_FIG), width=8),
                    dbc.Col(dcc.Graph(id=ID_TIME_FIG), width=4),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id=ID_PERF_FIG), width=6),
                    dbc.Col(dcc.Graph(id=ID_BOXPLOT_FIG), width=6),
                ]),
            ], icon="bar-chart"),
        ]),
        dbc.Tab(label="Model detail", tab_id="tab-model", children=[
            _card("Model detail", [
                dbc.Row([
                    dbc.Col(dbc.Label("Select model"), width=2),
                    dbc.Col(dcc.Dropdown(id=ID_MODEL_SELECT), width=6),
                ], className="mb-2"),
                dbc.Tabs(id="sub-model-tabs", active_tab="sub-lc", children=[
                    dbc.Tab(label="Learning curve", tab_id="sub-lc", children=[dcc.Graph(id=ID_LC_FIG)]),
                    dbc.Tab(label="Predictions", tab_id="sub-pred", children=[dcc.Graph(id=ID_PRED_FIG)]),
                    dbc.Tab(label="Residuals", tab_id="sub-resid", children=[dcc.Graph(id=ID_RESID_FIG)]),
                    dbc.Tab(label="Permutation importance", tab_id="sub-imp", children=[dcc.Graph(id=ID_IMP_FIG)]),
                    dbc.Tab(label="SHAP importance", tab_id="sub-shap", children=[dcc.Graph(id=ID_SHAP_FIG)]),
                    dbc.Tab(label="Params", tab_id="sub-params", children=[
                        dash_table.DataTable(id=ID_MODEL_PARAMS_TABLE,
                                             style_table={"overflowX": "auto"},
                                             page_size=15)]),
                ]),
            ], icon="search"),
        ]),
        dbc.Tab(label="Correlations", tab_id="tab-corr", children=[
            dbc.Row([
                dbc.Col(_card("Model prediction correlation (spearman, live)", [
                    dcc.Graph(id=ID_CORR_HEAT_FIG),
                    html.Small("How similar the out-of-fold predictions of each pair of "
                               "models are — highly correlated models can be seen as redundant.",
                               className="text-muted"),
                ], icon="grid-1x2"), width=6),
                dbc.Col(_card("Feature importance across models (live)", [
                    dcc.Graph(id=ID_FEAT_HEAT_FIG),
                    html.Small("Permutation importance (mean over folds) of every feature "
                               "for every trained model.",
                               className="text-muted"),
                ], icon="layout-3-column"), width=6),
            ]),
        ]),
        dbc.Tab(label="Chain & holdout", tab_id="tab-chain", children=[
            _card("Multi-target overview", [
                dcc.Graph(id=ID_CHAIN_FIG),
                dash_table.DataTable(id=ID_CHAIN_TABLE, style_table={"overflowX": "auto"},
                                     page_size=10, style_cell={"fontSize": 12}),
                html.H6("Holdout test metrics", className="mt-3"),
                dash_table.DataTable(id=ID_TEST_METRICS_TABLE,
                                     style_table={"overflowX": "auto"},
                                     style_cell={"fontSize": 12}),
            ], icon="diagram-3"),
        ]),
        dbc.Tab(label="Console", tab_id="tab-console", children=[
            _card("Training progress (live)", [
                dbc.Progress(id=ID_CONSOLE_PROGRESS, value=0, striped=True,
                             className="mb-1"),
                html.Small(id=ID_CONSOLE_PROGRESS_LABEL, className="text-muted"),
                html.Div(id=ID_CONSOLE_TARGETS_PROGRESS, className="mt-3"),
            ], icon="speedometer"),
            _card("Run summary (natural language, live)", [
                html.Div(id=ID_SUMMARY_OUT),
            ], icon="chat-square-text"),
            _card("Verbosity log (fixed height, scrollable)", [
                html.Pre(id=ID_CONSOLE, className="console-fixed",
                         children="Verbosity log will appear here…"),
            ], icon="terminal"),
        ]),
        dbc.Tab(label="Files", tab_id="tab-files", children=[
            _card("Session files", [
                dash_table.DataTable(id=ID_FILES_TABLE, style_table={"overflowX": "auto"},
                                     page_size=15, style_cell={"fontSize": 12}),
                html.Div(id="report-link", className="mt-2"),
            ], icon="folder2-open"),
        ]),
    ])


# --------------------------------------------------------------------------
# main layout
# --------------------------------------------------------------------------
def layout():
    return dbc.Container(fluid=True, children=[
        dcc.Store(id=ID_STORE_SESSION),
        html.Div([
            html.H4([html.I(className="bi bi-speedometer2 me-2"), "AutoML Dashboard",
                     html.Small(" — mlsuper / mljar-supervised + sklearn regressors",
                                className="text-muted fs-6 ms-2")],
                    className="d-inline-block"),
            html.Div([
                dbc.Badge([html.I(className="bi bi-lightning-charge me-1"),
                           html.Span(id="status-badge-top")],
                          color="secondary", className="me-2"),
                dbc.Badge([html.I(className="bi bi-folder2 me-1"),
                           html.Span(id="workdir-badge-top")], color="dark"),
            ], className="float-end pt-2"),
        ], className="app-header mb-2"),
        dcc.Tabs(id=ID_TABS_MAIN, value="tab-data", children=[
            dcc.Tab(label="Data", value="tab-data", children=data_tab()),
            dcc.Tab(label="Session", value="tab-session", children=session_tab()),
            dcc.Tab(label="Setup", value="tab-setup", children=setup_tab()),
            dcc.Tab(label="Results", value="tab-results", children=results_tab()),
        ]),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Working directory")),
            dbc.ModalBody([
                dbc.Label("Absolute path where sessions / data / results are stored:"),
                dbc.Input(id=ID_INPUT_WORKDIR, value=get_state().workdir),
            ]),
            dbc.ModalFooter(dbc.Button("Save", id=ID_BTN_WORKDIR_SAVE, color="primary")),
        ], id=ID_MODAL_WORKDIR),
        dcc.Interval(id=ID_POLL, interval=2000),
        html.Div(id="dummy-out", style={"display": "none"}),
    ], style={"minHeight": "100vh", "background": "#f4f6f9", "maxWidth": "1600px"})
