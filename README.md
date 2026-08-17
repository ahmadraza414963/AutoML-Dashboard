# AutoML Dashboard

A self-contained web UI (Dash / Plotly) for automated machine learning on tabular data.

Load a CSV/Excel file, configure an AutoML run (algorithms, feature
engineering, chained multi-target training), run it, and get leaderboards,
plots, predictions, pickled models, and a downloadable HTML report — all
through a browser.

The package is **fully isolated**: every line of code it needs lives inside
this `dashboard/` folder. The mljar-supervised fork it builds on is vendored
in `vendor/supervised`. There are no imports, paths, or artifacts outside
this folder (see [Isolation](#isolation)).

---

## Quick start

```bat
:: 1. create a virtual environment
python -m venv env

:: 2. install the pinned dependencies
env\Scripts\pip install -r requirements.txt

:: 3. run the dashboard
env\Scripts\python -m dashboard.app
```

Open http://127.0.0.1:8050 in your browser.

> `python -m dashboard.app` must be launched from the directory that contains
> the `dashboard` package (its parent), so that the package is importable.

### Configuration via environment variables

| Variable       | Default      | Description                                   |
| -------------- | ------------ | --------------------------------------------- |
| `DASH_HOST`    | `127.0.0.1`  | Host the web server binds to                  |
| `DASH_PORT`    | `8050`       | Port the web server listens on                |
| `DASH_DEBUG`   | `0`          | Set to `1` for Dash debug mode                |
| `DASH_WORKDIR` | `<dashboard>\workdir` | Where sessions / models / reports are stored |

---

## Usage walkthrough

1. **Workdir** — the storage root for all runs. Defaults to `<dashboard>\workdir`
   (overridable with `DASH_WORKDIR` or the workdir dialog in the UI).
2. **Upload data** — CSV or Excel files; columns are profiled automatically
   (numeric / categorical / datetime).
3. **Create a session** — pick features and targets; targets can be *chained*
   (each target model can reuse the predictions of the previous one as input
   features).
4. **Configure AutoML** — mode (`Compete`, etc.), metric (e.g. `rmse`),
   validation (k-fold / holdout), time limits, and algorithm selection:
   mljar built-ins (LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees,
   KNN, Linear, Neural Network, Baselines) plus **27 extra scikit-learn
   regressors** (Ridge, Lasso, ElasticNet, SVR, Gaussian Process, MLP, PLS,
   Kernel Ridge, …) via `core/sklearn_regressors.py`.
5. **Feature engineering** — golden features, K-means features, mix-encoding
   (mljar pipeline), with a live preview.
6. **Run** — progress is streamed to the UI; runs are **resumable** (a
   re-run of a finished session continues from disk instead of retraining).
7. **Results per target** — leaderboard, learning curves, feature importance,
   training-time bar, correlation heatmaps, OOF predictions.
8. **Export** — pickled `AutoML` objects for later reuse, holdout test
   predictions (CSV), and a self-contained HTML report.

Every run is logged to `logs.txt` inside its session directory.

---

## Project layout

```
dashboard/
├── app.py                  # entry point: create_app() / main()
├── __init__.py             # guarantees the vendored fork wins on sys.path
├── _vendored.py            # inserts <dashboard>/vendor at the front of sys.path
├── requirements.txt        # pinned runtime + test dependencies
├── assets/
│   └── style.css           # UI styling
├── ui/
│   ├── layout.py           # dash-bootstrap page layout
│   ├── callbacks.py        # all Dash callbacks (upload, sessions, run, poll…)
│   └── state.py            # process-global UI state + workdir handling
├── core/
│   ├── session.py          # session model: config, save/load, registry, pickles
│   ├── data_io.py          # file upload parsing & column profiling
│   ├── chain.py            # chained multi-target training runner (run_chain)
│   ├── fe_pipeline.py      # feature-engineering preview
│   ├── sklearn_regressors.py  # extra scikit-learn algorithm registration
│   ├── mlsuper_patch.py    # monkey-patches: disk persistence / resume support
│   ├── plot_fixes.py       # matplotlib dpi/bbox fixes for generated plots
│   ├── accelerate.py       # GPU acceleration (LightGBM GPU trainer, etc.)
│   ├── reports.py          # plotly figures + HTML report generation
│   └── summarizer.py       # natural-language run summary
├── vendor/
│   └── supervised/         # the full mljar-supervised 0.11.5 fork (vendored)
└── tests/
    └── e2e_smoke.py        # 79-check end-to-end validation suite
```

### Session storage layout

```
<workdir>/
└── sessions/
    ├── registry.json           # history of all sessions
    └── <session_id>/
        ├── session.json        # config + status + results summary
        ├── logs.txt            # captured training logs
        ├── data/               # uploaded data snapshots
        ├── results/<target>.automl/   # per-target AutoML artifacts
        ├── models/<target>.pkl # pickled AutoML object
        ├── chain_preds.csv     # out-of-fold chained predictions
        └── test_predictions.csv / fe_report/   # holdout preds / FE report
```

---

## Vendored mljar-supervised fork

`vendor/supervised` is a byte-for-byte copy of the `mlsuper`
(mljar-supervised 0.11.5) fork. The dashboard does **not** import
mljar-supervised from pip/site-packages:

- `dashboard/__init__.py` imports `_vendored.py`, which inserts
  `dashboard/vendor` at position 0 of `sys.path`.
- Every subsequent `import supervised` therefore resolves to
  `dashboard/vendor/supervised` — regardless of what is (or is not)
  installed in site-packages.

Do **not** delete `_vendored.py`/`__init__.py`: importing the package would
fail, or the app could silently pick up a different `supervised` version.

NOTE: if you have previously installed mljar-supervised (`supervised`) into
site-packages, remove it or rename the folder (e.g. `supervised.bak`) so the
vendored copy is the only one in play.

---

## Testing

End-to-end validation (patches, upload parsing, sessions, feature
engineering, chained training of 3 targets, resume/pause semantics, plotly
figures, HTML report, Dash app boot + callbacks):

```bat
env\Scripts\python -m dashboard.tests.e2e_smoke
```

Expected result: `PASS=79 FAIL=0` (run time ~4 minutes locally; the test
snakes a small synthetic dataset end-to-end). Model training inside the test
uses short time limits on purpose.

---

## Dependencies

All runtime and test dependencies (pinned) are listed in
`requirements.txt`. Install them into a clean virtual environment with
`pip install -r requirements.txt` — no other packages are required.

---

## Isolation

Everything the application needs is inside this folder:

- Code: `app.py`, `ui/`, `core/` — no imports outside the package.
- ML framework: the whole `supervised` fork in `vendor/`.
- Data: the default workdir is `<dashboard>/workdir` (internal).
- Runtime writes (sessions, models, reports) stay under the workdir.

The only external requirements are standard pip packages declared in
`requirements.txt` and a Python 3.9+ interpreter.