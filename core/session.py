# -*- coding: utf-8 -*-
"""
Session management: every training run is a resumable, reproducible session.

Layout of a session directory:

    <workdir>/sessions/
        registry.json               # history of all sessions
        <session_id>/
            session.json            # config + status + results summary
            logs.txt                # captured stdout / stderr (full verbosity)
            data/                   # uploaded / prepared data snapshots
            results/<target>.automl/# per-target AutoML results_path
            models/<target>.pkl     # pickled AutoML object (fast reuse)
            chain_preds.csv         # row-aligned OOF chain predictions
            fe_report/              # feature-engineering report artifacts
"""
import copy
import json
import os
import pickle
import shutil
import threading
import time
import uuid

import pandas as pd

REGISTRY_FNAME = "registry.json"
SESSION_FNAME = "session.json"
LOG_FNAME = "logs.txt"

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_FINISHED = "finished"
STATUS_ERROR = "error"
STATUS_STOPPED = "stopped"
STATUS_PAUSED = "paused"

SECONDS = "seconds"


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sanitize_name(name):
    return "".join(c if (c.isalnum() or c in "_- ") else "_" for c in str(name)).strip().replace(" ", "_")


def _default_config():
    return {
        "automl": {
            "mode": "Compete",
            "eval_metric": "rmse",
            "validation_type": "kfold",
            "k_folds": 5,
            "shuffle": True,
            "train_ratio": 0.75,
            "total_time_limit": 3600,
            "model_time_limit": None,
            "start_random_models": 10,
            "hill_climbing_steps": 3,
            "top_models_to_improve": 3,
            "train_ensemble": True,
            "stack_models": True,
            "explain_level": 2,
            "boost_on_errors": True,
            "random_state": 1234,
            "n_jobs": -1,
            "features_selection": True,
        },
        "columns": {"features": [], "targets": []},
        "split": {"enabled": False, "test_ratio": 0.15, "seed": 123},
        "algorithms": {
            "builtin": [],
            "sklearn": [],
            "params": {},
        },
        "feature_engineering": {
            "golden": True,
            "golden_count": None,
            "kmeans": True,
            "mix_encoding": True,
        },
        "chain": {"enabled": False},
    }


class Session:
    """One training experiment (single or chained multi-output)."""

    def __init__(self, session_dir, config=None):
        self.dir = session_dir
        self.config = _default_config()
        if config:
            self._deep_merge(self.config, config)
        self.status = STATUS_IDLE
        self.id = os.path.basename(session_dir)
        self.name = self.id
        self.created_at = now_str()
        self.updated_at = now_str()
        self.last_error = None
        self.automl = {}            # target -> AutoML object (in-memory)
        self.chain_preds = {}       # target -> pd.Series of OOF predictions
        self.log_buffer = []        # in-memory log tail
        self._log_lock = threading.Lock()
        self._data_lock = threading.Lock()
        self._df = None             # working dataframe (features + targets)
        self._df_meta = {}          # column dtypes / stats cache
        self.running = False
        self.stop_requested = False
        self.pause_requested = False
        self.resume_mode = False
        self.thread = None
        self._finished_targets = set()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _deep_merge(base, override):
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                Session._deep_merge(base[k], v)
            else:
                base[k] = copy.deepcopy(v)

    def set_config(self, config):
        self.config = _default_config()
        if config:
            self._deep_merge(self.config, config)
        self._sanitize_config()

    def _sanitize_config(self):
        cfg = self.config
        cfg.setdefault("automl", {})
        cfg.setdefault("columns", {"features": [], "targets": []})
        cfg.setdefault("algorithms", {"builtin": [], "sklearn": [], "params": {}})
        cfg.setdefault("split", {"enabled": False, "test_ratio": 0.15})
        cfg.setdefault("feature_engineering", {})
        cfg.setdefault("chain", {"enabled": False})

    # ------------------------------------------------------------------ #
    @property
    def results_root(self):
        return os.path.join(self.dir, "results")

    @property
    def models_root(self):
        return os.path.join(self.dir, "models")

    @property
    def data_root(self):
        return os.path.join(self.dir, "data")

    @property
    def fe_report_dir(self):
        return os.path.join(self.dir, "fe_report")

    @property
    def logs_path(self):
        return os.path.join(self.dir, LOG_FNAME)

    @property
    def config_path(self):
        return os.path.join(self.dir, SESSION_FNAME)

    @property
    def chain_preds_path(self):
        return os.path.join(self.dir, "chain_preds.csv")

    def target_results_path(self, target):
        return os.path.join(self.results_root, sanitize_name(target) + ".automl")

    def target_pickle_path(self, target):
        return os.path.join(self.models_root, sanitize_name(target) + ".pkl")

    # ------------------------------------------------------------------ #
    def log(self, message):
        ts = time.strftime("%H:%M:%S")
        with self._log_lock:
            self.log_buffer.append(f"[{ts}] {message}")
            if len(self.log_buffer) > 4000:
                self.log_buffer = self.log_buffer[-2000:]
        try:
            with open(self.logs_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(f"[{ts}] {message}\n")
        except Exception:
            pass

    def get_logs(self, tail=800):
        with self._log_lock:
            return "\n".join(self.log_buffer[-tail:])

    # ------------------------------------------------------------------ #
    def save(self):
        self.updated_at = now_str()
        payload = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "last_error": self.last_error,
            "config": self.config,
            "targets_finished": sorted(self._finished_targets),
            "automl": {  # lightweight run metadata for the UI
                "mode": self.config.get("automl", {}).get("mode"),
                "eval_metric": self.config.get("automl", {}).get("eval_metric"),
            },
        }
        os.makedirs(self.dir, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def to_meta(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "config": self.config,
        }

    # ------------------------------------------------------------------ #
    def summary(self):
        """Snapshot used by the polling callbacks."""
        per_target = []
        for t in self.config.get("columns", {}).get("targets", []):
            info = {"target": t, "status": "pending", "results_path": self.target_results_path(t)}
            st = self._target_status_from_disk(t)
            info.update(st)
            per_target.append(info)
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "last_error": self.last_error,
            "updated_at": self.updated_at,
            "targets": per_target,
            "leaderboard": self.load_leaderboard(limit=40),
            "config": self.config,
        }

    def _target_status_from_disk(self, target):
        rp = self.target_results_path(target)
        pjson = os.path.join(rp, "params.json")
        prog = os.path.join(rp, "progress.json")
        if not os.path.exists(rp):
            return {"status": "pending"}
        if not os.path.exists(pjson):
            return {"status": "preparing"}
        try:
            with open(pjson, "r", encoding="utf-8") as f:
                p = json.load(f)
            fit_level = p.get("fit_level")
            # the fork writes README.md only when the AutoML run truly ended,
            # so a run whose process died or was interrupted right after fit
            # still shows as finished instead of "running" forever
            ended = os.path.exists(os.path.join(rp, "README.md"))
            if fit_level == "finished" or ended:
                return {"status": "finished", "best_model": p.get("best_model"),
                        "results_path": rp, "leaderboard": self.load_leaderboard(rp, 20)}
            step = "fitting"
            if os.path.exists(prog):
                with open(prog, "r", encoding="utf-8") as f:
                    pr = json.load(f)
                step = pr.get("fit_level", "fitting")
            return {"status": "running", "step": step, "results_path": rp}
        except Exception:
            return {"status": "running", "results_path": rp}

    def target_model_progress(self, target):
        """Live per-model progress for one target: (completed_models,
        planned_models). Completed = model dirs that reached the fork's
        'status.txt' ('ALL OK!') marker; planned = sum of generated params in
        progress.json's all_params (falls back to completed/0 if unknown)."""
        rp = self.target_results_path(target)
        total = 0
        pj = os.path.join(rp, "progress.json")
        if os.path.exists(pj):
            try:
                with open(pj, "r", encoding="utf-8") as f:
                    p = json.load(f)
                total = sum(len(v or []) for v in (p.get("all_params") or {}).values())
            except Exception:
                total = 0
        completed = 0
        if os.path.isdir(rp):
            try:
                for name in os.listdir(rp):
                    d = os.path.join(rp, name)
                    if name.startswith("_") or not os.path.isdir(d):
                        continue
                    if os.path.exists(os.path.join(d, "status.txt")):
                        completed += 1
            except Exception:
                completed = 0
        return completed, total

    def target_models_detail(self, target):
        """Per-model status for one target: [{name, status}] where status is
        'done' (status.txt marker), 'training' (dir exists, no marker) or
        'pending'. The model list is the fork's planned params from
        progress.json (all_params), so it grows as steps are reached."""
        rp = self.target_results_path(target)
        planned = []
        pj = os.path.join(rp, "progress.json")
        if os.path.exists(pj):
            try:
                with open(pj, "r", encoding="utf-8") as f:
                    p = json.load(f)
                for step, params in (p.get("all_params") or {}).items():
                    for par in params or []:
                        name = par.get("name")
                        if name:
                            planned.append(name)
            except Exception:
                planned = []
        out = []
        for name in planned:
            d = os.path.join(rp, name)
            if os.path.exists(os.path.join(d, "status.txt")):
                out.append({"name": name, "status": "done"})
            elif os.path.isdir(d):
                out.append({"name": name, "status": "training"})
            else:
                out.append({"name": name, "status": "pending"})
        return out

    # ------------------------------------------------------------------ #
    def load_leaderboard(self, results_path=None, limit=40):
        path = results_path or os.path.join(self.results_root, "leaderboard.csv")
        if results_path is None:
            candidates = [self.target_results_path(t) + os.sep + "leaderboard.csv"
                          for t in self.config.get("columns", {}).get("targets", [])]
        else:
            candidates = [os.path.join(results_path, "leaderboard.csv")]
        for c in candidates:
            if os.path.exists(c):
                try:
                    df = pd.read_csv(c)
                    if limit and len(df) > limit:
                        df = df.head(limit)
                    return df.to_dict("records")
                except Exception:
                    return []
        return []

    # ------------------------------------------------------------------ #
    def pickle_automl(self, target, automl):
        os.makedirs(self.models_root, exist_ok=True)
        path = self.target_pickle_path(target)
        tmp = path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(automl, f)
            os.replace(tmp, path)
        except Exception as e:
            self.log(f"WARNING: could not pickle AutoML for {target}: {e}")

    def load_automl_pickle(self, target):
        path = self.target_pickle_path(target)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def save_chain_preds(self, df_chain):
        df_chain.to_csv(self.chain_preds_path, index=False)

    def load_chain_preds(self):
        if os.path.exists(self.chain_preds_path):
            try:
                return pd.read_csv(self.chain_preds_path)
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------ #
    def mark_finished_target(self, target):
        self._finished_targets.add(target)
        self.save()

    def last_finished_target(self):
        for t in self.config.get("columns", {}).get("targets", []):
            rp = self.target_results_path(t)
            if os.path.exists(os.path.join(rp, "leaderboard.csv")):
                return t
        return None

    def mark_running(self):
        self.status = STATUS_RUNNING
        self.running = True
        self.stop_requested = False
        self.pause_requested = False
        self.save()

    def mark_finished(self):
        self.status = STATUS_FINISHED
        self.running = False
        self.pause_requested = False
        self.save()

    def mark_paused(self):
        self.status = STATUS_PAUSED
        self.running = False
        self.pause_requested = False
        self.save()

    def mark_error(self, err):
        self.status = STATUS_ERROR
        self.last_error = str(err)
        self.running = False
        self.save()

    def mark_stopped(self):
        self.status = STATUS_STOPPED
        self.running = False
        self.pause_requested = False
        self.save()

    # ------------------------------------------------------------------ #
    def is_finished_on_disk(self, target):
        """True when a target already produced results (leaderboard present)."""
        return os.path.exists(os.path.join(self.target_results_path(target),
                                           "leaderboard.csv"))

    # ------------------------------------------------------------------ #
    def data_snapshot_path(self):
        return os.path.join(self.data_root, "working_data.csv")

    def save_data_snapshot(self, df):
        os.makedirs(self.data_root, exist_ok=True)
        df.to_csv(self.data_snapshot_path(), index=False)

    def load_data_snapshot(self):
        p = self.data_snapshot_path()
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                return None
        return None


# --------------------------------------------------------------------------
# Registry (session history)
# --------------------------------------------------------------------------
def sessions_dir(workdir):
    d = os.path.join(workdir, "sessions")
    os.makedirs(d, exist_ok=True)
    return d


def _registry_path(workdir):
    return os.path.join(sessions_dir(workdir), REGISTRY_FNAME)


def list_sessions(workdir, include_meta=True):
    d = sessions_dir(workdir)
    out = []
    rp = _registry_path(workdir)
    known = {}
    if os.path.exists(rp):
        try:
            with open(rp, "r", encoding="utf-8") as f:
                known = {s["id"]: s for s in json.load(f)}
        except Exception:
            known = {}
    for name in sorted(os.listdir(d), reverse=True):
        sdir = os.path.join(d, name)
        if not os.path.isdir(sdir):
            continue
        meta = known.get(name)
        if meta is None or not include_meta:
            s = load_session(sdir)
            meta = s.to_meta() if s is not None else {"id": name, "name": name, "status": "unknown"}
        out.append(meta)
    return out


def create_session(workdir, name=None):
    sid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    sdir = os.path.join(sessions_dir(workdir), sid)
    os.makedirs(sdir, exist_ok=True)
    s = Session(sdir)
    s.name = (name or "Session").strip() or sid
    s.id = sid
    s.save()
    _update_registry(workdir, s)
    return s


def load_session(session_dir):
    if not os.path.isdir(session_dir):
        return None
    s = Session(session_dir)
    p = os.path.join(session_dir, SESSION_FNAME)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            s.set_config(payload.get("config", {}))
            s.id = payload.get("id", os.path.basename(session_dir))
            s.name = payload.get("name", s.id)
            s.created_at = payload.get("created_at", s.created_at)
            s.updated_at = payload.get("updated_at", s.updated_at)
            s.status = payload.get("status", STATUS_IDLE)
            s.last_error = payload.get("last_error")
            s._finished_targets = set(payload.get("targets_finished", []))
        except Exception:
            pass
    # replay log tail from disk
    if os.path.exists(s.logs_path):
        try:
            with open(s.logs_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
            s.log_buffer = lines[-800:]
        except Exception:
            pass
    return s


def delete_session(workdir, sid):
    sdir = os.path.join(sessions_dir(workdir), sid)
    if os.path.isdir(sdir):
        shutil.rmtree(sdir, ignore_errors=True)
    _update_registry(workdir, None, remove=sid)


def _update_registry(workdir, session, remove=None):
    rp = _registry_path(workdir)
    rows = []
    if os.path.exists(rp):
        try:
            with open(rp, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:
            rows = []
    if remove:
        rows = [r for r in rows if r.get("id") != remove]
    elif session is not None:
        meta = session.to_meta()
        rows = [r for r in rows if r.get("id") != session.id] + [meta]
        rows = sorted(rows, key=lambda r: r.get("updated_at", ""), reverse=True)
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
