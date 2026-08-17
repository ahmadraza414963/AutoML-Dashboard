# -*- coding: utf-8 -*-
"""Background worker that executes a session's chain run in a daemon thread,
capturing the full stdout/stderr verbosity produced by AutoML."""
import io
import json
import os
import threading
import time
import sys


class TeeBuffer:
    """Re-routes written text to a callback (and optionally the real stream)."""

    def __init__(self, emit, also=None):
        self._emit = emit
        self._also = also or []
        self._buf = io.StringIO()

    def write(self, text):
        if not text:
            return
        try:
            self._emit(text)
        except Exception:
            pass
        try:
            for s in self._also:
                s.write(text)
        except Exception:
            pass
        self._buf.write(text)

    def flush(self):
        try:
            for s in self._also:
                s.flush()
        except Exception:
            pass

    def getvalue(self):
        return self._buf.getvalue()


class SessionRunner:
    def __init__(self, session):
        self.session = session

    def start(self, resume=False):
        session = self.session
        if session.running and session.thread and session.thread.is_alive():
            return False
        session.stop_requested = False
        session.pause_requested = False
        session.resume_mode = resume
        session._finished_targets = set(
            self._detect_finished_from_disk(session)
        )

        def progress(target, msg):
            if msg == "starting":
                session.log(f"START target '{target}'")
            else:
                session.log(f"END   target '{target}'")

        def stop_check():
            if session.pause_requested:
                return "pause"
            if session.stop_requested:
                return "stop"
            return False

        def run():
            session.mark_running()
            session.log("=" * 60)
            label = "RESUME" if resume else "RUN"
            session.log(f"{label} session {session.id} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
            if resume and session._finished_targets:
                session.log(f"Resuming — {len(session._finished_targets)} already-finished "
                            "target(s) will be skipped and reused from disk.")
            real_stdout, real_stderr = sys.stdout, sys.stderr

            def emit(text):
                for line in text.rstrip("\r\n").split("\n"):
                    if line.strip():
                        session.log(line.rstrip())

            tee = TeeBuffer(emit)  # do NOT echo back to console (keeps UI clean)
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = tee, tee
            try:
                from .chain import run_chain
                session.log("Worker thread started.")
                ok = run_chain(session, log=session.log,
                               stop_check=stop_check, progress=progress)
                if ok:
                    session.mark_finished()
                    session.log("Session finished successfully.")
                # pause/stop handled inside run_chain (marks paused/stopped)
                if session.status == "running":
                    session.mark_finished()
                if session.status == "finished":
                    try:
                        from .reports import build_html_report
                        build_html_report(session)
                        session.log("HTML report generated.")
                    except Exception as e:
                        session.log(f"WARNING: report generation failed: {e}")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                session.log(f"ERROR: {e}")
                session.log(tb)
                session.mark_error(f"{e}\n{tb[-1500:]}")
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        session.thread = threading.Thread(target=run, daemon=True, name=f"session-{session.id}")
        session.thread.start()
        return True

    @staticmethod
    def _detect_finished_from_disk(session):
        done = set()
        for t in session.config.get("columns", {}).get("targets", []):
            if session.is_finished_on_disk(t):
                done.add(t)
        return done

    def pause(self):
        self.session.pause_requested = True

    def resume(self):
        return self.start(resume=True)

    def stop(self):
        self.session.stop_requested = True


class RunnerRegistry:
    """All sessions that are currently running in this process."""
    _sessions = {}

    @classmethod
    def start(cls, session):
        runner = SessionRunner(session)
        ok = runner.start()
        if ok:
            cls._sessions[session.id] = runner
        return ok

    @classmethod
    def resume(cls, session):
        runner = SessionRunner(session)
        ok = runner.start(resume=True)
        if ok:
            cls._sessions[session.id] = runner
        return ok

    @classmethod
    def pause(cls, session_id):
        r = cls._sessions.get(session_id)
        if r:
            r.pause()

    @classmethod
    def stop(cls, session_id):
        r = cls._sessions.get(session_id)
        if r:
            r.stop()

    @classmethod
    def is_running(cls, session_id):
        r = cls._sessions.get(session_id)
        if not r:
            return False
        s = r.session
        return bool(s.running and s.thread and s.thread.is_alive())

    @classmethod
    def session_status(cls, session_id):
        r = cls._sessions.get(session_id)
        if not r:
            return "not_loaded"
        return r.session.status