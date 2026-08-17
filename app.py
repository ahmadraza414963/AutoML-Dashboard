# -*- coding: utf-8 -*-
"""
AutoML Dashboard entry point.

Run with:
    env\\python.exe -m dashboard.app
"""
import os
import sys

import dash
import dash_bootstrap_components as dbc


def _apply_patches():
    """Order matters: register sklearn regressors, fix plotting bugs, make
    the fork's in-memory data persistence disk-based (resume support)."""
    from .core.sklearn_regressors import register_sklearn_regressors
    register_sklearn_regressors()

    from .core.mlsuper_patch import apply_mlsuper_patches
    apply_mlsuper_patches()

    from .core.plot_fixes import apply_plot_fixes
    apply_plot_fixes()


def create_app():
    _apply_patches()
    from .ui.layout import layout
    from .ui.callbacks import register_callbacks

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP,
                              "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"],
        title="AutoML Dashboard",
        suppress_callback_exceptions=True,
    )
    app.layout = layout()
    register_callbacks(app)

    # serve generated HTML reports
    from .ui.state import get_state

    @app.server.route("/reports/<path:subpath>")
    def reports_route(subpath):
        from flask import send_from_directory
        sid = subpath.split("/")[0]
        fname = subpath.split("/")[1] if "/" in subpath else subpath
        sess_dir = os.path.join(get_state().workdir, "sessions", sid)
        return send_from_directory(sess_dir, fname)

    return app


def main():
    host = os.environ.get("DASH_HOST", "127.0.0.1")
    port = int(os.environ.get("DASH_PORT", "8050"))
    debug = os.environ.get("DASH_DEBUG", "0") == "1"
    app = create_app()
    print(f"AutoML Dashboard -> http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
