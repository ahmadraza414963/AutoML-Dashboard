# -*- coding: utf-8 -*-
"""AutoML Dashboard package.

Importing any `dashboard.*` module ensures the vendored fork
(`dashboard/vendor/supervised`) wins on sys.path before any other module that
depends on it — keeping the project fully self-contained inside this folder.
"""
from . import _vendored  # noqa: F401  (must run before `import supervised`)