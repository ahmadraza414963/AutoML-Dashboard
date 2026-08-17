# -*- coding: utf-8 -*-
"""Make the vendored fork (`dashboard/vendor/supervised`) the authoritative
`supervised` package so the whole project is self-contained inside this
folder and never imports the fork from site-packages or anywhere else."""
import os
import sys

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if os.path.isdir(os.path.join(_VENDOR, "supervised")):
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)