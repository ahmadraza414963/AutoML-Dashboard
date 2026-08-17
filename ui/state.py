# -*- coding: utf-8 -*-
"""Process-global UI state (single Dash process, so a module singleton is fine)."""
import os

import pandas as pd

DEFAULT_WORKDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workdir")


class AppState:
    def __init__(self):
        self.workdir = os.environ.get("DASH_WORKDIR", DEFAULT_WORKDIR)
        os.makedirs(self.workdir, exist_ok=True)
        self.session = None          # currently selected Session
        self.df = None               # currently loaded dataframe
        self.df_name = None
        self.df_sheet = None

    def set_workdir(self, path):
        path = path or DEFAULT_WORKDIR
        os.makedirs(path, exist_ok=True)
        self.workdir = os.path.abspath(path)
        self.session = None
        self.df = None

    def set_df(self, df, name, sheet=None):
        self.df = df
        self.df_name = name
        self.df_sheet = sheet

    def columns(self):
        return list(self.df.columns) if self.df is not None else []


_state = AppState()


def get_state():
    return _state