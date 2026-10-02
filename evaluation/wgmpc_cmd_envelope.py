#!/usr/bin/env python3
"""WG2 命令追蹤封裝 —— **權威定義在套件內**，本檔只是 evaluation 側的轉出口。

格式只有一份（`ammr_wholebody_mpc.cmd_envelope`），避免兩邊各寫一份而漂移。
"""
from __future__ import annotations

import os
import sys

_WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_WS, 'src', 'ammr_wholebody_mpc'))

from ammr_wholebody_mpc.cmd_envelope import (  # noqa: E402,F401
    COLS, NFIELD, NU, SCHEMA, ST_ADAPTER, ST_ENDPOINT, ST_SAFETY, ST_SOLVER,
    STAGE_NAME, TOPIC, decode, describe, encode, run_id_num)
