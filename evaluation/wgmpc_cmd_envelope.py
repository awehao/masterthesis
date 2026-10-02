#!/usr/bin/env python3
"""WG2 命令追蹤封裝 —— **權威定義在套件內**，本檔只是 evaluation 側的轉出口。

格式只有一份（`ammr_wholebody_mpc.cmd_envelope`），避免兩邊各寫一份而漂移。
"""
from __future__ import annotations

import os
import sys

_WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_WS, 'src', 'ammr_wholebody_mpc'))

# **全量轉出**：新增常數時不必兩邊各改一次（先前漏了 K_* 就踩過）。
from ammr_wholebody_mpc import cmd_envelope as _impl              # noqa: E402

for _n in dir(_impl):
    if not _n.startswith('_'):
        globals()[_n] = getattr(_impl, _n)
del _n
