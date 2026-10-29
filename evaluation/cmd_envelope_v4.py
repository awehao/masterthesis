#!/usr/bin/env python3
"""命令追蹤封裝 v4（mode 4 重設計 R1–R3；Codex 20261029_154350 §4：任期標記須來自求解端封裝，不由入口補成「當前任期」）。
v3（cmd_envelope_v3，19 欄，不改）之後加一欄 tenure，schema 4.0、共 20 欄。前 19 欄沿用 v3 的 encode3／decode3 嚴格檢查。
tenure：非負整數；求解端在哪一段全身任期下算出這筆（求解端發布時綁定，adapter 原樣轉送，不得改寫）。
話題與 v3 分開：求解端 /wgmpc/cmd_env4 → adapter（k3m4）→ /wb_vel_cmd_env4 → 執行端（ev9r8p）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmd_envelope_v3 as E3                                     # noqa: E402
import wgmpc_cmd_envelope as V2                                  # noqa: E402

SCHEMA4 = 4.0
NFIELD4 = E3.NFIELD3 + 1
COLS4 = list(E3.COLS3) + ['tenure']
TOPIC4 = {V2.ST_SOLVER: '/wgmpc/cmd_env4', V2.ST_ADAPTER: '/wb_vel_cmd_env4'}


def encode4(run_id_n, source_seq, stage_id, output_seq, src_seq, derived, stamp_sim_t, u, mode_seq, tenure, kind=V2.K_NORMAL):
    if type(tenure) is not int or tenure < 0:
        raise ValueError(f'tenure 必須是非負整數，收到 {tenure!r}')
    d = E3.encode3(run_id_n, source_seq, stage_id, output_seq, src_seq, derived, stamp_sim_t, u, mode_seq, kind=kind)
    d[0] = float(SCHEMA4)
    return d + [float(tenure)]


def decode4(data):
    d = [float(x) for x in data]
    if len(d) != NFIELD4:
        raise ValueError(f'v4 封裝應有 {NFIELD4} 欄，收到 {len(d)}')
    if abs(d[0] - SCHEMA4) > 1e-9:
        raise ValueError(f'v4 封裝 schema 不符：{d[0]} != {SCHEMA4}')
    tn = d[-1]
    if not (tn == tn) or abs(tn - round(tn)) > 1e-9 or tn < 0:
        raise ValueError(f'tenure 必須是非負整數，收到 {tn!r}')
    e = E3.decode3([float(E3.SCHEMA3)] + d[1:-1])
    e['schema'] = SCHEMA4
    e['tenure'] = int(round(tn))
    return e
