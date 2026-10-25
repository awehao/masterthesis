#!/usr/bin/env python3
"""命令追蹤封裝 v3（W2；Codex 20261025_201543：mode_seq 與 source_seq 一起穿過 adapter）。

在既有 schema 2.0 的 18 欄（ammr_wholebody_mpc.cmd_envelope，經 wgmpc_cmd_envelope 轉出，**不改**）之後加一欄 mode_seq，
schema 改為 3.0、共 19 欄。前 18 欄的語意與檢查**逐字沿用** v2 的 encode／decode（先把 schema 欄換回 2.0 交給 v2 驗證）。
mode_seq：非負整數；本筆命令屬於哪一段模式（求解端發布時綁定，adapter 原樣轉送，不得改寫或以「最新模式」猜測）。

話題與 v2 分開，避免 schema 2 的既有消費端收到 19 欄：
  求解端 /wgmpc/cmd_env3 → adapter（W2 版）→ /wb_vel_cmd_env3 → 執行端（W2 版）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wgmpc_cmd_envelope as V2                                  # noqa: E402

SCHEMA3 = 3.0
NFIELD3 = V2.NFIELD + 1
COLS3 = list(V2.COLS) + ['mode_seq']
TOPIC3 = {V2.ST_SOLVER: '/wgmpc/cmd_env3', V2.ST_ADAPTER: '/wb_vel_cmd_env3'}


def encode3(run_id_n, source_seq, stage_id, output_seq, src_seq, derived, stamp_sim_t, u, mode_seq, kind=V2.K_NORMAL):
    if type(mode_seq) is not int or mode_seq < 0:
        raise ValueError(f'mode_seq 必須是非負整數，收到 {mode_seq!r}')
    d = V2.encode(run_id_n, source_seq, stage_id, output_seq, src_seq, derived, stamp_sim_t, u, kind=kind)
    d[0] = float(SCHEMA3)
    return d + [float(mode_seq)]


def decode3(data):
    d = [float(x) for x in data]
    if len(d) != NFIELD3:
        raise ValueError(f'v3 封裝應有 {NFIELD3} 欄，收到 {len(d)}')
    if abs(d[0] - SCHEMA3) > 1e-9:
        raise ValueError(f'v3 封裝 schema 不符：{d[0]} != {SCHEMA3}')
    ms = d[-1]
    if not (ms == ms) or abs(ms - round(ms)) > 1e-9 or ms < 0:
        raise ValueError(f'mode_seq 必須是非負整數，收到 {ms!r}')
    e = V2.decode([float(V2.SCHEMA)] + d[1:-1])                 # 前 18 欄沿用 v2 的嚴格檢查
    e['schema'] = SCHEMA3
    e['mode_seq'] = int(round(ms))
    return e
