#!/usr/bin/env python3
"""WG2 命令追蹤的**封裝格式**（身分與九維值在同一份訊息）。

為什麼要封裝
------------
目前全鏈缺逐筆關聯：節點只有自己的發布時刻，執行端只有自己的
`recv_seq`（它自己的計數），中間還經過 adapter、安全層與 E2。
因此無法確定「這次究竟執行了哪筆、被改成什麼」，
`D_cmd`（發布 → 生效）也量不到。

設計要點
--------
* **身分與值同訊息**：`(run_id, source_seq)` 與九維命令放在同一筆 payload，
  **不另發旁路資料**，也不靠數值或接收順序猜配對。
* **每一段自己的輸出序號**：安全層每次輸出有 `output_seq`，並保留它
  由哪一筆來源導出（`src_seq`）。同一請求可能被**多次處理**，
  也可能被**修改**或**替換成停止命令** —— 後者 `derived = 0`。
* **既有路徑不變**：原本的九維話題與限制一字未改；
  封裝是 WG2 明確啟用的**額外**路徑。

**界線**：序號能讓我們知道「執行了哪筆、被改成什麼」，
**不能預先保證未來安全層不修改命令**。它是修好延遲估計與預推的
必要工具，不是加上序號就會穩定的保證。
"""
from __future__ import annotations

import hashlib

SCHEMA = 1.0
NU = 9

# 段別
ST_SOLVER, ST_SAFETY, ST_ADAPTER, ST_ENDPOINT = 0, 1, 2, 3
STAGE_NAME = {ST_SOLVER: 'solver', ST_SAFETY: 'safety',
              ST_ADAPTER: 'adapter', ST_ENDPOINT: 'endpoint'}

COLS = ['schema', 'run_id', 'source_seq', 'stage_id', 'output_seq',
        'src_seq', 'derived', 'stamp_sim_t'] + [f'u{i}' for i in range(NU)]
NFIELD = len(COLS)          # 8 + 9 = 17

# 話題（**額外**路徑；既有九維話題不變）
TOPIC = {ST_SOLVER: '/wgmpc/cmd_env',
         ST_SAFETY: '/wholebody_safety/cmd_env_out',
         ST_ADAPTER: '/wb_vel_cmd_env',
         ST_ENDPOINT: '/coman/applied_env'}


def run_id_num(run_id: str) -> float:
    """把 run_id 字串壓成**可精確放進 float64 的整數**（48 bit）。

    float64 的整數精度上限是 2^53，取 48 bit 留足餘裕，不會因浮點而失真。
    """
    h = hashlib.sha256(str(run_id).encode('utf-8')).digest()
    return float(int.from_bytes(h[:6], 'big'))


def encode(run_id_n: float, source_seq: int, stage_id: int, output_seq: int,
           src_seq: int, derived: bool, stamp_sim_t: float, u) -> list:
    u = [float(x) for x in u]
    if len(u) != NU:
        raise ValueError(f'命令必須是 {NU} 維，收到 {len(u)}')
    return ([float(SCHEMA), float(run_id_n), float(source_seq),
             float(stage_id), float(output_seq), float(src_seq),
             1.0 if derived else 0.0, float(stamp_sim_t)] + u)


def decode(data) -> dict:
    d = [float(x) for x in data]
    if len(d) != NFIELD:
        raise ValueError(f'封裝應有 {NFIELD} 欄，收到 {len(d)}')
    if abs(d[0] - SCHEMA) > 1e-9:
        raise ValueError(f'封裝 schema 不符：{d[0]} != {SCHEMA}')
    return {'schema': d[0], 'run_id': d[1], 'source_seq': int(d[2]),
            'stage_id': int(d[3]), 'output_seq': int(d[4]),
            'src_seq': int(d[5]), 'derived': bool(d[6] >= 0.5),
            'stamp_sim_t': d[7], 'u': d[8:8 + NU]}


def describe() -> dict:
    """供各段落盤的格式說明（latched meta 用）。"""
    return {
        'cols': list(COLS), 'schema': SCHEMA, 'n_fields': NFIELD,
        'stage_ids': {str(k): v for k, v in STAGE_NAME.items()},
        'topics': {STAGE_NAME[k]: v for k, v in TOPIC.items()},
        'identity_with_value': '身分 (run_id, source_seq) 與九維值在**同一份訊息**；'
                               '不另發旁路資料、不靠數值或接收順序猜配對',
        'output_seq': '本段自己的輸出計數；同一來源可能被**多次輸出**',
        'src_seq': '本筆由上游的哪個 output_seq 導出；'
                   'solver 段填自己的 source_seq',
        'derived': '1 = 由來源導出（可能被修改）；'
                   '0 = **本段自行產生**（停止命令、無命令等），無上游來源',
        'not_guaranteed': '序號只說明「執行了哪筆、被改成什麼」，'
                          '**不保證未來安全層不修改命令**',
        'existing_path': '既有的九維話題與限制一字未改；本封裝是 WG2 明確啟用的額外路徑',
    }
