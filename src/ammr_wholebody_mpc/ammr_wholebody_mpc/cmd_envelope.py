#!/usr/bin/env python3
"""WG2 命令追蹤的**封裝格式**（身分與九維值在同一份訊息）。

為什麼要封裝
------------
全鏈原本缺逐筆關聯：節點只有自己的發布時刻，執行端只有自己的
`recv_seq`（它自己的計數），中間還經過 adapter、安全層與 E2。
因此無法確定「這次究竟執行了哪筆、被改成什麼」，
`D_cmd`（發布 → 生效）也量不到。

設計要點
--------
* **身分與值同訊息**：`(run_id, source_seq)` 與九維命令在同一份 payload，
  **不另發旁路資料**，也不靠數值或接收順序猜配對。
* **每一段自己的輸出序號**：每次輸出都有 `output_seq`，並保留它由哪一筆
  來源導出（`src_seq`）。同一請求可能被**多次處理**、被**修改**、
  或被**替換成停止命令**。
* **`kind` 區分套用種類**：正常、被本段修改、自行產生的停止、無命令、
  失效閂鎖停止。**失效後自行產生的停止值不得冒稱原命令成功套用。**
* **`derived` 與 `src_seq` 是兩件事**：前者問「可不可歸屬到某筆**求解命令**」，
  後者記「由上游**哪一筆訊息**導出」。轉送一筆安全層自行產生的停止時，
  `derived = 0`（不可歸屬到求解命令）但 `src_seq` 仍保留上游的 output_seq。
* **唯一控制入口**：啟用封裝的那一段**只**從封裝取控制值；
  舊九維話題不得同時更新控制值（見各節點的互斥實作）。

**界線**：序號只說明「執行了哪筆、被改成什麼」，
**不能預先保證未來安全層不修改命令**。
"""
from __future__ import annotations

import hashlib
import math

SCHEMA = 2.0        # 2.0 起加入 kind，並對身分與時間做嚴格檢查
NU = 9

# 段別
ST_SOLVER, ST_SAFETY, ST_ADAPTER, ST_ENDPOINT = 0, 1, 2, 3
STAGE_NAME = {ST_SOLVER: 'solver', ST_SAFETY: 'safety',
              ST_ADAPTER: 'adapter', ST_ENDPOINT: 'endpoint'}

# 本段輸出的種類
K_NORMAL, K_MODIFIED, K_STOP_GEN, K_NO_COMMAND, K_FAIL_LATCHED = 0, 1, 2, 3, 4
KIND_NAME = {K_NORMAL: 'normal', K_MODIFIED: 'modified_by_this_stage',
             K_STOP_GEN: 'stop_generated', K_NO_COMMAND: 'no_command',
             K_FAIL_LATCHED: 'fail_latched_stop'}
# 這些種類**不是**由某筆來源導出 ⇒ derived 必為 0，不得冒認來源
KIND_NOT_DERIVED = (K_STOP_GEN, K_NO_COMMAND, K_FAIL_LATCHED)

COLS = ['schema', 'run_id', 'source_seq', 'stage_id', 'output_seq',
        'src_seq', 'derived', 'kind', 'stamp_sim_t'] + [f'u{i}'
                                                        for i in range(NU)]
NFIELD = len(COLS)          # 9 + 9 = 18

TOPIC = {ST_SOLVER: '/wgmpc/cmd_env',
         ST_SAFETY: '/wholebody_safety/cmd_env_out',
         ST_ADAPTER: '/wb_vel_cmd_env',
         ST_ENDPOINT: '/coman/applied_env'}


def run_id_num(run_id: str) -> float:
    """把 run_id 字串壓成**可精確放進 float64 的整數**（48 bit < 2^53）。"""
    h = hashlib.sha256(str(run_id).encode('utf-8')).digest()
    return float(int.from_bytes(h[:6], 'big'))


def _int_exact(x, name):
    """要求整數語意，**不靜默截斷**。"""
    if not math.isfinite(x):
        raise ValueError(f'{name} 非有限值：{x!r}')
    if abs(x - round(x)) > 1e-9:
        raise ValueError(f'{name} 必須是整數，收到 {x!r}（**不截斷**）')
    return int(round(x))


def encode(run_id_n: float, source_seq: int, stage_id: int, output_seq: int,
           src_seq: int, derived: bool, stamp_sim_t: float, u,
           kind: int = K_NORMAL) -> list:
    u = [float(x) for x in u]
    if len(u) != NU:
        raise ValueError(f'命令必須是 {NU} 維，收到 {len(u)}')
    if not all(math.isfinite(v) for v in u):
        raise ValueError('命令含非有限值，拒絕封裝')
    if not math.isfinite(stamp_sim_t):
        raise ValueError('stamp_sim_t 非有限值，拒絕封裝')
    if stage_id not in STAGE_NAME:
        raise ValueError(f'段別不合法：{stage_id}')
    if kind not in KIND_NAME:
        raise ValueError(f'kind 不合法：{kind}')
    if kind in KIND_NOT_DERIVED and derived:
        raise ValueError(f'{KIND_NAME[kind]} 不得標為 derived')
    if not derived and int(source_seq) != -1:
        raise ValueError('derived = 0 時 source_seq 必須為 −1'
                         '（不得冒認任何**求解命令**）')
    if derived and int(source_seq) < 0:
        raise ValueError('derived = 1 時 source_seq 必須 >= 0')
    return ([float(SCHEMA), float(run_id_n), float(source_seq),
             float(stage_id), float(output_seq), float(src_seq),
             1.0 if derived else 0.0, float(kind),
             float(stamp_sim_t)] + u)


def decode(data) -> dict:
    """嚴格解析：**身分與時間的有效性不靠下游碰巧補救。**"""
    d = [float(x) for x in data]
    if len(d) != NFIELD:
        raise ValueError(f'封裝應有 {NFIELD} 欄，收到 {len(d)}')
    if not all(math.isfinite(v) for v in d):
        bad = [COLS[i] for i, v in enumerate(d) if not math.isfinite(v)]
        raise ValueError(f'封裝含非有限值：{bad}')
    # NaN 比較永遠為 False，所以要先擋非有限值再比 schema
    if abs(d[0] - SCHEMA) > 1e-9:
        raise ValueError(f'封裝 schema 不符：{d[0]} != {SCHEMA}')
    src = _int_exact(d[2], 'source_seq')
    stg = _int_exact(d[3], 'stage_id')
    out = _int_exact(d[4], 'output_seq')
    ssq = _int_exact(d[5], 'src_seq')
    kind = _int_exact(d[7], 'kind')
    drv_raw = d[6]
    if abs(drv_raw) > 1e-9 and abs(drv_raw - 1.0) > 1e-9:
        raise ValueError(f'derived 必須是 0 或 1，收到 {drv_raw!r}')
    drv = drv_raw >= 0.5
    if stg not in STAGE_NAME:
        raise ValueError(f'段別不合法：{stg}')
    if kind not in KIND_NAME:
        raise ValueError(f'kind 不合法：{kind}')
    if kind in KIND_NOT_DERIVED and drv:
        raise ValueError(f'{KIND_NAME[kind]} 不得標為 derived')
    if not drv and src != -1:
        raise ValueError(f'derived = 0 但 source_seq={src}'
                         '（不得冒認任何求解命令）')
    if drv and src < 0:
        raise ValueError(f'derived = 1 但 source_seq={src}')
    return {'schema': d[0], 'run_id': d[1], 'source_seq': src,
            'stage_id': stg, 'output_seq': out, 'src_seq': ssq,
            'derived': drv, 'kind': kind, 'kind_name': KIND_NAME[kind],
            'stamp_sim_t': d[8], 'u': d[9:9 + NU]}


def describe() -> dict:
    return {
        'cols': list(COLS), 'schema': SCHEMA, 'n_fields': NFIELD,
        'stage_ids': {str(k): v for k, v in STAGE_NAME.items()},
        'kinds': {str(k): v for k, v in KIND_NAME.items()},
        'topics': {STAGE_NAME[k]: v for k, v in TOPIC.items()},
        'identity_with_value': '身分 (run_id, source_seq) 與九維值在**同一份訊息**；'
                               '不另發旁路資料、不靠數值或接收順序猜配對',
        'output_seq': '本段自己的輸出計數；同一來源可能被**多次輸出**',
        'src_seq': '本筆由上游的哪個 output_seq 導出。**與 derived 無關** ——'
                   '轉送一筆「安全層自行產生的停止」時，它仍有上游 output_seq，'
                   '那是來歷，要保留；沒有上游訊息時才是 −1',
        'derived': '**是否可歸屬到某筆求解命令**（solver 的 source_seq）。'
                   '1 = 可歸屬（**值可能已被修改**，見 kind）；'
                   '0 = 不可歸屬 ⇒ source_seq 必為 −1，'
                   '但 src_seq 仍可保留上游來歷',
        'kind_meaning': '**失效後自行產生的停止值（fail_latched_stop）'
                        '不得冒稱原命令成功套用**',
        'strict_decode': '非有限值、非整數序號、不合法段別／kind、'
                         'derived 與來源欄位不一致 ⇒ **一律拒絕**，不靜默截斷',
        'single_entry': '啟用封裝的那一段**只**從封裝取控制值；'
                        '舊九維話題不得同時更新控制值'
                        '（安全層忽略 ~/cmd_in；adapter 與執行端'
                        '**根本不建立**舊訂閱）',
        'existing_path': '既有的九維話題與限制一字未改；'
                         '封裝是 WG2 明確啟用的額外路徑，預設全部關閉',
        'not_guaranteed': '序號只說明「執行了哪筆、被改成什麼」，'
                          '**不保證未來安全層不修改命令**',
    }
