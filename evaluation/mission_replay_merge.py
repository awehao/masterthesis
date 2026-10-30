#!/usr/bin/env python3
"""同趟重播來源合併（門→抽屜串接 S1；Codex m5i 核可先交同趟影片）：把同一趟的門段 room_run.json 與抽屜段 room_run_drawer.json
依物理步號串成一份重播來源（不補造、不混用他趟）。欄位：step、sim_t、owner、kind、base_xyth、q_arm_meas、q_finger、door_q_rad、opening_m。
  - 門段列的 opening_m 取 0.0：門段紀錄沒有抽屜開度欄（抽屜在門段未被接觸）——此為揭露項，不是量測值；
  - 核對：抽屜段首步號＝門段末步號＋1、模擬時間單調、兩份來源 sha256 記入輸出。
    python3 evaluation/mission_replay_merge.py <run_dir>   ⇒ <run_dir>/room_run_mission_merged.json（已存在則拒絕覆寫）
"""
import hashlib
import json
import os
import sys

COLS = ['step', 'sim_t', 'owner(0=nav,1=wb,2=glide)', 'kind', 'base_xyth', 'q_arm_meas', 'q_finger', 'door_q_rad', 'opening_m']


def main():
    d = sys.argv[1]
    out = os.path.join(d, 'room_run_mission_merged.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    src = {k: os.path.join(d, f) for k, f in (('door', 'room_run.json'), ('drawer', 'room_run_drawer.json'))}
    raw = {k: open(p, 'rb').read() for k, p in src.items()}
    R = {k: json.loads(v) for k, v in raw.items()}
    rows = []
    for k in ('door', 'drawer'):
        c = R[k]['steps_cols']
        ix = {n: c.index(n) for n in COLS if n in c}
        miss = [n for n in COLS if n not in ix and not (k == 'door' and n == 'opening_m')]
        if miss:
            raise SystemExit(f'{k} 紀錄缺欄 {miss}')
        for x in R[k]['steps']:
            rows.append([x[ix[n]] if n in ix else 0.0 for n in COLS])
    nd = len(R['door']['steps'])
    s_last, s_first = rows[nd - 1][0], rows[nd][0]
    if s_first != s_last + 1:
        raise SystemExit(f'步號不銜接：門段末 {s_last}、抽屜段首 {s_first}')
    bad = [i for i in range(1, len(rows)) if not (rows[i][1] > rows[i - 1][1] and rows[i][0] == rows[i - 1][0] + 1)]
    if bad:
        raise SystemExit(f'步號／模擬時間非逐步遞增：{len(bad)} 處，首 {bad[:3]}')
    rep = {'steps_cols': COLS, 'steps': rows,
           'merge': {'version': 'mission_replay_merge/r1', 'sources': {k: {'path': os.path.relpath(p, d), 'sha256': hashlib.sha256(raw[k]).hexdigest(),
                                                                          'n_steps': len(R[k]['steps'])} for k, p in src.items()},
                     'door_last_step': s_last, 'drawer_first_step': s_first,
                     'door_rows_opening_note': '門段紀錄無抽屜開度欄 ⇒ 門段列 opening_m 取 0.0（抽屜未被接觸；揭露項，非量測）'}}
    json.dump(rep, open(out, 'w'), ensure_ascii=False)
    print(f'合併 {len(rows)} 步（門段 {nd}＋抽屜段 {len(R["drawer"]["steps"])}）；門段末步 {s_last} → 抽屜段首步 {s_first}；→ {out}')


if __name__ == '__main__':
    main()
