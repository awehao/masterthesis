#!/usr/bin/env python3
"""ref pb1b（切法 B；Codex 20261028）：ref pb1 只改 segments['3']——去掉退離前的原地張爪等待（s≈0 且 qf 張開中的列），
自最後一個原地列（靜止、與退離起點同構型）起算 t＝0，退離時間表、構型、速度／加速度不變（平滑起步沿用原 LSPB 起點靜止）；
qf＝原段自 t＝0 起的張開曲線照原時間軸套到新段（r2；r1 以 0.75 s 截斷在 0.0083，原曲線退離開始後仍張到 0.0089，r1 輸出改名 *_r1_qf_truncated.json）；
只供監看預測，手指命令唯一來源是權威 finger_target。
s、src_index、stage 隨列保留；mode3_s_end_m 不變。其餘段與欄位逐字同 ref pb1；登錄自驗須通過。

    python3 evaluation/door_regrasp_ref_pb1b.py   # 寫 results/door/door_regrasp_ref_pb1b.json（不覆寫）
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_regrasp_ref_registry as REG                # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb1.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb1b.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p = json.load(open(SRC))
    s3 = p['segments']['3']
    n = len(s3['t'])
    i_mov = next(i for i in range(n) if s3['s'][i] > 1e-6 or any(abs(a - b) > 1e-12 for a, b in zip(s3['base'][i] + s3['arm'][i], s3['base'][0] + s3['arm'][0])))
    i0 = i_mov - 1
    T_open = s3['t'][i0]
    qf0, qf1 = s3['qf'][0], s3['qf'][i0]
    new = {k: (v[i0:] if isinstance(v, list) and len(v) == n else v) for k, v in s3.items()}
    t0 = new['t'][0]
    new['t'] = [t - t0 for t in new['t']]
    new['qf'] = [s3['qf'][j] for j in range(len(new['t']))]                       # r2：原張開曲線、同一時間軸
    p['segments']['3'] = new
    p['schema'] = p.get('schema', '') + '+pb1b'
    p['pb1b'] = {'derived_from': {'path': os.path.relpath(SRC, HERE), 'sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest()},
                 'dropped_rows': i0, 'T_open_s': T_open, 'qf': [qf0, new['qf'][-1]], 'seg3_T_old': s3['t'][-1], 'seg3_T_new': new['t'][-1],
                 'note': '張爪與退離同步自 mode 3 提交步開始（切法 B）；退離時間表不變'}
    ok, why = REG.validate(json.loads(json.dumps(p, default=float)))
    p['registry_validate'] = {'ok': ok, 'why': why}
    if not ok:
        raise SystemExit(f'登錄自驗不過：{why}')
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(p['pb1b'], ensure_ascii=False, default=float)[:400])
    print('前 1.2 s：', [(round(t, 2), round(q, 4), round(s * 1e3, 1)) for t, q, s in zip(new['t'], new['qf'], new['s'])][:25:3])


if __name__ == '__main__':
    main()
