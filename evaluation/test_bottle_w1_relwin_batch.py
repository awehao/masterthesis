#!/usr/bin/env python3
"""多任務第一批 v2（bottle_w1_vhigh_relwin_ctrl）八案＋零位移退化的離線核對（Codex 20261102_050244 第 4 項；假世界、假候選，非物理）。
每案一個程序（第一桌高度在匯入前依案例改）：瓶位＝名目 (0, 0.42)＋偏移、第一桌高＝案例值、第二桌不變；假候選只在 V_HIGH 保持段送（中心＝生成瓶位 ± 0.3 mm、支撐＝桌高）。
生成瓶位只進假世界與評估，控制器只收假候選（模擬量測）。C1／C2 的動態物不在假世界（只核幾何與排程；動態交互在物理另報）。
每案分列：取得目標（採用時刻、V_HIGH 段）、平移量與排程、全窗分支（採用時 window_branch 結果）、實際相位路徑與結果、
  速率（設定點速率峰、0.05 s 控制命令變化率峰）、環境／自碰（PCK 全程最小）、停止核對（漸變起到抬升完成，每 10 個物理步〔n_step % 10＝0.1 s，Codex 055603 更正：不是每個控制拍〕 compactcarry21.stop_traj 現在停／延後一拍停：
  非手指與手指對第一桌、持瓶對第一桌〔只在離桌判定成立後；之前瓶站在桌上〕、自碰、關節餘裕；非有限、未停完）。PCK 全程最小另列連桿對桌與持瓶對桌（持瓶項含離桌前 0 值）。
退化：v2 名目（偏移 0）vs v1 名目，相位時刻與結果一致（視覺雜訊造成的微小平移另報）。
    python3 evaluation/test_bottle_w1_relwin_batch.py all
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, 'experiments', '20261102_multi_motm_tasks.json')))
CASES = {c['id']: (c['table1_z_m'], c['bottle_dx_m'], c['bottle_dy_m']) for c in M['stage_1']['cases']}
CASES.update({'N2': (0.60, 0.0, 0.0), 'N1': (0.60, 0.0, 0.0)})          # N2＝v2 名目（退化）、N1＝v1 名目（對照）
OUT = os.path.join(HERE, 'results', 'bottle', 'relwin_batch_offline.json')

if len(sys.argv) > 1 and sys.argv[1] == 'all':
    from concurrent.futures import ThreadPoolExecutor

    def one(k):
        p = subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, timeout=3600)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        return k, (json.loads(line[-1]) if line else dict(case=k, error=p.stderr[-2000:]))
    order = ['N1', 'N2', 'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'C1', 'C2']
    with ThreadPoolExecutor(3) as ex:
        res = dict(ex.map(one, order))
    n1, n2 = res['N1'], res['N2']
    res['degeneracy'] = dict(same_outcome=n1.get('outcome') == n2.get('outcome'), phases_v1=n1.get('phases'), phases_v2=n2.get('phases'), dx_v2=(n2.get('relwin') or {}).get('dx'),
                             t_done=[n1.get('t_done'), n2.get('t_done')])
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in order:
        r = res[k]
        print(k, {kk: r.get(kk) for kk in ('outcome', 't_adopt', 'adopt_phase', 'window', 't_done', 'rate_peak', 'du_dt_max', 'self_min_mm', 'tables_min_mm', 'stop_fails', 'error')})
    print('退化', res['degeneracy'])
    sys.exit(0)

CASE = sys.argv[1]
Z1, DXB, DYB = CASES[CASE]
sys.argv = [sys.argv[0], f'{Z1:.2f}', 'nom']
sys.path.insert(0, HERE)
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = 0.0 + DXB, 0.42 + DYB
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402  （依 argv 的 Z1 在匯入時改第一桌幾何）
import bottle_branch as BB                              # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s0b_path_check as PCK                     # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import bottle_w1_vhigh_relwin_ctrl as RW                # noqa: E402

TB = [(c_, s_) for _, c_, s_ in G.table_boxes()]
BPT = G.bottle_points()
ST = dict(n=0, worst={}, where={}, nonfinite=0, unfinished=0)
U = []


def chk(q9, T_rel, f_cmd):
    W = G.link_world(q9, f_cmd)
    P_nf = np.vstack([v for k, v in W.items() if k not in G.C.FINGERS])
    P_f = np.vstack([v for k, v in W.items() if k in G.C.FINGERS])
    out = dict(nonfinger_table=min(float(G.box_dist(P_nf, c_, s_).min()) for c_, s_ in TB), finger_table=min(float(G.box_dist(P_f, c_, s_).min()) for c_, s_ in TB),
               self=float(G.C.self_clear(q9, f_cmd)[0]), margin=BB.margin(q9[3:]))
    if T_rel is not None:
        Tb = G.C.K.fk(q9, 'link_tcp') @ T_rel
        Pb = (Tb[:3, :3] @ BPT.T).T + Tb[:3, 3]
        out['held_bottle_table'] = min(float(G.box_dist(Pb, c_, s_).min()) for c_, s_ in TB)
    return out


def make(base):
    class Ctl(base):
        def _advance_inner(self, T_tcp, close_done, q):
            if (self.blending or self.st.phase in ('PREGRASP', 'INSERT', 'CLOSE', 'LIFT')) and self.judge.stop is None and np.isfinite(q).all() and self.n_step % 10 == 0:
                ST['n'] += 1
                for e in (0, 1):
                    Q = C21.stop_traj(np.asarray(q, float), self.s, self.pipe, self.u_last, self.cfg, e)
                    if len(Q) >= 2000 // C21.SAMPLE_EVERY + 1:
                        ST['unfinished'] += 1
                    if not np.isfinite(Q).all():
                        ST['nonfinite'] += 1
                        continue
                    for qq in Q[::2]:
                        for k, v in chk(qq, self.judge.T_rel if self.judge.t_lift_ok is not None else None, self.f_cmd).items():   # 持瓶對桌只在離桌判定成立後核（之前瓶本來站在桌上）
                            key = f'e{e}:{k}'
                            if v < ST['worst'].get(key, 9.9):
                                ST['worst'][key], ST['where'][key] = v, (round(float(self.t), 3), self.st.phase)
            super()._advance_inner(T_tcp, close_done, q)

        def _solve(self, q):
            u = super()._solve(q)
            if u is not None:
                U.append((float(self.t), np.asarray(u, float).copy()))
            return u
    return Ctl


def main():
    base = TV.VH.W1PlaceController if CASE == 'N1' else RW.W1PlaceController
    TV.VH.W1PlaceController = make(base)
    c, f, r, st = TV.run()
    why = (r['stop'] or {}).get('why')
    done = bool(r['mission_done'])
    ev = [(e['t'], e.get('to') or e['what']) for e in r['events'] if e['what'] in ('phase', 'blend_start', 'segtarget_freeze', 'relwin_shift', 'close_request', 'mission_done', 'stop')]
    ph = {}
    for t_, w_ in ev:
        ph.setdefault(w_, t_)
    fz = next((e for e in r['events'] if e['what'] == 'segtarget_freeze'), None)
    seqlog = r['vhigh_seq']['log']
    adopt_ph = None
    if fz is not None:
        for t_, p_, x_, T_ in seqlog:
            if t_ <= fz['t'] + 1e-9:
                adopt_ph = p_
    mins, sm, ns, rate, marg = PCK.check(f)
    du = [float(np.abs(U[i][1][3:] - U[i - 1][1][3:]).max() / (U[i][0] - U[i - 1][0])) for i in range(1, len(U)) if 0 < U[i][0] - U[i - 1][0] <= 0.051]
    lim = dict(nonfinger_table=0.01, finger_table=0.005, held_bottle_table=0.0055, self=0.0, margin=0.05)
    fails = sorted(k for k, v in ST['worst'].items() if v < lim[k.split(':')[1]]) + (['nonfinite'] if ST['nonfinite'] else []) + (['unfinished'] if ST['unfinished'] else [])
    tbl = {k: (round(v[0] * 1e3, 1) if isinstance(v, tuple) else v) for k, v in mins.items() if ('table' in k or 'leg' in k) and 'held_bottle' not in k}
    held = {k: (round(v[0] * 1e3, 1) if isinstance(v, tuple) else v) for k, v in mins.items() if 'held_bottle' in k}
    out = dict(case=CASE, z1=Z1, bottle=[DXB, 0.42 + DYB], outcome='完整抓放' if done else f'停止 {why}', stop=r['stop'], t_adopt=None if fz is None else fz['t'], adopt_phase=adopt_ph,
               relwin=r.get('relwin'), window=None if fz is None else dict(margin=fz.get('margin')), phases={k: v for k, v in ph.items()}, t_done=ph.get('mission_done'), seq=seqlog,
               rate_peak=round(rate, 3), du_dt_max=round(max(du), 3) if du else None, self_min_mm=round(sm[0] * 1e3, 2), joint_margin=round(marg, 3),
               tables_min_mm=min((v for v in tbl.values() if isinstance(v, (int, float))), default=None), mins=tbl, held_mins=held, close_req=c.t_close_req, finite=TV.T.finite_cmds(f),
               stop_check=dict(n_ticks=ST['n'], worst={k: round(v, 5) for k, v in ST['worst'].items()}, where=ST['where'], limits=lim, nonfinite=ST['nonfinite'], unfinished=ST['unfinished']),
               stop_fails=fails)
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
