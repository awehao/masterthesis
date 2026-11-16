#!/usr/bin/env python3
"""主線 B 影格包絡（Codex 20261117_033806）：既有 Isaac 遠距趟（far_capture_1–3、far_door_1–9）手腕相機**逐格**分類與事後誤差。
far_door_* 沒有 truth.json：這些趟的事後評估（eval_only）用把手 prim 即時真值；門角紀錄（far_log door_deg）整趟 <1e-6°，
且以 far_capture 的 T_handle_start 重算 range_m 與紀錄值差 ≤0.5 mm（四捨五入）⇒ 逐格核對後才把該格把手真值定為 T_handle_start；核對不過的格標缺失、不假補。
每格（含失敗格）：
  幾何量：相機到把手中心距離、斜視角（相機→把手方向與門面法向夾角）、把手中心離光軸角；
  幾何可見＝兩端（中心 ±100 mm，資產橫桿長 200 mm）與中心投影都在影像內（邊距 5 px，同偵測器端點條件）且在相機前方；
  夾爪遮擋＝相機→兩端／中心的射線 5 mm 內有手臂／夾爪碰撞點（離把手點 2 cm 內的不算）；
  實際估計有效＝凍結偵測＋幾何估計回傳有效（frames_meta valid；分類照偵測器 cls／why）。
盲段只能由「實際估計有效」起算；包絡＝實際有效格在（距離、斜視、離軸）上的範圍與樣本數。姿態誤差（理想里程下 T_base_handle 換世界）換算接觸位置影響：
端點力臂 100 mm、抓持接觸點力臂＝|T_HG 平移|（夾爪參考點到把手中心）。只讀既有紀錄、不跑模擬。
    python3 door_vis_frames_all.py OUT.json"""
import glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_motm_grasp_ctrl as G
import door_wbopt as W
K, DOC = G.K, W.DOC
HALF = 0.100
TH0 = np.array(json.load(open(os.path.join(HERE, 'runs', 'm8house_far_capture_3', 'truth.json')))['T_handle_start'], float)
NRM = np.array([0.0, 1.0, 0.0])                                      # 門面法向（機器人側 y<0 ⇒ 看向門＝+y）
ARM = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2']
ARM = [n for n in ARM if n in DOC.CL]


def occluded(cpos, pts, q9):
    for X in pts:
        seg = X - cpos; L = np.linalg.norm(seg); dv = seg / L
        for n in ARM:
            Tn = K.fk(q9, n)
            C = (Tn[:3, :3] @ DOC.CL[n].T).T + Tn[:3, 3]
            v = C - cpos; tt = np.clip(v @ dv, 0.0, L)
            dd = np.linalg.norm(v - tt[:, None] * dv, axis=1)
            if np.any((dd < 0.005) & (np.linalg.norm(C - X, axis=1) >= 0.02) & (tt > 0.02) & (tt < L - 0.01)):
                return n
    return None


rows, miss = [], []
for rd in sorted(glob.glob(os.path.join(HERE, 'runs', 'm8house_far_capture_[0-9]*'))) + sorted(glob.glob(os.path.join(HERE, 'runs', 'm8house_far_door_*'))):
    fm = os.path.join(rd, 'frames_meta.json')
    if not os.path.exists(fm):
        miss.append(dict(run=os.path.basename(rd), why='no_frames_meta')); continue
    M = json.load(open(fm)); Kc = np.array(M['K'], float); W_, H_ = 640, 480
    tj = os.path.join(rd, 'truth.json')
    Th = np.array(json.load(open(tj))['T_handle_start'], float) if os.path.exists(tj) else TH0
    door_ok = True
    lg = os.path.join(rd, 'far_log.npz')
    if not os.path.exists(tj):
        if not os.path.exists(lg):
            miss.append(dict(run=os.path.basename(rd), why='no_truth_no_door_log')); continue
        door_ok = float(np.abs(np.load(lg)['L'][:, -1]).max()) < 0.1
    for f in M['frames']:
        ev = f.get('eval_only') or {}
        Tc = np.array(f['cam_T_fk'], float); c = Tc[:3, 3]
        rng = float(np.linalg.norm(c - Th[:3, 3]))
        if ev.get('range_m') is not None and (abs(rng - float(ev['range_m'])) > 0.001 or not door_ok):
            miss.append(dict(run=os.path.basename(rd), seq=f['seq'], why='truth_not_verified')); continue
        P = [Th[:3, 3], Th[:3, 3] + Th[:3, 0] * HALF, Th[:3, 3] - Th[:3, 0] * HALF]
        Ti = np.linalg.inv(Tc); inimg = True
        for X in P:
            pc = (Ti @ np.r_[X, 1.0])[:3]
            if pc[2] < 0.05:
                inimg = False; break
            u, v = (Kc @ (pc / pc[2]))[:2]
            if not (5 < u < W_ - 5 and 5 < v < H_ - 5):
                inimg = False
        dv = (Th[:3, 3] - c) / rng
        obl = math.degrees(math.acos(max(-1.0, min(1.0, float(dv @ NRM)))))
        offax = math.degrees(math.acos(max(-1.0, min(1.0, float(dv @ Tc[:3, 2])))))
        q9 = np.asarray(f['q'], float)
        occ = occluded(c, P, q9) if inimg else None
        rot = None
        if f.get('valid') and f.get('T_base_handle') is not None:
            b = q9[:3]; cb, sb = math.cos(b[2]), math.sin(b[2])
            Tb = np.eye(4); Tb[:2, :2] = [[cb, -sb], [sb, cb]]; Tb[:2, 3] = b[:2]
            dR = (Tb @ np.array(f['T_base_handle'], float))[:3, :3].T @ Th[:3, :3]
            rot = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
        info = f.get('info') or {}
        rows.append(dict(run=os.path.basename(rd), seq=f['seq'], t=f['t_obs'], valid=bool(f.get('valid')), cls=info.get('cls'), why=info.get('why'), range=rng, oblique=obl, offaxis=offax,
                         geom_vis=inimg, occ=occ, err_mm=ev.get('err_pos_mm'), rot_deg=rot))
    print(os.path.basename(rd), 'rows', sum(1 for r in rows if r['run'] == os.path.basename(rd)), flush=True)


def cat(r):
    if r['valid']:
        return 'valid'
    if not r['geom_vis']:
        return 'not_geom_visible'
    if r['occ']:
        return 'occluded'
    return 'visible_but_rejected:' + str(r['why'])


RB = [(0.2, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4.0)]
OB = [(0, 15), (15, 30), (30, 45), (45, 90)]
tab = []
for lo, hi in RB:
    for olo, ohi in OB:
        S = [r for r in rows if lo <= r['range'] < hi and olo <= r['oblique'] < ohi]
        if not S:
            continue
        cats = {}
        for r in S:
            cats[cat(r)] = cats.get(cat(r), 0) + 1
        V = [r for r in S if r['valid']]
        gv = [r for r in S if r['geom_vis'] and not r['occ']]
        e = np.array([r['err_mm'] for r in V if r['err_mm'] is not None]); ro = np.array([r['rot_deg'] for r in V if r['rot_deg'] is not None])
        tab.append(dict(range=[lo, hi], oblique=[olo, ohi], n=len(S), cats=cats, valid_given_geomvis=None if not gv else round(sum(r['valid'] for r in gv) / len(gv), 3),
                        n_valid=len(V), offaxis_valid=None if not V else [round(min(r['offaxis'] for r in V), 1), round(max(r['offaxis'] for r in V), 1)],
                        pos_mm=None if not len(e) else dict(p50=round(float(np.median(e)), 2), p95=round(float(np.percentile(e, 95)), 2), max=round(float(e.max()), 2)),
                        rot_deg=None if not len(ro) else dict(p50=round(float(np.median(ro)), 3), p95=round(float(np.percentile(ro, 95)), 3), max=round(float(ro.max()), 3))))
        print(tab[-1], flush=True)
V = [r for r in rows if r['valid']]
lev = float(np.linalg.norm(np.asarray(G.T_HG, float)[:3, 3])) if hasattr(G, 'T_HG') else None
near = [r for r in V if r['range'] < 1.2]
rmax = max((r['rot_deg'] for r in near if r['rot_deg'] is not None), default=None)
emax = max((r['err_mm'] for r in near if r['err_mm'] is not None), default=None)
impact = None if rmax is None else dict(rot_max_deg=round(rmax, 3), pos_max_mm=emax, end_lever_m=HALF, end_mm=round(math.radians(rmax) * HALF * 1e3, 2), end_total_mm=round(emax + math.radians(rmax) * HALF * 1e3, 2),
                                        contact_lever_m=None if lev is None else round(lev, 4), contact_total_mm=None if lev is None else round(emax + math.radians(rmax) * lev * 1e3, 2))
env = None if not near else dict(range=[round(min(r['range'] for r in near), 3), round(max(r['range'] for r in near), 3)], oblique=[round(min(r['oblique'] for r in near), 1), round(max(r['oblique'] for r in near), 1)],
                                 offaxis=[round(min(r['offaxis'] for r in near), 1), round(max(r['offaxis'] for r in near), 1)], n=len(near))
print('near(<1.2 m) valid envelope', env); print('impact', impact); print('missing', len(miss), {m['why'] for m in miss})
json.dump(dict(note='既有遠距 Isaac 趟逐格分類（幾何可見／夾爪遮擋／偵測器實際有效）與單格事後誤差；far_door 真值經逐格 range 與門角核對；理想里程；單格、不含融合；只讀紀錄',
               half_len_m=HALF, n=len(rows), n_valid=len(V), runs=sorted({r['run'] for r in rows}), near_valid_envelope=env, contact_impact=impact, bins=tab, missing=miss,
               frames=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]), open(sys.argv[1], 'w'), ensure_ascii=False, indent=0)
