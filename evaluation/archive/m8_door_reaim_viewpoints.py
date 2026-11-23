#!/usr/bin/env python3
"""門把手「途中重瞄」靜態擷取的視點登錄（Codex 20261112 核可一次最小靜態 RGB-D 擷取）。擷取前產生並登錄，之後不改。
來源：**一個事前固定的遠距估計**——既有擷取 m8doorcap_1 的開發視點 SRC（凍結偵測器＋N-obs 估計）；候選由該估計的把手座標生成，不讀真值、不按真值重新置中。
候選（與離線 r3 同一候選族）：相機到把手 {0.35, 0.50} m × 方位 {0°, −25°, +25°} × 俯仰 {0°, 25°}＝12 個相機位姿；底盤在「SRC 起點→r3 選出的停位」連線的 50% 與 100%
  ⇒ 最多 24 個視點。每個視點：手臂 IK（影像水平軸沿桿的兩個方向都試，取關節餘裕大者，記 flip）、關節餘裕、全碰撞對靜態核對（含手指／夾爪殼對橫桿）、底盤圓盤淨空。
  不可達或碰撞者照實登錄為非 ok（擷取端會跳過），**不為湊滿數量改候選**。
分組（依相機位姿組事前切分；同一相機位姿的 50%／100% 兩個視點同組）：12 個相機位姿依固定列舉順序，偶數序＝dev、奇數序＝holdout。
這是靜態置姿擷取，不是機器人已執行接近軌跡的證據。    python3 evaluation/m8_door_reaim_viewpoints.py OUT.json"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_door_reaim_online_r3 as R3                               # noqa: E402

R2, R1, PG, DOC, E, K, IDX = R3.R2, R3.R1, R3.PG, R3.DOC, R3.E, R3.K, R3.IDX
SRC_RUN, SRC = 'm8doorcap_1', 'd1.5_x0.5_y+0_OBS'


def main():
    D = os.path.join(HERE, 'runs', SRC_RUN, 'wrist_door')
    det = json.load(open(os.path.join(D, 'door_detect_dev.json')))
    meta = json.load(open(os.path.join(D, 'meta.json')))
    reg0 = json.load(open(os.path.join(R2.WS, meta['registration'])))
    vp = {v['id']: v for v in reg0['viewpoints']}[SRC]
    FR = {f['stem']: f for f in meta['frames']}
    Kint = np.array(meta['intrinsics_readback']['K'])
    Rg, T_HG, a_H = PG.baseline_reference()
    r = [x for x in det['rows'] if x['viewpoint'] == SRC and x['cls'] == 'center_valid'][0]
    f = FR[r['stem']]
    dep = np.load(os.path.join(D, 'frames', f"{r['stem']}_depth_m.npy"))
    dep = dep[:, :, 0] if dep.ndim == 3 else dep
    est = E.estimate(dict(path=r['stem'], L1=r['L1'], L2=r['L2'], reject=None, reject_L2=None, depth=dep, K=Kint, T_cam=np.array(f['cam_T_fk']), t_obs=f['rendering_time']), 'N-obs', Rg, T_HG, a_H, query_t=f['rendering_time'])
    assert est['ok']
    T_WH = np.asarray(est['T_WH'], float)

    def T_WE_at(s):
        return np.asarray(E.OT.object_target(T_WH, T_HG, a_H, float(s))['T_WE'], float)
    base0, q_obs = np.array(vp['base'], float), np.array(vp['q_arm'], float)
    qf0 = np.zeros(9)
    qf0[:3], qf0[IDX] = base0, q_obs
    T_mount = np.linalg.inv(K.fk(qf0, 'link_eef')) @ np.array(f['cam_T_fk'])
    sel = R2.plan_viewpoint2(T_WH, T_WE_at(PG.S_PRE), T_WE_at(0.0), T_WE_at, base0, q_obs, T_mount, Kint)
    assert sel['ok'], sel
    b1, q_pre = np.array(sel['chosen']['base']), np.array(sel['_q']['q_pre'])
    T_cam_tcp = np.linalg.inv(np.linalg.inv(K.fk(qf0, 'link_tcp')) @ K.fk(qf0, 'link_eef') @ T_mount)
    db = np.r_[b1[:2] - base0[:2], R1.wrap(b1[2] - base0[2])]
    cts = R3.cam_targets(T_WH)
    poses = {}
    for ct in cts:
        poses.setdefault((ct['d'], ct['el'], ct['az']), []).append(ct)
    out = []
    for gi, (key, pair) in enumerate(poses.items()):
        d, el, az = key
        grp = 'dev' if gi % 2 == 0 else 'holdout'
        for lam in (0.5, 1.0):
            bl = base0 + lam * db
            vid = f"reaim_d{d:.2f}_az{az:+.0f}_el{el:.0f}_lam{lam:.1f}"
            rec = dict(id=vid, group=grp, pose_group=gi, cam_dist=d, azi_deg=az, elev_deg=el, lam=lam, base=[round(float(v), 6) for v in bl], arm='REAIM', source=SRC)
            best = None
            for ct in pair:
                ql = PG.ik(bl, q_obs if lam < 0.9 else q_pre[IDX], ct['T'] @ T_cam_tcp, extra_seeds=True)
                if ql is None:
                    continue
                m = DOC.jmargin(ql[IDX])[0]
                if best is None or m > best[0]:
                    best = (m, ql, ct)
            if best is None:
                out.append(dict(rec, status='ik_fail', q_arm=None))
                continue
            m, ql, ct = best
            cc = R3.cfg_clear_full(ql, T_WH)
            sv = R3.see(K.fk(ql, 'link_eef') @ T_mount, Kint, T_WH, ql, 'link_eef')
            status = 'ok'
            if m < R1.LIM_M:
                status = 'limit_fail'
            elif not cc['ok']:
                status = 'config_collision'
            elif not sv['ok']:
                status = 'geom_not_visible'
            Tc = K.fk(ql, 'link_eef') @ T_mount
            out.append(dict(rec, status=status, q_arm=[round(float(v), 6) for v in ql[IDX]], flip=bool(ct['flip']), joint_margin=round(float(m), 4),
                            clear=dict(env=round(cc['env'], 4), base=round(cc['base'], 4), self=round(cc['self'], 4), bar_tool=round(cc['bar_tool'], 4)),
                            cam_pos_fk=[round(float(v), 4) for v in Tc[:3, 3]], cam_z_fk=[round(float(v), 4) for v in Tc[:3, 2]], cam_x_fk=[round(float(v), 4) for v in Tc[:3, 0]],
                            geom_vis=dict(ok=sv['ok'], why=sv['why'], center_uv=sv['pts']['center']['uv'], center_dist=sv['pts']['center']['dist'])))
    cnt = {}
    for v in out:
        cnt[v['status']] = cnt.get(v['status'], 0) + 1
    reg = dict(spec='m8_door_reaim_viewpoints/1', codex='20261112 門把手途中重瞄 r3 裁定：核可一次最小靜態 RGB-D 擷取', source=dict(run=SRC_RUN, viewpoint=SRC, stem=r['stem'], T_WH_est=T_WH.tolist(), note='凍結偵測器＋N-obs 估計；不讀真值'),
               standoff=dict(base=b1.round(5).tolist(), base0=base0.tolist()), frames_per_viewpoint=3, groups=dict(rule='12 個相機位姿依列舉順序，偶數序 dev、奇數序 holdout；同位姿的 50%／100% 同組'),
               counts=cnt, n_ok_by_group={g: sum(1 for v in out if v['status'] == 'ok' and v['group'] == g) for g in ('dev', 'holdout')}, n_ok_by_lam={str(l): sum(1 for v in out if v['status'] == 'ok' and v['lam'] == l) for l in (0.5, 1.0)},
               note='靜態置姿擷取；不是機器人已執行接近軌跡的證據；非 ok 視點照實保留、擷取端跳過', viewpoints=out)
    json.dump(reg, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: reg[k] for k in ('counts', 'n_ok_by_group', 'n_ok_by_lam', 'standoff')}, ensure_ascii=False))
    for v in out:
        print(v['id'], v['group'], v['status'], v.get('flip'), v.get('joint_margin'), v.get('clear'))


if __name__ == '__main__':
    main()
