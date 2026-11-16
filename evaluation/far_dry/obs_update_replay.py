#!/usr/bin/env python3
"""近距觀察段連續視覺更新（N3）的離線重播：把已錄的 Isaac 趟（每拍實際位姿／設定點／在途命令／放行命令＋逐格實際影像估計）
送進「真正的更新入口」（merge_node.worker 的 tick，allow_update），與不更新的同一重播對照。
開迴路重播：已錄狀態不會對改變後的命令反應——驗的是入口的收／拒、參考的連續性、可達性、兩種停止核對、命令差異與耗時；不是閉迴路成效。
    python3 evaluation/far_dry/obs_update_replay.py RUN_DIR OUT.json"""
import json, multiprocessing as mp, os, sys
from collections import Counter
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'far_dry')); sys.path.insert(0, HERE)
import numpy as np
import merge_node as MN

RUN = sys.argv[1]
J = json.load(open(os.path.join(RUN, 'ros_merge.json')))
Z = np.load(os.path.join(RUN, 'ros_merge.npz')); R, c = Z['R'], [str(x) for x in Z['cols']]; i = c.index
Lz = np.load(os.path.join(RUN, 'log.npz')); L, lc = Lz['L'], [str(x) for x in Lz['cols']]; iu = lc.index('uin_x')
FM = json.load(open(os.path.join(RUN, 'frames_meta.json')))
TR = json.load(open(os.path.join(RUN, 'truth.json')))
obs = [dict(seq=f['seq'], t_obs=f['t_obs'], T=(MN._T3(f['odom_base_at_capture']) @ np.array(f['T_base_handle'])).tolist()) for f in FM['frames'] if f['valid']]
an = J['anchor']
arm, st = R[:, i('arm(0HOLD,1PREP,2TRACK)')], R[:, i('stop')]
rows = np.where((arm == 2) & (st == 0))[0]
k_plan = int(np.argmin(np.abs(R[:, 0] - an['t'])))


def run(allow):
    ctx = mp.get_context('fork'); pc, cc = ctx.Pipe(); wp = ctx.Process(target=MN.worker, args=(cc,)); wp.start(); pc.recv()
    pc.send(('prior', dict(ver=1))); pc.recv()
    pc.send(('anchor', dict(ver=2, T_odom_base=MN._T3(an['odom_base']).tolist(), T_base_handle=an['T_base_handle'], T_mo=an['map_odom'], seq=an['seq'], t_obs=an['t_obs']))); pc.recv()
    pc.send(('plan', dict(ver=3, q=R[k_plan, 2:11].tolist(), budget=8.0))); _, _, pl = pc.recv()
    out = []
    for r_ in rows:
        t = float(R[r_, 0])
        k = int(round(R[r_, i('t_world')] / 0.01))
        k = int(np.argmin(np.abs(L[:, 0] - R[r_, i('t_world')])))
        pipe = L[k - 7:k + 1, iu:iu + 9]
        s = R[r_, 11:17]
        cand = [o for o in obs if o['t_obs'] <= t]
        msg = dict(ver=10, mode='TRACK', t=t, q=R[r_, 2:11].tolist(), pipe=pipe.tolist(), rel=R[r_, i('rel_x'):i('rel_x') + 3].tolist(), s=s.tolist(), s_end=(s + 0.01 * pipe[:, 3:].sum(axis=0)).tolist(), judge=False, judge_refs=[],
                   v_meas=[0, 0], T_mo=R[r_, i('mo_x'):i('mo_x') + 3].tolist(), u_last=[0] * 9, u_prev=[0] * 9, obs=cand[-1] if cand else None, allow_update=allow)
        pc.send(('tick', msg)); tag, _, res = pc.recv()
        if tag != 'tick':
            out.append(dict(t=t, error=str(res)[:200])); continue
        sc = res.get('stop')
        out.append(dict(t=t, along=float(R[r_, i('along')]), ref=res.get('ref'), u=res['track']['u'], track_ok=res['track']['ok'], err_now=res['track']['err'], sigma=res['track']['sigma'], disposition=MN.stop_disposition('tick', sc), stop_ms=None if sc is None else sc['ms'],
                        stop_margin=None if sc is None else sc['margin_after_interval_bound'], u_recorded=R[r_, i('ua0'):i('ua0') + 6].tolist()))
    pc.send(None)
    return pl, out


plA, A = run(False)
plB, B = run(True)
p_truth = np.array(TR['T_handle_start'])[:3, 3]
p_fix = (MN._T3(an['odom_base']) @ np.array(an['T_base_handle']))[:3, 3]
okB = [b for b in B if 'error' not in b]
st_new = Counter((b['ref']['status'].split(':')[0] + (':' + b['ref']['status'].split(':')[1] if b['ref']['status'].startswith('reject') else '')) for b in okB if b['ref'].get('obs_seq') is not None)
commits = [b for b in okB if b['ref'].get('committed')]
rej_steps = Counter(b['ref'].get('step_rejected') for b in okB if b['ref'].get('committed') is False)
refp = np.array([b['ref']['ref_p'] for b in okB])
uA, uB = np.array([a_['u'] for a_ in A if 'error' not in a_]), np.array([b['u'] for b in okB])
n = min(len(uA), len(uB))
summ = dict(source=RUN, note='開迴路重播（已錄狀態不對新命令反應）；真值對照只作事後評估', plan_same=dict(recorded=dict(standoff=J['plan']['pg']['standoff'], along=J['plan']['pg']['along']), replay=dict(standoff=plB['pg']['standoff'], along=plB['pg']['along'])),
            n_track_ticks=len(rows), n_valid_obs_total=len(obs), n_obs_entered=sum(st_new.values()), entry_status=dict(st_new), n_ref_steps_committed=len(commits), ref_ver_final=okB[-1]['ref']['ver'] if okB else None, step_rejected=dict(rej_steps),
            ref_step_mm_max=max((b['ref']['step_mm'] for b in commits), default=None), update_params=MN.UPD,
            ref_shift_total_mm=[round(float(x) * 1e3, 2) for x in (refp[-1] - p_fix)] if len(refp) else None,
            eval_only=dict(fixed_ref_err_mm=round(float(np.linalg.norm(p_fix - p_truth)) * 1e3, 2), updated_ref_err_mm_final=round(float(np.linalg.norm(refp[-1] - p_truth)) * 1e3, 2) if len(refp) else None, updated_ref_err_mm_max=round(float(np.linalg.norm(refp - p_truth, axis=1).max()) * 1e3, 2) if len(refp) else None,
                           obs_err_mm=dict(median=round(float(np.median([np.linalg.norm(np.array(o['T'])[:3, 3] - p_truth) for o in obs if o['t_obs'] >= R[rows[0], 0] - 0.5])) * 1e3, 2))),
            induced_joint_target_step_per_tick=dict(max_rad=round(float(np.abs(np.diff((uB[:n] - uA[:n]) * 0.05, axis=0)).max()), 5), equivalent_extra_rate_rad_s=round(float(np.abs(np.diff((uB[:n] - uA[:n]) * 0.05, axis=0)).max()) / 0.05, 4),
                                                    note='(更新版命令−無更新版命令)·dt 的逐拍變化＝更新每拍讓關節目標多移動的量；閉迴路下設定點會跟上，命令的額外量是這個量級，不是下面的累積差'),
            continuity=dict(open_loop_note='開迴路下已錄設定點不跟新目標 ⇒ 累積的目標偏移全部除以 dt 出現在命令差，這不是每拍的不連續', arm_cmd_diff_vs_no_update_max=round(float(np.abs(uB[:n] - uA[:n]).max()), 4), arm_cmd_tick_to_tick_change_max=dict(no_update=round(float(np.abs(np.diff(uA, axis=0)).max()), 4), update=round(float(np.abs(np.diff(uB, axis=0)).max()), 4)), unit='rad/s'),
            tracker_ok=dict(no_update=sum(1 for a_ in A if a_.get('track_ok')), update=sum(1 for b in okB if b['track_ok'])), sigma_min=dict(no_update=round(min(a_['sigma'] for a_ in A if a_.get('sigma') is not None), 4), update=round(min(b['sigma'] for b in okB if b.get('sigma') is not None), 4)),
            err_now_max=dict(no_update=round(max(a_['err_now'] for a_ in A if a_.get('err_now') is not None), 4), update=round(max(b['err_now'] for b in okB if b.get('err_now') is not None), 4), note='開迴路：已錄關節角對新目標的差，更新後目標移動會直接反映在這個量'),
            stop_disposition=dict(no_update=dict(Counter(a_['disposition'] for a_ in A if 'error' not in a_)), update=dict(Counter(b['disposition'] for b in okB))), stop_ms_max=dict(no_update=max(a_['stop_ms'] for a_ in A if a_.get('stop_ms')), update=max(b['stop_ms'] for b in okB if b.get('stop_ms'))),
            stop_margin_self_min=dict(update=min(b['stop_margin']['self'] for b in okB if b.get('stop_margin'))), errors=dict(no_update=sum(1 for a_ in A if 'error' in a_), update=len(B) - len(okB)))
json.dump(dict(summary=summ, ticks_update=B, ticks_no_update=A), open(sys.argv[2], 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
