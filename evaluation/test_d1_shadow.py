#!/usr/bin/env python3
"""D1 S4 離線測試（Codex reviews/20261005_144036_reply.md：延遲影格匹配、配對淘汰、覆蓋對帳、單位／座標、等價）。

    python3 evaluation/test_d1_shadow.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d1_shadow_core import (OPT, WORLD, LatestSlot, PairBuffer, PoseHistory,  # noqa: E402
                            check_contract, quat_to_T)

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


DT = 0.01
# ---------------------------------------------------------------- PoseHistory
h = PoseHistory(DT, horizon_s=1.0)
for k in range(8000, 8010):                      # t ≈ 80 s：float32 精度約 7.6e-6 s
    h.add(k * DT, [k, 0, 0], [1, 0, 0, 0], truth=[0, 0, 0], step=k)
e, why = h.match(float(np.float32(8007 * DT)))   # 延遲 2 步、float32 時間
check('delayed_frame_matches_unique_step', why is None and e['step'] == 8007, (why, e and e['step']))
e, why = h.match(8009 * DT)
check('current_step_matches', why is None and e['step'] == 8009, why)
e, why = h.match(8009 * DT + 0.5 * DT)
check('between_steps_rejected', e is None and why == 'no_pose_at_render_time', why)
e, why = h.match(70.0)
check('older_than_history_rejected', e is None and why == 'older_than_history', why)
e, why = h.match(None)
check('missing_rendering_time_rejected', e is None and why == 'no_rendering_time', why)
e, why = h.match(float('nan'))
check('nan_rendering_time_rejected', e is None and why == 'no_rendering_time', why)
h2 = PoseHistory(DT, horizon_s=1.0)
for k in range(500):
    h2.add(k * DT, [0, 0, 0], [1, 0, 0, 0])
check('history_bounded', len(h2.buf) == 100, len(h2.buf))
h3 = PoseHistory(DT)
h3.add(1.0, [0, 0, 0], [1, 0, 0, 0])
h3.add(1.0, [1, 0, 0], [1, 0, 0, 0])
check('duplicate_time_ambiguous', h3.match(1.0)[1] == 'ambiguous_pose_match')

# ---------------------------------------------------------------- PairBuffer
pb = PairBuffer(cap=3, timeout_s=1.0)
k1 = (10, 0)
done = None
for i, part in enumerate(('meta', 'pose', 'depth', 'info')):  # 任意順序
    done, ev = pb.put(k1, part, part, now=100.0 + 0.01 * i)
check('pair_completes_any_order', done is not None and set(done) == {'depth', 'info', 'pose', 'meta'} and not pb.d)
pb = PairBuffer(cap=3, timeout_s=1.0)
pb.put((1, 0), 'depth', 'd', now=0.0)
ev = pb.expire(now=1.5)
check('pair_timeout_evicts_with_parts', ev == [((1, 0), 'pair_timeout', ['depth'])], ev)
pb = PairBuffer(cap=3, timeout_s=10.0)
for s in range(3):
    pb.put((s, 0), 'depth', 'd', now=0.0)
_, ev = pb.put((3, 0), 'depth', 'd', now=0.1)
check('pair_overflow_evicts_oldest', ev == [((0, 0), 'pair_overflow', ['depth'])] and len(pb.d) == 3, ev)
_, ev = pb.put((3, 0), 'depth', 'd2', now=0.2)
check('duplicate_part_reported', any(w == 'duplicate_depth' for _, w, _p in ev), ev)
left = pb.drain()
check('drain_reports_unpaired', len(left) == 3 and all(w == 'shutdown_unpaired' for _, w, _p in left), left)

# ---------------------------------------------------------------- LatestSlot
sl = LatestSlot()
check('slot_first_put_no_overwrite', sl.put({'n': 1, 'stamp': [1, 0]}) is None)
old = sl.put({'n': 2, 'stamp': [1, 200000000]})
check('slot_overwrite_returns_old_identity', old == {'n': 1, 'stamp': [1, 0]}, old)
it = sl.take(timeout=0.1)
check('slot_take_newest_stamp_unchanged', it == {'n': 2, 'stamp': [1, 200000000]}, it)
check('slot_empty_after_take', sl.take(timeout=0.05) is None)
sl.put({'n': 3, 'stamp': [2, 0]})
check('slot_drain_returns_unprocessed', sl.drain() == {'n': 3, 'stamp': [2, 0]})

# ---------------------------------------------------------------- 契約（單位／座標）
W, H = 640, 480
dep = np.full((H, W), 0.75, np.float32)
dep[0, 0] = np.inf
dep[0, 1] = np.nan
K = [465.6, 0, 320, 0, 465.6, 240, 0, 0, 1]


def msgs(enc='32FC1', fid=OPT, pfid=WORLD, w=W, hh=H, big=0, data=None, q=(1, 0, 0, 0)):
    d = {'encoding': enc, 'frame_id': fid, 'width': w, 'height': hh, 'is_bigendian': big,
         'step': w * 4, 'data': dep.astype('<f4').tobytes() if data is None else data}
    return d, {'frame_id': fid, 'width': w, 'height': hh, 'k': K}, \
        {'frame_id': pfid, 'pos': [1.0, 2.0, 0.5], 'quat_wxyz': list(q)}


dd, KK, TT, why = check_contract(*msgs())
check('contract_valid_32FC1_m_roundtrip', why is None and np.array_equal(dd[1:], dep[1:]) and np.isinf(dd[0, 0])
      and np.isnan(dd[0, 1]) and dd.dtype == np.float32, why)
check('contract_pose_translation', why is None and np.allclose(TT[:3, 3], [1, 2, 0.5]) and np.allclose(TT[:3, :3], np.eye(3)))
check('contract_16UC1_rejected', check_contract(*msgs(enc='16UC1'))[3] == 'depth_encoding_16UC1')
check('contract_optical_frame_required', check_contract(*msgs(fid='base_link'))[3] == 'optical_frame_id_mismatch')
check('contract_pose_world_frame_required', check_contract(*msgs(pfid=OPT))[3] == f'pose_frame_{OPT}')
check('contract_size_mismatch', check_contract(*msgs(w=320))[3] in ('size_mismatch',))
check('contract_bigendian_rejected', check_contract(*msgs(big=1))[3] == 'bigendian')
check('contract_length_mismatch', check_contract(*msgs(data=b'\0' * 10))[3] == 'step_or_length_mismatch')
check('contract_bad_quat', check_contract(*msgs(q=(0, 0, 0, 0)))[3] == 'bad_pose')
_d, _i, _p = msgs()
_i_nan = dict(_i, k=[465.6, 0, float('nan'), 0, 465.6, 240, 0, 0, 1])      # cx = NaN（Codex 反例）
check('contract_intrinsics_nan_rejected', check_contract(_d, _i_nan, _p)[3] == 'bad_intrinsics')
_i_len = dict(_i, k=[465.6, 0, 320, 0, 465.6, 240, 0, 0])                    # K 長度 8
check('contract_intrinsics_wrong_length_rejected', check_contract(_d, _i_len, _p)[3] == 'bad_intrinsics')
_i_none = dict(_i, k=None)
check('contract_intrinsics_none_rejected', check_contract(_d, _i_none, _p)[3] == 'bad_intrinsics')
_p_nan = dict(_p, pos=[float('nan'), 0.0, 0.0])
check('contract_pose_nan_rejected', check_contract(_d, _i, _p_nan)[3] == 'bad_pose')
_p_len = dict(_p, pos=[1.0, 2.0])
check('contract_pose_wrong_length_rejected', check_contract(_d, _i, _p_len)[3] == 'bad_pose')
# 四元數 → 旋轉：繞 z 90°
T = quat_to_T([0, 0, 0], [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
check('quat_to_T_z90', np.allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0]))

# ---------------------------------------------------------------- 等價：S3 開發集經線上契約路徑 ⇒ 與 rev2 逐格相同
import d1_handle_detect as D1                     # noqa: E402
from wrist_v0_capture import wxyz_to_R            # noqa: E402

V = os.path.join(HERE, 'runs', 'd1_dev_f02P', 'wrist_v0')
if os.path.exists(os.path.join(V, 'd1_detect_rev2.json')):
    m = json.load(open(os.path.join(V, 'meta.json')))
    ref = {r['n']: r for r in json.load(open(os.path.join(V, 'd1_detect_rev2.json')))['rows']}
    Km = np.array(m['intrinsics_readback']['K'])
    rng = np.random.default_rng(0)
    bad = []
    for f in m['frames']:
        d0 = np.load(os.path.join(V, 'frames', f'f{f["n"]:02d}_depth_m.npy'))
        d0 = d0[:, :, 0] if d0.ndim == 3 else d0
        dmsg = {'encoding': '32FC1', 'frame_id': OPT, 'width': W, 'height': H, 'is_bigendian': 0,
                'step': W * 4, 'data': d0.astype('<f4').tobytes()}
        dd, KK, TT, why = check_contract(dmsg, {'frame_id': OPT, 'width': W, 'height': H,
                                                'k': list(Km.reshape(-1))},
                                         {'frame_id': WORLD, 'pos': f['cam_pos_world'],
                                          'quat_wxyz': f['cam_quat_wxyz_world']})
        if why is not None:
            bad.append((f['n'], why))
            continue
        Tm = D1.cam_T(f)
        if not np.allclose(TT, Tm, atol=1e-9):
            bad.append((f['n'], 'pose_T_differs'))
        det, _ = D1.detect(dd, KK, TT, rng)
        r = ref[f['n']]
        if (det['reject'], det['reject_L2'], det['L1'], det['L2']) != (r['reject'], r['reject_L2'], r['L1'], r['L2']):
            bad.append((f['n'], 'detect_differs'))
    check('equivalence_dev_frames_via_contract_path_87', not bad and len(m['frames']) == 87, bad[:5])
else:
    check('equivalence_dev_frames_via_contract_path_87', False, '缺 d1_detect_rev2.json')

print(f"{len(fails)} 失敗")
sys.exit(1 if fails else 0)
