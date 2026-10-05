#!/usr/bin/env python3
"""d1_s4_audit.py 的反例測試（合成紀錄，不需 ROS／模擬器）；依 Codex reviews/20261005_145312_reply.md。

    python3 evaluation/test_d1_s4_audit.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
fails = []
K = [465.6, 0, 320, 0, 465.6, 240, 0, 0, 1]
T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
HC = [0.0, 0.0, 0.7]                                   # 相機在原點朝 +z，把手中心 0.7 m（落在 0.4–1.0 m 箱）


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


def det(l2=True):
    return {'L0': None, 'L1': None, 'L2': ({'center': [0.001, 0.0, 0.7]} if l2 else None),
            'reject': None if l2 else 'no_candidate', 'reject_L2': None}


def build(frames, node_events, counters, worker_alive=False, in_progress=None, rejected=0,
          truth_skip=(), thermal=None, align=100.0, summ_drop=(), fault=()):
    """frames：來源已發布影格 n 的清單（t = n·0.2）。node_events：(ev, n, extra) 依序。"""
    R = tempfile.mkdtemp(prefix='s4audit_')
    os.makedirs(os.path.join(R, 'wrist_live'))
    os.makedirs(os.path.join(R, 'd1_shadow'))
    st = {n: [int(n * 0.2), int(round((n * 0.2 % 1) * 1e9))] for n in frames}
    with open(os.path.join(R, 'wrist_live', 'capture.jsonl'), 'w') as f:
        for i in range(rejected):
            f.write(json.dumps({'ev': 'rejected', 'why': 'no_pose_at_render_time', 'rendering_time': 0.05 + i}) + '\n')
        for n in frames:
            f.write(json.dumps({'ev': 'published', 'n': n, 'stamp': st[n], 'rendering_time': n * 0.2,
                                'read_minus_render_s': 0.0}) + '\n')
    with open(os.path.join(R, 'wrist_live', 'truth.jsonl'), 'w') as f:
        for n in frames:
            if n in truth_skip:
                continue
            f.write(json.dumps({'n': n, 'stamp': st[n], 't_cap': n * 0.2, 'handle_center_world_at_capture': HC,
                                'cam_pos_world': [0, 0, 0], 'cam_quat_wxyz_world': [1, 0, 0, 0]}) + '\n')
    json.dump({'intrinsics_readback_K': [K[0:3], K[3:6], K[6:9]]}, open(os.path.join(R, 'wrist_live', 'meta.json'), 'w'))
    with open(os.path.join(R, 'd1_shadow', 'd1_shadow.jsonl'), 'w') as f:
        for ev, n, extra in node_events:
            rec = {'ev': ev, 'wall_t': 1000 + (n or 0) * 0.2}
            if ev == 'received':
                rec.update(stamp=st[n], part=extra, n=n if extra == 'meta' else None)
            elif ev == 'evicted':
                rec.update(stamp=st[n], why=extra, parts=['depth'])
            else:
                rec.update(n=n, stamp=st[n])
                if ev == 'processed':
                    rec.update(det=det(extra), T=T, detect_ms=100.0, out_sim_t=n * 0.2 + 0.15,
                               sim_age_s=0.15, wall_age_s=0.2, level='L2' if extra else None)
                if ev == 'contract_reject':
                    rec['why'] = extra
                if ev == 'worker_error':
                    rec.update(stage='detect', why='RuntimeError()')
            f.write(json.dumps(rec) + '\n')
    summ = {'why': 'SIGTERM', 'counters': dict(counters), 'worker_alive_at_exit': worker_alive,
            'in_progress_at_exit_n': in_progress, 'unprocessed_in_slot': 0, 'unpaired_at_exit': 0,
            'fault_inject_n': list(fault), 'detector_sha256': 'x' * 64}
    for k in summ_drop:                                  # 'worker_error' ⇒ 刪 counters 內的計數；其他 ⇒ 刪頂層欄位
        (summ['counters'].pop(k) if k == 'worker_error' else summ.pop(k))
    json.dump(summ, open(os.path.join(R, 'd1_shadow', 'd1_shadow_summary.json'), 'w'))
    json.dump({'events': [{'phase': 'ALIGN', 'sim_t': align}]}, open(os.path.join(R, 'task.json'), 'w'))
    if thermal:
        with open(os.path.join(R, 'thermal.csv'), 'w') as f:
            f.write('wall_s,iso,cpu_c,load1,gpu_util,gpu_c\n')
            for i, (c, g) in enumerate(thermal):
                f.write(f'{i},00:00:0{i},{c},1.0,{g},{g}\n')
    return R


def recv_all(n):
    return [('received', n, p) for p in ('depth', 'info', 'pose', 'meta')]


def ctr(**kw):
    c = {f'received_{p}': 0 for p in ('depth', 'info', 'pose', 'meta')}
    c.update(paired=0, processed=0, published=0, worker_error=0)
    c.update(kw)
    return c


def audit(R):
    r = subprocess.run([sys.executable, os.path.join(HERE, 'd1_s4_audit.py'), R], capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr[-1500:]
    return json.load(open(os.path.join(R, 'analysis', 'd1_s4_audit.json'))), None


# 1 Codex 反例：paired 2、processed 1、worker_alive True ⇒ 不能 PASS；第 2 格判 received_no_fate（不補成 transport_drop）
ev = recv_all(1) + recv_all(2) + [('paired', 1, None), ('paired', 2, None), ('processed', 1, True)]
A, err = audit(build([1, 2], ev, ctr(received_depth=2, received_info=2, received_pose=2, received_meta=2,
                                       paired=2, processed=1, published=1), worker_alive=True))
check('codex_case_runs', A is not None, err)
if A:
    v = A['verdict']['通路與對帳']
    check('codex_case_fails', v.startswith('FAIL'), v)
    check('codex_case_no_fate_not_transport', A['reconciliation']['per_frame_fate_counts'].get('received_no_fate') == 1
          and 'transport_drop' not in A['reconciliation']['per_frame_fate_counts'], A['reconciliation'])
    check('codex_case_reports_worker_alive', 'worker_alive_at_exit 不是明確 False（True）' in v, v)

# 2 節點完全沒收到 ⇒ transport_drop（推定）；其餘乾淨 ⇒ PASS
ev = recv_all(1) + [('paired', 1, None), ('processed', 1, True)]
A, err = audit(build([1, 2], ev, ctr(received_depth=1, received_info=1, received_pose=1, received_meta=1,
                                       paired=1, processed=1, published=1)))
check('transport_drop_inferred', A and A['reconciliation']['per_frame_fate_counts'].get('transport_drop') == 1, A and A['reconciliation'])
check('transport_drop_case_passes', A and A['verdict']['通路與對帳'] == 'PASS', A and A['verdict'])

# 3 worker_error ⇒ FAIL 並保留影格識別
ev = recv_all(1) + recv_all(2) + [('paired', 1, None), ('processed', 1, True), ('paired', 2, None), ('worker_error', 2, None)]
A, err = audit(build([1, 2], ev, ctr(received_depth=2, received_info=2, received_pose=2, received_meta=2,
                                       paired=2, processed=1, published=1, worker_error=1)))
check('worker_error_fails', A and A['verdict']['通路與對帳'].startswith('FAIL') and 'worker_error 1' in A['verdict']['通路與對帳'], A and A['verdict'])
check('worker_error_fate_attributed', A and A['reconciliation']['per_frame_fate_counts'].get('worker_error:detect') == 1, A and A['reconciliation'])

# 4 收尾時仍在處理 ⇒ FAIL
ev = recv_all(1) + recv_all(2) + [('paired', 1, None), ('processed', 1, True), ('paired', 2, None),
                                  ('shutdown_in_progress', 2, None)]
A, err = audit(build([1, 2], ev, ctr(received_depth=2, received_info=2, received_pose=2, received_meta=2,
                                       paired=2, processed=1, published=1), worker_alive=True, in_progress=2))
check('in_progress_fails', A and '仍在處理' in A['verdict']['通路與對帳'], A and A['verdict'])

# 5 計數器與事件不閉合 ⇒ FAIL
ev = recv_all(1) + [('paired', 1, None), ('processed', 1, True)]
A, err = audit(build([1], ev, ctr(received_depth=3, received_info=1, received_pose=1, received_meta=1,
                                    paired=1, processed=1, published=1)))
check('counter_mismatch_fails', A and '不閉合' in A['verdict']['通路與對帳'], A and A['verdict'])

# 6 兩種分母與連續有效窗：5 格來源（1、2 L2；3 被覆蓋；4 L2；5 傳輸遺失）＋擷取端拒絕 1 格
ev = (recv_all(1) + recv_all(2) + recv_all(3) + recv_all(4)
      + [('paired', 1, None), ('processed', 1, True), ('paired', 2, None), ('processed', 2, True),
         ('paired', 3, None), ('overwritten', 3, None), ('paired', 4, None), ('processed', 4, True)])
A, err = audit(build([1, 2, 3, 4, 5], ev, ctr(received_depth=4, received_info=4, received_pose=4, received_meta=4,
                                                paired=4, processed=3, published=3), rejected=1))
if A:
    b = A['observation']['by_distance_source_denominator'].get('0.4-1.0', {})
    check('source_denominator_includes_drops', b.get('n_source_published') == 5 and b.get('n_L2') == 3
          and b.get('L2_rate_of_source（含丟格）') == 0.6 and b.get('L2_rate_of_processed') == 1.0, b)
    w = A['observation']['L2_continuous_windows_on_source_timeline_s']
    check('overwrite_breaks_window', w == [[0.2, 0.4], [0.8, 0.8]], w)
    check('capture_reject_counted_distance_unknown', A['observation']['source_frames_distance_unknown']['capture_rejected'] == 1)
    check('denominator_case_pipeline_pass', A['verdict']['通路與對帳'] == 'PASS', A['verdict'])
    check('approach_L2_pass', A['verdict']['有效觀測（接近窗口 ≥ 1 格 L2）'] == 'PASS', A['verdict'])
else:
    check('denominator_case_runs', False, err)

# 7 缺真值 ⇒ 有效觀測證據不足（不靜默略過）
ev = recv_all(1) + [('paired', 1, None), ('processed', 1, True)]
A, err = audit(build([1], ev, ctr(received_depth=1, received_info=1, received_pose=1, received_meta=1,
                                    paired=1, processed=1, published=1), truth_skip=(1,)))
check('missing_truth_evidence_gap', A and A['verdict']['有效觀測（接近窗口 ≥ 1 格 L2）'].startswith('證據不足'), A and A['verdict'])

# 8 溫度只讀 cpu_c（gpu_c 較高時不能被取到）
ev = recv_all(1) + [('paired', 1, None), ('processed', 1, True)]
A, err = audit(build([1], ev, ctr(received_depth=1, received_info=1, received_pose=1, received_meta=1,
                                    paired=1, processed=1, published=1), thermal=[(60, 95), (72, 99)]))
check('thermal_reads_cpu_c', A and A.get('thermal_cpu_c_max') == 72.0, A and A.get('thermal_cpu_c_max'))

# 9 時間欄位：輸出頻率用 out_sim_t、擷取時序頻率另名
ev = recv_all(1) + recv_all(2) + recv_all(3) + [(e, n, x) for n in (1, 2, 3) for e, x in (('paired', None), ('processed', True))]
A, err = audit(build([1, 2, 3], ev, ctr(received_depth=3, received_info=3, received_pose=3, received_meta=3,
                                          paired=3, processed=3, published=3)))
pr = (A or {}).get('processing', {})
check('time_fields_named', 'processed_capture_cadence_hz（已處理影格的擷取時序頻率，非輸出頻率）' in pr
      and 'output_rate_sim_hz（以輸出當下模擬時間）' in pr
      and any(k.startswith('wall_age_s（模擬器讀取匹配後') for k in pr), list(pr))

# 10–13 Codex 20261005_ 複核反例：收尾必要欄位與故障事件
ok_ev = recv_all(1) + [('paired', 1, None), ('processed', 1, True)]
ok_c = dict(received_depth=1, received_info=1, received_pose=1, received_meta=1, paired=1, processed=1, published=1)
A, err = audit(build([1], ok_ev, ctr(**ok_c), summ_drop=('worker_alive_at_exit',)))
check('missing_worker_alive_fails', A and 'worker_alive_at_exit 不是明確 False' in A['verdict']['通路與對帳'], A and A['verdict'])
A, err = audit(build([1], ok_ev, ctr(**ok_c), summ_drop=('worker_error',)))
check('missing_worker_error_counter_fails', A and '缺 worker_error 計數' in A['verdict']['通路與對帳'], A and A['verdict'])
we_ev = recv_all(1) + recv_all(2) + [('paired', 1, None), ('processed', 1, True), ('paired', 2, None), ('worker_error', 2, None)]
A, err = audit(build([1, 2], we_ev, ctr(received_depth=2, received_info=2, received_pose=2, received_meta=2,
                                          paired=2, processed=1, published=1, worker_error=0)))
check('worker_error_event_with_zero_counter_fails', A and A['verdict']['通路與對帳'].startswith('FAIL')
      and 'worker_error 1 次（事件）' in A['verdict']['通路與對帳'] and '≠ 事件 1' in A['verdict']['通路與對帳'], A and A['verdict'])
ip_ev = recv_all(1) + recv_all(2) + [('paired', 1, None), ('processed', 1, True), ('paired', 2, None),
                                     ('shutdown_in_progress', 2, None)]
A, err = audit(build([1, 2], ip_ev, ctr(received_depth=2, received_info=2, received_pose=2, received_meta=2,
                                          paired=2, processed=1, published=1), worker_alive=False, in_progress=None))
check('in_progress_event_with_clean_summary_fails', A and 'shutdown_in_progress 1 次（事件）' in A['verdict']['通路與對帳'], A and A['verdict'])
# 14 功能趟的故障注入集合必須為空
A, err = audit(build([1], ok_ev, ctr(**ok_c), fault=(7,)))
check('nonempty_fault_inject_fails', A and '故障注入集合非空' in A['verdict']['通路與對帳'], A and A['verdict'])
A, err = audit(build([1], ok_ev, ctr(**ok_c), summ_drop=('fault_inject_n',)))
check('missing_fault_inject_field_fails', A and '缺 fault_inject_n' in A['verdict']['通路與對帳'], A and A['verdict'])
A, err = audit(build([1], ok_ev, ctr(**ok_c)))
check('clean_summary_still_passes', A and A['verdict']['通路與對帳'] == 'PASS', A and A['verdict'])

print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
