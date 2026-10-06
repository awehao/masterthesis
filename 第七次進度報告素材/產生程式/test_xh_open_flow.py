#!/usr/bin/env python3
"""XH3 開封流程反例測試（不渲染、不跑模型）。

1 編排器：閘門步驟失敗 ⇒ 評估步驟不執行、紀錄 failed、結束碼非 0；已開封 ⇒ 拒跑 92；SIGTERM 中斷 ⇒ 紀錄 interrupted。
2 測試評估 run()：重複影格鍵、評估中途缺 RGB ⇒ 逐格去向保留、complete False、來源鍵閉合；無正樣本的 R 資產統計填 NA。
3 擷取閘門：既有 20 條 train／dev 報告全部通過；竄改報告（雜湊不符、N 有 handle_center）⇒ 不通過。

    python3 evaluation/test_xh_open_flow.py
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import xh_capture_gate as GATE                          # noqa: E402
import xh_open_test as OPN                              # noqa: E402
import xh_test_eval as TE                               # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


# 1 編排器
tmp = tempfile.mkdtemp(prefix='xhopen_')
op = os.path.join(tmp, 'opening.json')
marker = os.path.join(tmp, 'eval_ran')
steps = [('capture', ['true']), ('capture_check', ['true']), ('capture_gate', ['false']), ('test_eval', ['touch', marker])]
rc = OPN.run(steps=steps, opening=op, cwd=tmp, check_freeze=False)
rec = json.load(open(op))
check('gate_fail_stops_before_eval', rc != 0 and not os.path.exists(marker) and rec['state'] == 'failed' and rec['failed_step'] == 'capture_gate', (rc, rec.get('state')))
check('already_opened_refuses', OPN.run(steps=steps, opening=op, cwd=tmp, check_freeze=False) == 92)
op2 = os.path.join(tmp, 'opening2.json')
code = (f"import sys; sys.path.insert(0, {HERE!r}); import xh_open_test as O; "
        f"sys.exit(O.run(steps=[('capture', ['sleep', '20'])], opening={op2!r}, cwd={tmp!r}, check_freeze=False))")
pr = subprocess.Popen([sys.executable, '-c', code])
t0 = time.time()
while not os.path.exists(op2) and time.time() - t0 < 10:
    time.sleep(0.1)
time.sleep(0.5)
pr.send_signal(signal.SIGTERM)
pr.wait(timeout=30)
r2 = json.load(open(op2))
check('sigterm_records_interrupted', pr.returncode != 0 and r2['state'] == 'interrupted', (pr.returncode, r2.get('state')))

# 2 測試評估 run()：合成 R_x25_150 擷取（相機朝外 ⇒ 無正樣本）
reg = yaml.safe_load(open(os.path.join(HERE, 'results', 'vision', 'XH1_registration.yaml')))
assets = {x['id']: x for x in reg['assets']}
root = os.path.join(tmp, 'runs')
D = os.path.join(root, 'xh_R_x25_150_A_front', 'wrist_v0')
os.makedirs(os.path.join(D, 'frames'))
K = [[465.6, 0, 320.0], [0, 465.6, 240.0], [0, 0, 1.0]]
fr = {'cam_pos_world': [0.0, 0.0, 0.6], 'cam_quat_wxyz_world': [1.0, 0.0, 0.0, 0.0], 'handle_center_world_at_capture': [0.0, 1.165, 0.55]}
meta = {'intrinsics_readback': {'K': K}, 'frames': [{'n': 1, **fr}, {'n': 1, **fr}, {'n': 2, **fr}]}
json.dump(meta, open(os.path.join(D, 'meta.json'), 'w'))
json.dump({'target_kind': 'R_bar'}, open(os.path.join(D, 'truth.json'), 'w'))
for n in (1, 2):
    np.save(os.path.join(D, 'frames', f'f{n:02d}_depth_m.npy'), np.full((480, 640), 2.0, np.float32))
Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(os.path.join(D, 'frames', 'f01_rgb.png'))   # f02 缺 RGB
res = TE.run(lambda img: (False, np.zeros(img.shape[:2], bool)), assets, ['R_x25_150'], root,
             os.path.join(WS, 'src/my_omnibot_description/config/xh'))
fz = res['fates']
check('duplicate_key_recorded', any(i['why'] == 'duplicate_frame_key' for i in res['evidence_issues']), res['evidence_issues'])
check('missing_rgb_recorded_not_crash', fz.get('xh_R_x25_150_A_front:2') == 'evidence:missing_rgb', fz)
check('missing_capture_recorded', fz.get('xh_R_x25_150_B_oblique:*') == 'missing_capture', fz)
check('closure_and_incomplete', res['closure']['closed'] and res['closure']['rows_match_evaluated'] and res['complete'] is False, res['closure'])
s = res['summary']['R_x25_150']
check('no_positive_stats_NA', s['n_positive'] == 0 and s['L1'] == 'NA' and s['L1_mask_iou_mean_missed0'] == 'NA'
      and res['summary']['asset_equal_weight_R'] == 'NA', s)

# 3 擷取閘門
seqs = sorted(p.split('/runs/xh_')[1].split('/analysis')[0] for p in
              __import__('glob').glob(os.path.join(HERE, 'runs', 'xh_*', 'analysis', 'xh_capture_check.json')))
r = subprocess.run([sys.executable, os.path.join(HERE, 'xh_capture_gate.py')] + seqs, capture_output=True, text=True)
check('gate_passes_existing_20', r.returncode == 0 and len(seqs) == 20, r.stdout[-300:])
rep = json.load(open(os.path.join(HERE, 'runs', 'xh_N_k35_A_front', 'analysis', 'xh_capture_check.json')))
bad = dict(rep, asset_sha_file_now='0' * 64, has_handle_center_field=True)
w = GATE.gate(bad)
check('gate_rejects_tampered', 'asset_sha_mismatch' in w and 'handle_center_field_semantics' in w, w)
check('gate_rejects_missing_report', GATE.gate(None) == ['report_missing'])
print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
