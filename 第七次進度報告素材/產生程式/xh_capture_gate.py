#!/usr/bin/env python3
"""XH 擷取核對閘門：讀 xh_capture_check.py 寫出的 analysis/xh_capture_check.json，依明確條件判定；任一條不通過即結束碼 1。

條件（與 XH2 採集核對一致）：報告存在；資產雜湊＝登錄且模擬器載入該資產；真值欄位逐格齊全；真值對資產幾何最大差 < 0.01 mm；
渲染時間嚴格遞增且影格唯一；位姿與渲染時間差 < 1e-6 s；標註讀取無 EvidenceError；N 資產不得有 handle_center、R 資產必須有。

    python3 evaluation/xh_capture_gate.py <序列名> [...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def gate(rep):
    why = []
    if rep is None:
        return ['report_missing']
    if rep.get('asset_sha_expected') != rep.get('asset_sha_file_now'):
        why.append('asset_sha_mismatch')
    if not rep.get('asset_path_in_sim_log'):
        why.append('asset_not_loaded_by_sim')
    a, b = str(rep.get('truth_field_present', '0/1')).split('/')
    if a != b or b == '0':
        why.append('truth_fields_incomplete')
    tv = rep.get('truth_vs_expected_max_mm')
    if tv is None or not tv < 0.01:
        why.append('truth_geometry_mismatch')
    fr = rep.get('frames') or {}
    fc = fr.get('fresh_check') or {}
    if not (fc.get('rendering_time_strictly_increasing') and fc.get('rendering_frame_unique')):
        why.append('frame_time_order')
    pm = fr.get('pose_minus_render_max_s')
    if pm is None or not pm < 1e-6:
        why.append('pose_time_mismatch')
    if 'evidence_error' in (rep.get('labels') or {}) or 'labels' not in rep:
        why.append('label_evidence_error')
    is_N = str(rep.get('asset', '')).startswith('N')
    if is_N == bool(rep.get('has_handle_center_field')):
        why.append('handle_center_field_semantics')
    return why


def main():
    bad = {}
    for seq in sys.argv[1:]:
        p = os.path.join(HERE, 'runs', f'xh_{seq}', 'analysis', 'xh_capture_check.json')
        rep = json.load(open(p)) if os.path.exists(p) else None
        w = gate(rep)
        if w:
            bad[seq] = w
    print(json.dumps({'pass': not bad, 'fail': bad}, ensure_ascii=False))
    sys.exit(0 if not bad and sys.argv[1:] else 1)


if __name__ == '__main__':
    main()
