#!/usr/bin/env python3
"""bottle_f1_select 的單元核對（合成估計，不用模型或 ROS）：區域外不選、不補位近瓶、歧義不任選、鎖定後不換瓶、尺寸閘拒絕視為疑似合併。
    python3 evaluation/test_bottle_f1_select.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_f1_select as FS                           # noqa: E402

REG = (0.0, 0.35, 0.36, 0.62)


def e(x, y=0.42, r=0.02, valid=True, xy=True):
    return dict(valid=valid, r_silhouette=r, z_visible_span=0.15, xy_valid=xy, xy_valid_silhouette=True, center_known_r=[x, y], center_silhouette=[x, y], surface_centroid=[x, y], support=dict(observable=True, z=0.6), n_pts=100)


def locked():
    s = FS.F1Selector(REG)
    for t in range(3):
        s.step(t, [e(0.15)])
    return s


def main():
    c = {}
    s = FS.F1Selector(REG)
    r, k, _, rec = s.step(0, [e(-0.15)])
    c['遠瓶不見時不拿近瓶補位'] = (not r['valid'] and r['why'] == 'out_of_region' and k is None, rec['why'])
    s = FS.F1Selector(REG)
    out = [s.step(t, [e(0.15)]) for t in range(3)]
    c['連續 3 幀才鎖定，前兩幀不放行'] = ([x[0]['valid'] for x in out] == [False, False, True] and out[0][0]['why'] == 'locking' and s.lock is not None, [x[0].get('why') for x in out])
    s = FS.F1Selector(REG)
    s.step(0, [e(0.15)])
    s.step(1, [e(0.25)])
    c['跳動超過 R_LOCK 重算累計'] = (len(s.hist) == 1 and s.lock is None, s.hist)
    s = FS.F1Selector(REG)
    r, k, _, rec = s.step(0, [e(0.15), e(0.25)])
    c['區域內兩個合格：歧義，不任選'] = (r['why'] == 'ambiguous' and k is None and 'n_in_region' in rec['amb'], rec['amb'])
    s = FS.F1Selector(REG)
    r, k, _, rec = s.step(0, [e(0.10, r=0.20)])
    c['尺寸閘拒絕落在區域內：疑似合併歧義'] = (r['why'] == 'ambiguous' and 'size_rejected_near_region' in rec['amb'], rec['amb'])
    s = FS.F1Selector(REG)
    r, k, _, rec = s.step(0, [e(0.03)])
    c['分界線附近：關聯不明歧義'] = (r['why'] == 'ambiguous' and 'near_split' in rec['amb'], rec['amb'])
    s = FS.F1Selector(REG)
    s.step(0, [e(0.15)])
    r, k, _, rec = s.step(1, [e(0.15), e(0.03)])
    c['歧義清空累計'] = (s.hist == [] and r['why'] == 'ambiguous', s.hist)
    s = locked()
    r, k, _, rec = s.step(5, [e(-0.15)])
    c['鎖定後只剩另一瓶：lock_no_match，不改鎖'] = (r['why'] == 'lock_no_match' and abs(s.lock[0] - 0.15) < 1e-9, s.lock)
    s = locked()
    r, k, _, rec = s.step(5, [e(-0.15), e(0.151)])
    c['鎖定後信心排序不影響：選到鎖定那瓶'] = (r.get('valid') and k == 1, k)
    s = locked()
    r, k, _, rec = s.step(5, [e(0.15), e(0.16)])
    c['鎖定範圍內兩個：歧義'] = (r['why'] == 'ambiguous', r.get('why'))
    s = FS.F1Selector(REG)
    r, k, _, rec = s.step(0, [e(0.15, y=1.0)])
    c['有界區域：y 超出不選'] = (r['why'] == 'out_of_region', rec['why'])
    s = FS.F1Selector(REG)
    r, k, _, rec = s.step(0, [dict(valid=False, why='few_mask_depth', n_pts=3)])
    c['無效估計照原因'] = (r['why'] == 'few_mask_depth', r['why'])
    s = FS.F1Selector(REG)
    r, k, _, rec = s.step(0, [e(0.15, xy=False)])
    c['只有輪廓中心不累計鎖定'] = (s.hist == [] and not r['valid'], s.hist)
    bad = [k_ for k_, v in c.items() if not v[0]]
    for k_, v in c.items():
        print(('✓' if v[0] else '✗'), k_, v[1])
    print('全過' if not bad else f'未過 {bad}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
