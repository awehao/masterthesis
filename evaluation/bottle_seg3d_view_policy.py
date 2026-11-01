#!/usr/bin/env python3
"""有界自適應觀察的視點決策（Codex 20261101_230248 (b)；純函式，不讀桌高或佈置編號，只讀旁路輸出的有效旗標與影像截斷／外緣分類）。
輸入：某個視點停穩後收到的旁路輸出列（bottle_seg3d schema ≥ 3 的欄位 dict）。只採用新鮮的列：擷取戳 ≥ 到達＋SETTLE_S、收到時年齡 ≤ MAX_AGE_S。
判斷（以多數決，比例 ≥ MAJ）：
  complete   ＝xy_valid、top_observed、support_observed、height_valid 同時成立 ⇒ 接受（完整三維目標：中心、瓶頂、支撐、高度都由視覺量得）。
  缺底       ＝support_observed 不成立且 bottom_cut（遮罩碰影像下緣）⇒ 下一視點 V_LOW（瞄準點較低、相機較遠）。
  缺頂／缺支撐（未出畫）＝top_observed 不成立（top_cut 或上緣外側無深度）或 support_observed 不成立但沒有 bottom_cut ⇒ 下一視點 V_HIGH（相機較高）。
  沒有有效實例 ⇒ 下一視點 V_HIGH（瓶身入畫比例最高）；新鮮列不足 MIN_ROWS ⇒ 等待（呼叫端計時，逾時視同沒有有效實例）。
上限：最多 MAX_ADJUST＝2 次視點調整；已用過的視點不再回去；仍不完整就明確拒絕（reject，附缺項）。
另報 known_type_candidate：xy_valid 且 support_observed、但瓶頂未確認 ⇒ 若採用已知瓶型尺寸先驗才可能構成抓取目標；只標記，不算 complete、不進控制。
"""
import numpy as np

SETTLE_S, MAX_AGE_S, MIN_ROWS, MAJ, MAX_ADJUST = 0.4, 0.5, 5, 0.6, 2
ORDER = ('V0', 'V_LOW', 'V_HIGH')


def fresh(rows, t_arrive):
    return [r for r in rows if r['stamp'] >= t_arrive + SETTLE_S and r.get('recv_age_s', 0.0) <= MAX_AGE_S]


def assess(rows):
    """回傳 dict(enough, frac_valid, complete, deficits, known_type_candidate, frac)。rows 已是新鮮列。"""
    n = len(rows)
    if n < MIN_ROWS:
        return dict(enough=False, n=n)
    V = [r for r in rows if r['valid'] > 0.5]
    f = lambda key, R=V: (sum(1 for r in R if r[key] > 0.5) / len(R)) if R else 0.0   # noqa: E731
    frac = dict(valid=len(V) / n, xy=f('xy_valid'), top=f('top_observed'), support=f('support_observed'), height=f('height_valid'),
                bottom_cut=f('bottom_cut'), top_cut=f('top_cut'), top_unknown=(sum(1 for r in V if r['edge_top_unknown'] >= 0.5) / len(V)) if V else 0.0)
    if frac['valid'] < MAJ:
        return dict(enough=True, n=n, complete=False, deficits=['no_target'], known_type_candidate=False, frac=frac)
    complete = frac['xy'] >= MAJ and frac['top'] >= MAJ and frac['support'] >= MAJ and frac['height'] >= MAJ
    deficits = []
    if frac['support'] < MAJ:
        deficits.append('bottom_out_of_frame' if frac['bottom_cut'] >= MAJ else 'support_unobserved')
    if frac['top'] < MAJ:
        deficits.append('top_cut' if frac['top_cut'] >= MAJ else ('top_background_unknown' if frac['top_unknown'] >= MAJ else 'top_unconfirmed'))
    if frac['xy'] < MAJ:
        deficits.append('xy_invalid')
    return dict(enough=True, n=n, complete=complete, deficits=[] if complete else deficits,
                known_type_candidate=(not complete) and frac['xy'] >= MAJ and frac['support'] >= MAJ, frac=frac)


def next_view(tried, a):
    """tried：已用過的視點（含目前）；a：assess 結果。回傳 ('accept', None)、('move', 視點) 或 ('reject', 缺項)。"""
    if a.get('complete'):
        return 'accept', None
    if len(tried) - 1 >= MAX_ADJUST:
        return 'reject', a.get('deficits')
    want = []
    d = a.get('deficits', [])
    if 'bottom_out_of_frame' in d:
        want.append('V_LOW')
    if any(x in d for x in ('top_cut', 'top_background_unknown', 'top_unconfirmed', 'support_unobserved', 'no_target', 'xy_invalid')):
        want.append('V_HIGH')
    want += [v for v in ORDER if v not in want]                    # 首選都用過時，退而用尚未用過的
    for v in want:
        if v not in tried:
            return 'move', v
    return 'reject', d


def run_offline(seq):
    """seq：依序每個視點的新鮮列 {視點: rows}；照策略走一遍（離線核對用）。回傳決策軌跡。"""
    tried, trace, view = ['V0'], [], 'V0'
    while True:
        a = assess(seq.get(view, []))
        if not a.get('enough'):
            a = dict(enough=True, n=a['n'], complete=False, deficits=['no_target'], known_type_candidate=False, frac=None)
        act, arg = next_view(tried, a)
        trace.append(dict(view=view, n=a['n'], complete=a['complete'], deficits=a['deficits'], known_type_candidate=a['known_type_candidate'], action=act, arg=arg,
                          frac=None if a.get('frac') is None else {k: round(float(v), 2) for k, v in a['frac'].items()}))
        if act != 'move' or arg not in seq:
            return trace
        view = arg
        tried.append(view)


if __name__ == '__main__':
    mk = lambda **kw: dict(dict(stamp=1.0, valid=1.0, xy_valid=1.0, top_observed=1.0, support_observed=1.0, height_valid=1.0, bottom_cut=0.0, top_cut=0.0, edge_top_unknown=0.0), **kw)   # noqa: E731
    cases = {
        '完整': ({'V0': [mk()] * 6}, 'accept'),
        '缺底→V_LOW 完整': ({'V0': [mk(support_observed=0.0, height_valid=0.0, bottom_cut=1.0)] * 6, 'V_LOW': [mk()] * 6}, 'accept'),
        '缺頂→V_HIGH 完整': ({'V0': [mk(top_observed=0.0, height_valid=0.0, edge_top_unknown=1.0)] * 6, 'V_HIGH': [mk()] * 6}, 'accept'),
        '兩次都不足→拒絕': ({'V0': [mk(top_observed=0.0, height_valid=0.0)] * 6, 'V_HIGH': [mk(top_observed=0.0, height_valid=0.0)] * 6,
                        'V_LOW': [mk(top_observed=0.0, height_valid=0.0)] * 6}, 'reject'),
        '無實例→V_HIGH→仍無→V_LOW→拒絕': ({'V0': [mk(valid=0.0)] * 6, 'V_HIGH': [mk(valid=0.0)] * 6, 'V_LOW': [mk(valid=0.0)] * 6}, 'reject'),
        '列不足→視同無實例': ({'V0': [mk()] * 3, 'V_HIGH': [mk()] * 6}, 'accept'),
    }
    bad = 0
    for nm, (seq, want) in cases.items():
        tr = run_offline(seq)
        ok = tr[-1]['action'] == want and len(tr) - 1 <= MAX_ADJUST
        bad += not ok
        print(('  ✓ ' if ok else '  ✗ ') + nm + '：' + ' → '.join(f"{t['view']}({t['action']}{'' if t['arg'] is None else ':' + str(t['arg'])})" for t in tr))
    print('全過' if not bad else f'未過 {bad}')
