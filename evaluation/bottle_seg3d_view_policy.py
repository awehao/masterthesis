#!/usr/bin/env python3
"""有界自適應觀察的視點決策（Codex 20261101_230248 (b)；純函式，不讀桌高或佈置編號，只讀旁路輸出的有效旗標與影像截斷／外緣分類）。
輸入：某個視點停穩後收到的旁路輸出列（bottle_seg3d schema ≥ 3 的欄位 dict）。只採用新鮮的列（fresh：觀察窗、拒收缺值／非有限／負年齡／逾期／未來戳／重複戳），
  決策時窗內最新一列仍須未逾 MAX_AGE_S。
判斷（比例 ≥ MAJ）：
  complete   ＝同一列 xy_valid、top_observed、support_observed、height_valid 同時成立的列占整個窗 ≥ MAJ（逐列，不拼接不同幀；Codex 20261101_231633）⇒ 接受。
             接受代表「取得符合此啟發式條件的觀察」，不是抓取成功；中心仍含已知半徑先驗，不能稱全部幾何均由視覺量得。
  缺項分類用各欄比例（只用來選下一視點）：
  缺底       ＝support_observed 不成立且 bottom_cut（遮罩碰影像下緣）⇒ 下一視點 V_LOW（瞄準點較低、相機較遠）。
  缺頂／缺支撐（未出畫）＝top_observed 不成立（top_cut 或上緣外側無深度）或 support_observed 不成立但沒有 bottom_cut ⇒ 下一視點 V_HIGH（相機較高）。
  沒有有效實例 ⇒ 下一視點 V_HIGH（瓶身入畫比例最高）；新鮮列不足 MIN_ROWS ⇒ 等待（呼叫端計時，逾時視同沒有有效實例）。
上限：最多 MAX_ADJUST＝2 次視點調整；已用過的視點不再回去；仍不完整就明確拒絕（reject，附缺項）。
另報 known_type_candidate：xy_valid 且 support_observed、但瓶頂未確認 ⇒ 若採用已知瓶型尺寸先驗才可能構成抓取目標；只標記，不算 complete、不進控制。
"""
import numpy as np

SETTLE_S, MAX_AGE_S, MIN_ROWS, MAJ, MAX_ADJUST = 0.4, 0.5, 5, 0.6, 2
ORDER = ('V0', 'V_LOW', 'V_HIGH')


def fresh(rows, t_arrive, t_now):
    """觀察窗＝擷取戳在 [到達＋SETTLE_S, t_now] 的列；拒收：缺 stamp／recv_age_s、非有限、負年齡、年齡 > MAX_AGE_S、未來戳、重複戳（保留先到者）。
    決策時另要求窗內最新一列在 t_now 仍未逾 MAX_AGE_S（由 assess 的 latest_age 檢查）。"""
    out, seen = [], set()
    for r in rows:
        st, ag = r.get('stamp'), r.get('recv_age_s')
        if st is None or ag is None or not np.isfinite(st) or not np.isfinite(ag):
            continue
        if ag < 0.0 or ag > MAX_AGE_S or st > t_now + 1e-9 or st < t_arrive + SETTLE_S:
            continue
        key = round(float(st), 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def assess(rows, t_now=None):
    """回傳 dict(enough, n, n_complete, complete, deficits, known_type_candidate, frac)。rows 已是 fresh() 的結果。
    完整以「同一列」判斷：該列 valid、xy_valid、top_observed、support_observed、height_valid 同時成立，完整列占整個窗 ≥ MAJ 才算 complete（不拼接不同幀）。
    known_type_candidate 同樣逐列：同列 xy_valid 且 support_observed、但不完整的列占窗 ≥ MAJ。t_now 給定時，窗內最新一列在 t_now 須未逾 MAX_AGE_S，否則 enough＝False（stale）。"""
    n = len(rows)
    if n < MIN_ROWS:
        return dict(enough=False, n=n)
    if t_now is not None and t_now - max(r['stamp'] for r in rows) > MAX_AGE_S:
        return dict(enough=False, n=n, stale=True)
    V = [r for r in rows if r['valid'] > 0.5]
    f = lambda key, R=V: (sum(1 for r in R if r[key] > 0.5) / len(R)) if R else 0.0   # noqa: E731
    frac = dict(valid=len(V) / n, xy=f('xy_valid'), top=f('top_observed'), support=f('support_observed'), height=f('height_valid'),
                bottom_cut=f('bottom_cut'), top_cut=f('top_cut'), top_unknown=(sum(1 for r in V if r['edge_top_unknown'] >= 0.5) / len(V)) if V else 0.0)
    if frac['valid'] < MAJ:
        return dict(enough=True, n=n, n_complete=0, complete=False, deficits=['no_target'], known_type_candidate=False, frac=frac)
    row_ok = lambda r: all(r[k] > 0.5 for k in ('valid', 'xy_valid', 'top_observed', 'support_observed', 'height_valid'))   # noqa: E731
    n_complete = sum(1 for r in rows if row_ok(r))
    n_ktc = sum(1 for r in rows if (not row_ok(r)) and r['valid'] > 0.5 and r['xy_valid'] > 0.5 and r['support_observed'] > 0.5)
    frac.update(complete_rows=n_complete / n, known_type_rows=n_ktc / n)
    complete = n_complete / n >= MAJ
    deficits = []
    if frac['support'] < MAJ:
        deficits.append('bottom_out_of_frame' if frac['bottom_cut'] >= MAJ else 'support_unobserved')
    if frac['top'] < MAJ:
        deficits.append('top_cut' if frac['top_cut'] >= MAJ else ('top_background_unknown' if frac['top_unknown'] >= MAJ else 'top_unconfirmed'))
    if frac['xy'] < MAJ:
        deficits.append('xy_invalid')
    return dict(enough=True, n=n, n_complete=n_complete, complete=complete, deficits=[] if complete else (deficits or ['incomplete_rows']),
                known_type_candidate=(not complete) and n_ktc / n >= MAJ, frac=frac)


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
    if any(x in d for x in ('top_cut', 'top_background_unknown', 'top_unconfirmed', 'support_unobserved', 'no_target', 'xy_invalid', 'incomplete_rows')):
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
        a = assess(seq.get(view, []))      # 離線序列已是新鮮列
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
    mk = lambda **kw: dict(dict(stamp=1.0, recv_age_s=0.05, valid=1.0, xy_valid=1.0, top_observed=1.0, support_observed=1.0, height_valid=1.0, bottom_cut=0.0, top_cut=0.0, edge_top_unknown=0.0), **kw)   # noqa: E731
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
    # 反例（Codex 231633）：拼接、缺值、負年齡、未來戳、重複戳、決策時最新列逾期
    split = [mk(stamp=1.0 + 0.2 * i, xy_valid=1.0, top_observed=0.0, support_observed=0.0, height_valid=0.0) for i in range(2)] + \
            [mk(stamp=1.4 + 0.2 * i, xy_valid=0.0, top_observed=1.0, support_observed=1.0, height_valid=1.0) for i in range(2)] + [mk(stamp=1.8)]
    a_ = assess(split)
    ok = a_['complete'] is False and a_['n_complete'] == 1
    bad += not ok
    print(('  ✓ ' if ok else '  ✗ ') + f"反例 拼接：5 列中同列完整 1 ⇒ complete {a_['complete']}（n_complete {a_['n_complete']}）")
    t_arr, t_now = 0.0, 3.0
    good = [mk(stamp=1.0 + 0.2 * i) for i in range(5)]
    for nm, extra, want_n in (('缺 recv_age_s', [{k: v for k, v in mk(stamp=2.5).items() if k != 'recv_age_s'}], 5), ('負年齡', [mk(stamp=2.6, recv_age_s=-0.01)], 5),
                              ('非有限', [mk(stamp=float('nan'))], 5), ('未來戳', [mk(stamp=3.5)], 5), ('重複戳', [mk(stamp=1.0)], 5), ('逾期', [mk(stamp=2.7, recv_age_s=0.8)], 5)):
        n_ = len(fresh(good + extra, t_arr, t_now))
        ok = n_ == want_n
        bad += not ok
        print(('  ✓ ' if ok else '  ✗ ') + f'反例 {nm}：新鮮列 {n_}（應 {want_n}）')
    a_ = assess(fresh(good, t_arr, t_now), t_now)
    ok = a_['enough'] is False and a_.get('stale') is True
    bad += not ok
    print(('  ✓ ' if ok else '  ✗ ') + f"反例 決策時最新列逾期（最新戳 1.8、t_now 3.0）⇒ enough {a_['enough']} stale {a_.get('stale')}")
    print('全過' if not bad else f'未過 {bad}')
