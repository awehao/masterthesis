#!/usr/bin/env python3
"""第八次第一包：登錄假世界五案的彙整（逐案讀 m8_fake_report.json；先對每案跑 m8_fake_report.py）。
判準（Codex 20261110_190030 第 4 項）：期限內到達、無安全包絡侵入、命令變化率（報平滑器與最終）、無失效資料放行或來回卡死、報等待／行程／耗時／鬆弛／Shield。
    python3 evaluation/m8_fake_cases_summary.py PREFIX OUT.json   （例：m8fake_r3b_T）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    prefix, out = sys.argv[1], sys.argv[2]
    rows = []
    for T in ('25p83', '26p0', '32p0', '38p0', '44p0'):
        run = prefix + T
        p = os.path.join(HERE, 'runs', run, 'm8_fake_report.json')
        if not os.path.exists(p):
            rows.append(dict(run=run, missing=True))
            continue
        d = json.load(open(p))
        clr = d['clearance']
        worst = min((v['disc_min'] for v in clr.values() if v['disc_min'] is not None), default=None)
        who = min(((k, v['disc_min']) for k, v in clr.items() if v['disc_min'] is not None), key=lambda x: x[1], default=(None, None))[0]
        mt = d.get('mode_time_s') or {}
        rows.append(dict(run=run, result=d['result'], arrived=d['result'] == 'done', t_world_start=d['t_cruise_world'], t_world_end=d['t_end_world'],
                         duration_s=None if d['t_cruise_world'] is None else round(d['t_end_world'] - d['t_cruise_world'], 2), path_len_m=d['path_len_m'],
                         intrusion=d['safety']['intrusion'], min_disc_m=worst, min_disc_obj=who, wait_s=mt.get('WAIT', 0.0), stop_blocked_s=mt.get('STOP_BLOCKED', 0.0),
                         exit_s=mt.get('EXIT', 0.0), hold_stale_s=mt.get('HOLD_STALE', 0.0), episodes=d.get('mode_episodes'), eps0_max=d['cbf']['eps0_max'], min_h_min=d['cbf']['min_h_min'],
                         shield_active=(d.get('shield') or {}).get('n_active'), shield_cycles=(d.get('shield') or {}).get('n_cycles'),
                         rate_final_max=d['rate']['final']['a_max'] if d['rate'].get('final') else None, rate_final_n_over=d['rate']['final']['n_over'] if d['rate'].get('final') else None))
    passed = [r for r in rows if not r.get('missing') and r['arrived'] and not r['intrusion']]
    S = dict(prefix=prefix, n=len(rows), n_pass_arrive_no_intrusion=len(passed), rows=rows,
             note='開發場景的有限相位核對，不是泛化成功率；假世界不模擬接觸力；變化率另列，未納入這個計數')
    json.dump(S, open(out, 'w'), ensure_ascii=False, indent=1)
    for r in rows:
        if r.get('missing'):
            print(r['run'], '缺')
            continue
        print(f"{r['run']}: {r['result']}  起訖 {r['t_world_start']}→{r['t_world_end']}  行程 {r['path_len_m']} m  最小圓盤 {r['min_disc_m']}（{r['min_disc_obj']}）"
              f"  等 {r['wait_s']} s  停擋 {r['stop_blocked_s']} s  側移 {r['exit_s']} s  Shield {r['shield_active']}/{r['shield_cycles']}")
    print(f"到達且無侵入：{S['n_pass_arrive_no_intrusion']}／{S['n']}")


if __name__ == '__main__':
    main()
