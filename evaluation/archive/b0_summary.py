#!/usr/bin/env python3
"""B0 六條件彙整與對照圖（只讀 runs/b0_<條件>_<TAG>/probe.json；不重算 Isaac）。
輸出：results/door/b0_summary_<TAG>.json、第七次素材／圖表／b0_無門殘差六條件_<TAG>.png。

    python3 evaluation/b0_summary.py 1
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import base_residual_probe_core as C                   # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else '1'
OUT = os.path.join(HERE, 'results', 'door', f'b0_summary_{TAG}.json')
FIG = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', f'b0_無門殘差六條件_{TAG}.png')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    rows = []
    for cond in C.CONDITIONS:
        rid = f'b0_{cond}_{TAG}'
        p = os.path.join(HERE, 'runs', rid, 'probe.json')
        d = json.load(open(p))
        s = d['summary']
        rows.append({'condition': cond, 'run_id': rid, 'pose': d['pose'], 'mode': d['mode'], 'complete': s.get('complete'),
                     'why': s.get('why'), 'class': s.get('class'),
                     'residual_body_mm_s': s.get('residual_body_mm_s'), 'residual_norm_mm_s': s.get('residual_norm_mm_s'),
                     'angle_to_door_ref_deg': s.get('angle_to_door_ref_deg'),
                     'step_dv_after_minus_written_median_mm_s': s.get('step_dv_after_minus_written_median_mm_s'),
                     'api_write_readback_maxdiff_mm_s': s.get('api_write_readback_maxdiff_mm_s'),
                     'api_write_readback_n_valid': s.get('api_write_readback_n_valid'),
                     'net_disp_mm': s.get('net_disp_mm'), 'cmd_disp_mm': s.get('cmd_disp_mm'),
                     'park_steady': s.get('park_steady'), 'rim_joints': d['dof']['rim'], 'n_roller': d['dof']['n_roller']})
    res = {'schema': 'b0_summary/1', 'tag': TAG, 'all_complete': all(r['complete'] for r in rows), 'door_ref': C.DOOR_REF,
           'judge': C.JUDGE, 'conditions': rows}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.5, 1]})
    poses = list(C.POSES)
    lab = {'k2_ready': '收攏 K2_READY', 'q_pre': 'Q_PRE', 'grasp': '夾持末姿態'}
    col = {'zero': '#2a6fdb', 'park_hold': '#d9822b'}
    w = 0.36
    for j, m in enumerate(C.MODES):
        xs = [i + (j - 0.5) * w for i in range(len(poses))]
        ys = [next(r['residual_norm_mm_s'] for r in rows if r['pose'] == p and r['mode'] == m) for p in poses]
        ax[0].bar(xs, ys, w * 0.92, color=col[m], label={'zero': '零命令', 'park_hold': 'PARK_HOLD'}[m])
        for x, y in zip(xs, ys):
            ax[0].text(x, y + 0.05, f'{y:.2f}', ha='center', fontsize=8)
    ref = C.DOOR_REF['phase_norm_mm_s']
    for (k, (lo, hi)), c in zip(ref.items(), ('#888', '#555', '#222')):
        ax[0].axhspan(lo, hi, color=c, alpha=0.18)
        ax[0].text(2.55, (lo + hi) / 2, '門趟 ' + k.split('（')[0], fontsize=7, va='center')
    ax[0].set_xticks(range(len(poses)), [lab[p] for p in poses])
    ax[0].set_ylabel('表觀速度殘差 |·|（mm/s）')
    ax[0].set_xlim(-0.6, 3.2)
    ax[0].legend(fontsize=8, frameon=False, loc='upper left')
    ax[0].set_title('無門、直接根速度寫入：殘差大小（灰帶＝門趟各相位範圍）', fontsize=9)
    ax[1].axhline(0, color='#bbb', lw=0.6)
    ax[1].axvline(0, color='#bbb', lw=0.6)
    d = C.DOOR_REF['dir_body']
    ax[1].annotate('', xy=(d[0] * 3, d[1] * 3), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))
    ax[1].text(d[0] * 3, d[1] * 3 + 0.15, '門趟方向', fontsize=8, color='#666', ha='center')
    mk = {'k2_ready': 'o', 'q_pre': 's', 'grasp': '^'}
    for r in rows:
        v = r['residual_body_mm_s']
        ax[1].plot(v[0], v[1], mk[r['pose']], color=col[r['mode']], ms=8, label=f"{lab[r['pose']]}／{r['mode']}")
    ax[1].set_xlabel('本體 x（mm/s）')
    ax[1].set_ylabel('本體 y（mm/s）')
    ax[1].set_aspect('equal')
    ax[1].set_xlim(-3.2, 0.8)
    ax[1].set_ylim(-0.6, 2.6)
    ax[1].legend(fontsize=6.5, frameon=False, loc='lower left')
    ax[1].set_title('殘差方向（本體）', fontsize=9)
    for a_ in ax:
        a_.grid(alpha=0.25)
    fig.suptitle(f'B0 無門底盤表觀速度殘差六條件（TAG {TAG}；各 10 s，全部完整）', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    for r in rows:
        print(r['condition'], r['complete'], r['class'], [round(v, 3) for v in r['residual_body_mm_s']], round(r['residual_norm_mm_s'], 3),
              None if r['angle_to_door_ref_deg'] is None else round(r['angle_to_door_ref_deg'], 1),
              r['park_steady'] and round(r['park_steady']['anchor_err_mm_mean'], 3))


if __name__ == '__main__':
    main()
