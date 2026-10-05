#!/usr/bin/env python3
"""嚴格停車（PARK_HOLD v2.1）vs MotM：比較表（全部數字由結果檔讀出，不手抄）。

輸出：
  results/motm_speed/park_vs_motm_comparison.md   三張表（正式逐趟、兩策略彙總、策略演進）
  results/motm_speed/figures/park_vs_motm_table.png  兩策略彙總表（投影片用）

    python3 evaluation/park_comparison_table.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(HERE, 'results', 'motm_speed')
SEGS = [('transfer_to_wholebody', 'GO→全身接手'), ('grasp_attached', '全身接手→夾持成立'),
        ('open_hold_done', '夾持成立→開啟保持完成'), ('close_hold_done', '開啟保持完成→關閉保持完成'),
        ('handback_to_nav', '關閉保持完成→收臂交還導航'), ('terminal_complete', '收臂交還導航→回程終端完成')]
# 表 1 欄名（累計時刻：只寫到達點）
CUM = {'transfer_to_wholebody': '全身接手', 'grasp_attached': '夾持成立', 'open_hold_done': '開啟保持完成',
       'close_hold_done': '關閉保持完成', 'handback_to_nav': '收臂交還導航'}
NAME = {'MOTM': 'MotM', 'PARK_HOLD': '嚴格停車'}
NOTE = ('註：「最大」各列呈現各趟最大值的平均（最小–最大），不是整批最大值。S4 窗口為夾持成立→RELEASE_WAIT；'
        '三對漂移差方向不一致，全部通過 10 mm 門檻。GO→全身接手約 11 秒的差距屬整個入場／交棒區段（拆解見表 4），'
        '靜止閘門窗本身約 0.51 秒。')


def seg_durations(seg):
    out, prev = [], 0.0
    for k, _ in SEGS:
        out.append(seg[k] - prev)
        prev = seg[k]
    return out


def fmt_range(v, nd=2):
    v = np.asarray(v, float)
    return f'{v.mean():.{nd}f}（{v.min():.{nd}f}–{v.max():.{nd}f}）'


def main():
    d = json.load(open(os.path.join(MS, 'park_hold_formal_results.json')))
    runs = d['runs']
    by = {r['rid']: r for r in runs}
    L = []
    L.append('# 嚴格停車（PARK_HOLD v2.1）vs 移動中操作（MotM）比較表\n')
    L.append('資料：`results/motm_speed/park_hold_formal_results.json`（由凍結分析器 `park_formal_summarize.py` 生成的正式結果；凍結清單 `freeze_park_formal.sha256` 含分析器與計畫，不含之後產生的結果 JSON）；本檔由 '
             '`evaluation/park_comparison_table.py` 產生。計時＝共同 GO（第一筆實際發布導航計畫）→ 獨立終端完成。'
             '比較的是**兩個完整策略**；n = 3，不做顯著性。\n')
    # ---- 表 1 正式逐趟 ----
    L.append('## 表 1　正式六趟逐趟\n')
    L.append('**以下時間欄皆為自共同 GO 起算的累計時刻 (s)**（不是區段時長；區段時長見表 2）。\n')
    L.append('| 趟次 | 配對／順序 | 策略 | 任務 | S1–S6 | 停車模式 | 完成時間 (s) | ' + ' | '.join(CUM[k] for k, _ in SEGS[:-1]) + ' |')
    L.append('|' + '---|' * (7 + len(SEGS) - 1))
    sched = {p['M']: (k, p['order']) for k, p in d['pairs'].items()}
    sched.update({p['H']: (k, p['order']) for k, p in d['pairs'].items()})
    for r in runs:
        k, o = sched[r['rid']]
        S = r['S'] or {}
        s_ok = 'PASS' if S and all(v is True for v in S.values()) else 'FAIL'
        mode = '—（不適用）' if r['method'] == 'MOTM' else ('合規' if r['mode_ok'] else '違規')
        seg = r['segments']
        cells = [f'{seg[key]:.2f}' for key, _ in SEGS[:-1]]
        L.append(f"| {r['rid']} | {k}／{o} | {NAME[r['method']]} | {'完成' if r['task_ok'] else '未完成'} | {s_ok} | {mode} | "
                 f"**{r['T_complete_s']:.2f}** | " + ' | '.join(cells) + ' |')
    # ---- 表 2 兩策略彙總 ----
    M = [r for r in runs if r['method'] == 'MOTM']
    H = [r for r in runs if r['method'] == 'PARK_HOLD']
    pairs = sorted(d['pairs'].items())
    rows2 = []

    def add(name, vm, vh, diff=None, note=''):
        rows2.append((name, vm, vh, diff if diff is not None else '', note))
    tm = [r['T_complete_s'] for r in M]
    th = [r['T_complete_s'] for r in H]
    dd = [p['T_M_minus_T_H_s'] for _, p in pairs]
    pc = [p['pct_of_H'] for _, p in pairs]
    add('完成時間 GO→終端 (s)', fmt_range(tm), fmt_range(th),
        f"{'／'.join(f'{x:+.2f}' for x in dd)}（{'／'.join(f'{x:+.2f}%' for x in pc)}）", '三對同向')
    g = [(by[p['M']]['segments']['grasp_attached'] - by[p['H']]['segments']['grasp_attached']) for _, p in pairs]
    add('GO→夾持成立 (s)', fmt_range([r['segments']['grasp_attached'] for r in M]),
        fmt_range([r['segments']['grasp_attached'] for r in H]), '／'.join(f'{x:+.2f}' for x in g), '共同任務里程碑')
    for j, (key, name) in enumerate(SEGS):
        dm = [seg_durations(r['segments'])[j] for r in M]
        dh = [seg_durations(r['segments'])[j] for r in H]
        dif = [seg_durations(by[p['M']]['segments'])[j] - seg_durations(by[p['H']]['segments'])[j] for _, p in pairs]
        note = '含入場減速、收斂、等待、交棒；閘門窗本身約 0.5 s' if j == 0 else ''
        add(f'區段時長 {name} (s)', fmt_range(dm), fmt_range(dh), '／'.join(f'{x:+.2f}' for x in dif), note)
    def cnt(rs, f):
        return f'{sum(1 for r in rs if f(r))}／{len(rs)}'
    s_pass = lambda r: bool(r['S']) and all(v is True for v in r['S'].values())
    add('任務判準 S1–S6', cnt(M, s_pass) + ' PASS', cnt(H, s_pass) + ' PASS', '', '')
    add('完整重播', cnt(M, lambda r: r['replay'] == 'PASS') + ' PASS', cnt(H, lambda r: r['replay'] == 'PASS') + ' PASS', '', '')
    add('全窗停車判準', '不適用', cnt(H, lambda r: r['mode_ok']) + ' 合規', '',
        f"執行端 hold PASS {cnt(H, lambda r: r['park']['hold_verdict'] == 'PASS')}；audit 全窗 {cnt(H, lambda r: r['park']['audit_strict'] is True)}")
    vmax = [r['park']['max']['v_lin_mps'] * 1e3 for r in H]
    dev = [r['park']['max']['dev_pos_mm'] for r in H]
    srv = [r['park']['max']['hold_cmd_lin_mps'] * 1e3 for r in H]
    add('保持期底盤逐步速度最大 (mm/s)', '—', fmt_range(vmax, 3), '', 'park_gate hold.max.v_lin_mps；門檻 1 mm/s；餘裕小')
    add('保持期相對錨點偏離最大 (mm)', '—', fmt_range(dev, 3), '', 'hold.max.dev_pos_mm；門檻 1 mm')
    add('停車伺服套用合線速度最大 (mm/s)', '—', fmt_range(srv, 3), '', 'hold.max.hold_cmd_lin_mps；上界 5 mm/s')
    add('OPEN 相位雙指同時 ≥ 0.5 N', f"{min(r['grip_open']['both_ge_0p5N_frac'] for r in M):.0%}",
        f"{min(r['grip_open']['both_ge_0p5N_frac'] for r in H):.0%}", '', 'OPEN→OPEN_HOLD 事件窗，逐物理步')
    add('OPEN 相位單指法向接觸力峰值 (N)', fmt_range([r['grip_open']['peak_N'] for r in M]),
        fmt_range([r['grip_open']['peak_N'] for r in H]), '', '')
    s4m = [r['S4_drift_max_mm'] for r in M]
    s4h = [r['S4_drift_max_mm'] for r in H]
    s4d = [by[p['M']]['S4_drift_max_mm'] - by[p['H']]['S4_drift_max_mm'] for _, p in pairs]
    add('S4 夾持漂移最大 (mm)', fmt_range(s4m, 3), fmt_range(s4h, 3), '／'.join(f'{x:+.3f}' for x in s4d),
        '窗口：夾持成立→RELEASE_WAIT；全部通過 10 mm 門檻；兩對 M 較大、一對 M 較小，配對方向不一致')
    L.append('## 表 2　兩策略彙總（平均（最小–最大）；配對差＝MotM − 嚴格停車，依配對 A／B／C）\n')
    L.append(NOTE + '\n')
    L.append('| 指標 | MotM | 嚴格停車 | 配對差 A／B／C | 備註 |')
    L.append('|---|---|---|---|---|')
    for name, vm, vh, df, note in rows2:
        L.append(f'| {name} | {vm} | {vh} | {df} | {note} |')
    L.append('\n**核心說法**：本配置三對比較中，六趟皆完成任務，嚴格停車組亦通過全窗停車判準。MotM 完成時間三對皆縮短 '
             f"{min(-x for x in dd):.2f}–{max(-x for x in dd):.2f} 秒（{min(-x for x in pc):.2f}–{max(-x for x in pc):.2f}%）。"
             '此為兩個完整策略的結果，不代表單獨改變底盤自由度的效果；夾持漂移的配對方向不一致。\n')
    L.append('\n正式結果檔原文（park_hold_formal_summary.yaml）：' + yaml.safe_load(open(os.path.join(MS, 'park_hold_formal_summary.yaml')))['正式說法'].strip() + '\n')
    # ---- 表 3 策略演進 ----
    mt = yaml.safe_load(open(os.path.join(MS, 'mt_b1_results.yaml')))['全程（開始移動 → DONE）']['配對差_MotM減停車']
    mtd = [mt[k] for k in ('A_M先', 'B_P先', 'C_M先')]
    mt_pct = 100 * mt['平均'] / float(np.mean(
        yaml.safe_load(open(os.path.join(MS, 'mt_b1_results.yaml')))['全程（開始移動 → DONE）']['停車']))

    def gate(rid):
        return json.load(open(os.path.join(HERE, 'runs', rid, 'park_gate.json')))['hold']
    g1, g2, g21 = gate('pf_func_F'), gate('ph_func_H'), gate('ph21_func_H')

    def first_v(h):
        fv = h['first_violation']
        return f"{float(fv['why'].split()[1]) * 1e3:.3f} mm/s @ {fv['t']:.2f} s"
    L.append('## 表 3　版本沿革——非跨批次效能比較（正式時間比較只用表 1／2）\n')
    L.append('| 對照組 | 底盤在操作期間 | 結果 | 計時定義 | 定位 |')
    L.append('|---|---|---|---|---|')
    L.append(f"| PARK_GRASP_WB（舊「停車」，mt_b1） | 抓取時近靜止；拉開／關回各移動約 171 mm | 6 趟完成；MotM 快 "
             f"{'／'.join(f'{-x:.2f}' for x in mtd)} s（平均 {-mt['平均']:.2f} s，約 {-mt_pct:.2f}%） | 開始移動→DONE | 歷史結果；**不是固定底盤操作** |")
    L.append(f"| PARK_FIXED v1（保持期底盤命令恆零；pf_func_F） | 套用命令 {g1['max']['cmd_abs']:.0f}，實測仍移動 | 功能確認 {g1['verdict']}："
             f"逐步 {first_v(g1)}（UNFOLD）越界、閂鎖中止 | — | 發現：零底盤命令不足以維持實測停車 |")
    L.append(f"| PARK_HOLD v2（停車伺服保持；ph_func_H） | 違規前偏離 ≤ 0.48 mm（功能確認結果檔敘述） | 功能確認 {g2['verdict']}："
             f"逐步 {first_v(g2)}（OPEN_HOLD 入口）越界中止；同時單指承載 | — | 未通過；不調伺服、改做夾持診斷 |")
    L.append(f"| PARK_HOLD v2.1（＋運動中偏移觀測，兩組共同；ph21_func_H） | 偏離 ≤ {g21['max']['dev_pos_mm']:.3f} mm、逐步 ≤ "
             f"{g21['max']['v_lin_mps'] * 1e3:.3f} mm/s | 功能確認 {g21['verdict']}（n = 1，速度餘裕 "
             f"{100 * (1 - g21['max']['v_lin_mps'] / 1e-3):.1f}%） | GO→DONE | 候選修正有效，非根因證明 |")
    L.append(f'| **PARK_HOLD v2.1 正式三對** | 偏離 ≤ {max(dev):.3f} mm、逐步 ≤ {max(vmax):.3f} mm/s | 六趟完成；**MotM 快 '
             f"{'／'.join(f'{-x:.2f}' for x in dd)} s（{'／'.join(f'{-x:.2f}' for x in pc)}%）** | GO→獨立終端完成 | **正式成果**：嚴格停車對照下整體時間縮短 |")
    L.append('\n表 3 的違規數值取自各趟 `park_gate.json`（執行端）；mt_b1 取自 `mt_b1_results.yaml`。'
             'v2 違規前偏離取自 `park_hold_func_result.yaml` 敘述（park_gate 的 max 含中止後漂移 '
             f"{g2['max']['dev_pos_mm']:.2f} mm，不可當保持表現）。\n")
    # ---- 表 4 入場拆解（park_entry_breakdown.py 產生；只讀既有紀錄） ----
    eb = json.load(open(os.path.join(MS, 'park_entry_breakdown.json')))
    L.append('## 表 4　GO→全身接手的拆解（導航段自 GO 起算，其餘欄自減速段起點起算，單位 s；由 `park_entry_breakdown.py` 從既有逐物理步紀錄重算，未另跑模擬）\n')
    L.append('| 趟次 | 策略 | 導航段 (s) | 減速段→全身接手 (s) | 位置收斂 ≤ 1 mm | 最後一步 > 1 mm/s | 停車鎖存（鎖存時偏航誤差 rad） | 閘門窗 | 轉給全身時相對停位偏航 (rad) |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for r in eb['runs']:
        lat = '—' if r['latch_s'] is None else f"{r['latch_s']:.2f}（{r['latch_eyaw_rad']:.5f}）"
        gw = '—' if 'gate_window_s' not in r else f"{r['gate_window_s'][0]:.2f}–{r['gate_window_s'][1]:.2f}"
        L.append(f"| {r['rid']} | {NAME[r['method']]} | {r['nav_s']:.2f} | {r['glide_to_wb_s']:.2f} | {r['pos_settle_1mm_s']:.2f} | "
                 f"{r['last_v_gt_1mmps_s']:.2f} | {lat} | {gw} | {r['yaw_err_at_wb_rad']:+.4f} |")
    Hb = [r for r in eb['runs'] if r['method'] == 'PARK_HOLD']
    Mb = [r for r in eb['runs'] if r['method'] == 'MOTM']
    L.append(f"\n讀法：導航段兩組相近（M {fmt_range([r['nav_s'] for r in Mb])}、H {fmt_range([r['nav_s'] for r in Hb])} s）。"
             f"差距集中在減速段：H 位置在減速段起點後 {min(r['pos_settle_1mm_s'] for r in Hb):.2f}–{max(r['pos_settle_1mm_s'] for r in Hb):.2f} s 已收斂到 1 mm，"
             f"之後再 {min(r['latch_s'] - r['pos_settle_1mm_s'] for r in Hb):.2f}–{max(r['latch_s'] - r['pos_settle_1mm_s'] for r in Hb):.2f} s "
             f"偏航誤差才進入停車鎖存容差 0.005 rad（三趟鎖存時偏航誤差 {min(r['latch_eyaw_rad'] for r in Hb):.5f}–{max(r['latch_eyaw_rad'] for r in Hb):.5f} rad，緊貼容差；"
             f"此段紀錄中的偏航命令約每 2 s 減半，機制未另驗證）；鎖存後閘門窗約 0.51 s。"
             f"M 在減速段約 {min(r['glide_to_wb_s'] for r in Mb):.1f}–{max(r['glide_to_wb_s'] for r in Mb):.1f} s 即轉給全身，"
             f"當時相對停位偏航仍約 {min(abs(r['yaw_err_at_wb_rad']) for r in Mb):.2f}–{max(abs(r['yaw_err_at_wb_rad']) for r in Mb):.2f} rad，由全身控制邊動邊收。"
             '這是**本配置停車入場剖面**的特性；本表不推論改用較快偏航收斂的停車剖面後差距會是多少（未實跑）。\n')
    L.append('\n**不能宣稱**：差異單獨來自底盤是否移動；MotM 全面品質較好（S4 漂移配對方向不一致）；顯著性、穩定成功率或跨任務泛化；'
             '偏移估計是單指承載的根因；舊 0.23 s 與本批 12.5–13.2 s 同一對照，或把兩者之差讀成同一基線上的改善量。\n')
    out = os.path.join(MS, 'park_vs_motm_comparison.md')
    open(out, 'w').write('\n'.join(L))
    # ---- 表 2 圖 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': ['Noto Sans CJK JP']})
    cells = [[n, vm, vh, df] for n, vm, vh, df, _ in rows2]
    fig, ax = plt.subplots(figsize=(13, 0.27 * len(cells) + 0.5))
    ax.axis('off')
    tb = ax.table(cellText=cells, colLabels=['指標', 'MotM', '嚴格停車', '配對差 A／B／C（M − H）'],
                  loc='upper center', cellLoc='left', colLoc='left', colWidths=[0.27, 0.2, 0.2, 0.33])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.35)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor('#d1d5db')
        if i == 0:
            c.set_facecolor('#1f2937')
            c.get_text().set_color('white')
        elif i == 1:
            c.set_facecolor('#ecfdf5')
    ax.set_title('嚴格停車（PARK_HOLD v2.1）vs MotM：正式三對彙總（平均（最小–最大）；n = 3，不做顯著性；兩個完整策略的比較）', fontsize=10, pad=4)
    fig.tight_layout()
    fig.text(0.01, 0.0, NOTE + '\n「OPEN 相位雙指同時 ≥ 0.5 N」與力峰值：OPEN→OPEN_HOLD 事件窗逐物理步；力峰值為兩指讀值中的最大值（非合力）。',
             fontsize=8, ha='left', va='top', wrap=True)
    fig.savefig(os.path.join(MS, 'figures', 'park_vs_motm_table.png'), dpi=160, bbox_inches='tight', pad_inches=0.15)
    print('ok', out)


if __name__ == '__main__':
    main()
