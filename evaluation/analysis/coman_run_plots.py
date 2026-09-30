"""三類繪圖框架。**空值不補零、失敗不隱藏、不預填成果。**

1. `opening`  開度與操作事件：目標／實測開度，共用時間軸標相位與連接／解除事件
2. `pull`     拉動期間的實測運動：底盤線速／角速 ＋ 手臂關節速率，標出同動區間
3. `compare`  固定／開放底盤比較：逐趟完成狀態、追蹤誤差、最小關節限位餘裕、操作時間

用法
----
    # 由單趟分析輸出繪圖
    python3 evaluation/analysis/coman_run_plots.py --analysis <out>/analysis.json \\
        --run <run_dir> --out <out>/fig
    # 比較圖（多趟分析輸出）
    python3 evaluation/analysis/coman_run_plots.py --compare a/analysis.json b/… \\
        --out <dir>
    # 合成資料自我驗證（圖面會標明非實驗成果）
    python3 evaluation/analysis/coman_run_plots.py --synthetic --out <dir>

規則
----
* 時間軸一律 **sim**（秒）；耗時類另圖，軸標 **wall**。
* 缺測**留空**（NaN 斷線），不以 0 代替、不內插。
* 比較圖保留**失敗趟次**：未完成者以空心標記並標註分類。
* 合成資料的圖一律在標題與圖內標 `合成測試資料，非實驗成果`。
* 不自動產生「哪一組較好」的結論文字。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
import matplotlib.font_manager as _fm                             # noqa: E402

# 中文字型：**缺字形會讓圖面字被畫成方框**，所以明確指定可用的 CJK 字型。
# 取系統實際存在的第一個；都沒有就退回預設並警告（圖仍可產生，但中文會缺）。
_HAVE = {f.name for f in _fm.fontManager.ttflist}
# 候選順序按**實測缺字數**排：AR PL UMing/UKai 在本機拉丁與中文皆 0 缺字；
# Droid Sans Fallback 缺 110 個、Noto Sans 缺 32 個（皆為中文）。
for _c in ('Noto Sans CJK TC', 'Noto Sans CJK SC', 'AR PL UMing CN',
           'AR PL UKai CN', 'WenQuanYi Zen Hei', 'Droid Sans Fallback'):
    if _c in _HAVE:
        # 用**單一**同時含拉丁與中文的字型；matplotlib 3.6 的逐字形回退
        # 對 sans-serif 清單並不可靠（實測仍報缺字）。
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [_c]
        matplotlib.rcParams['axes.unicode_minus'] = False
        CJK_FONT = _c
        break
else:                                                             # noqa: PLW0120
    CJK_FONT = None
    print('警告：找不到 CJK 字型，圖面中文可能缺字', file=sys.stderr)

SYN = '合成測試資料，非實驗成果'
PHASE_C = {'approach': '#9ecae1', 'engage': '#ffb703', 'align': '#ffd6a5',
           'postengage': '#fdc500', 'pull': '#2a9d8f', 'hold': '#8ecae6',
           'release': '#e07a5f', 'retreat': '#adb5bd', 'idle': '#e9ecef',
           'reach': '#cdb4db', 'settle': '#dee2e6', 'done': '#dee2e6'}


def _ax_phases(ax, windows, y0, y1):
    seen = set()
    for w in windows or []:
        c = PHASE_C.get(w['phase'], '#cccccc')
        ax.axvspan(w['t0'], w['t1'], color=c, alpha=0.30,
                   label=None if w['phase'] in seen else w['phase'])
        seen.add(w['phase'])


def _mark(ax, t, text, color='#d00000'):
    if t is None:
        return
    ax.axvline(t, color=color, ls='--', lw=1.2)
    ax.annotate(text, (t, ax.get_ylim()[1]), rotation=90, va='top',
                ha='right', fontsize=7, color=color)


def _note(fig, synthetic, extra=''):
    txt = (SYN + ('　' + extra if extra else '')) if synthetic else extra
    if txt:
        fig.text(0.01, 0.01, txt, fontsize=8, color='#b00020')


def fig_opening(rep, log_cols, log, out, synthetic=False):
    """圖 1：目標／實測開度 ＋ 相位 ＋ 連接／解除事件。"""
    ix = {c: i for i, c in enumerate(log_cols or [])}
    if 't' not in ix or 'opening' not in ix:
        print('  圖1 跳過：缺 t 或 opening 欄', file=sys.stderr)
        return None
    t = [float(r[ix['t']]) for r in log]
    o = [(float(r[ix['opening']]) * 1000.0
          if isinstance(r[ix['opening']], (int, float)) else float('nan'))
         for r in log]
    tgt = (rep.get('run') or {}).get('target_opening_used_m')
    fig, ax = plt.subplots(figsize=(9, 3.4))
    _ax_phases(ax, rep.get('phase_windows'), min(o), max(o))
    ax.plot(t, o, lw=1.3, color='#023047', label='實測開度')
    if tgt is not None:
        ax.axhline(float(tgt) * 1000.0, color='#fb8500', lw=1.2,
                   label=f'目標 {float(tgt)*1000:.1f} mm')
        p1 = (rep.get('thresholds_from_spec') or {}).get(
            'P1_final_opening_err_m_max')
        if p1:
            ax.fill_between([t[0], t[-1]],
                            (float(tgt) - float(p1)) * 1000.0,
                            (float(tgt) + float(p1)) * 1000.0,
                            color='#fb8500', alpha=0.15,
                            label=f'v1 P1 ±{float(p1)*1000:.1f} mm')
    s = rep.get('simultaneity_pull') or {}
    _mark(ax, s.get('attach_sim_t'), '連接')
    for m in (rep.get('events_marks') or []):
        _mark(ax, m.get('t'), m.get('label'), '#6a4c93')
    ax.set_xlabel('模擬時間 sim (s)')
    ax.set_ylabel('開度 (mm)')
    ax.set_title('開度與操作事件')
    ax.legend(fontsize=7, ncol=4, loc='lower right')
    ax.grid(alpha=0.25)
    _note(fig, synthetic, '缺測留空，不補零')
    fig.tight_layout()
    p = os.path.join(out, 'fig1_opening_events.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_pull_motion(rep, log_cols, log, out, synthetic=False):
    """圖 2：拉動期間的**實測**運動 ＋ 同動區間。

    底盤速度欄位缺少時**不畫假線** —— 圖面直接標明無法繪製的原因。
    """
    ix = {c: i for i, c in enumerate(log_cols or [])}
    s = rep.get('simultaneity_pull') or {}
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    have_base = all(c in ix for c in ('base_vx', 'base_vy')) or \
        all(c in ix for c in ('base_x', 'base_y'))
    jn = [f'joint{i}' for i in range(1, 7)]
    have_j = all(j in ix for j in jn)
    t = [float(r[ix['t']]) for r in log] if 't' in ix else []
    win = s.get('window_sim_t')
    if win:
        for ax in axs:
            ax.axvspan(win[0], win[1], color='#2a9d8f', alpha=0.12,
                       label='拉動窗（已連接且 pull）')
    if have_base:
        if 'base_vx' in ix:
            v = [math.hypot(float(r[ix['base_vx']]), float(r[ix['base_vy']]))
                 for r in log]
            src = '實測底盤線速度'
        else:
            v = [float('nan')]
            xs = [float(r[ix['base_x']]) for r in log]
            ys = [float(r[ix['base_y']]) for r in log]
            for i in range(1, len(t)):
                dt = t[i] - t[i - 1]
                v.append(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) / dt
                         if dt > 0 else float('nan'))
            src = '由實測底盤位置差分'
        axs[0].plot(t, [x * 1000 for x in v], lw=1.2, color='#023047',
                    label=src)
        bt = (s.get('thresholds') or {}).get('base_lin_min_mps')
        if bt:
            axs[0].axhline(bt * 1000, color='#fb8500', ls=':', lw=1.1,
                           label=f'門檻 {bt*1000:.1f} mm/s')
    else:
        axs[0].text(0.5, 0.5, '**無底盤實測速度或可定向位置欄位**\n'
                              '（只有 base_drift 距離純量）⇒ 不繪製\n'
                              '見 FIELD_INVENTORY G3',
                    ha='center', va='center', transform=axs[0].transAxes,
                    color='#b00020', fontsize=9)
    axs[0].set_ylabel('底盤線速度 (mm/s)')
    axs[0].legend(fontsize=7, loc='upper right')
    axs[0].grid(alpha=0.25)
    if have_j and t:
        rates = [float('nan')]
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            if dt <= 0:
                rates.append(float('nan'))
                continue
            rates.append(max(abs(float(log[i][ix[j]]) - float(log[i - 1][ix[j]]))
                             for j in jn) / dt)
        axs[1].plot(t, rates, lw=1.2, color='#6a4c93',
                    label='手臂最大關節速率（關節角差分）')
        jt = (s.get('thresholds') or {}).get('joint_rate_min_rps')
        if jt:
            axs[1].axhline(jt, color='#fb8500', ls=':', lw=1.1,
                           label=f'門檻 {jt} rad/s')
    else:
        axs[1].text(0.5, 0.5, '無 joint1…joint6 ⇒ 不繪製',
                    ha='center', va='center', transform=axs[1].transAxes,
                    color='#b00020')
    axs[1].set_ylabel('關節速率 (rad/s)')
    axs[1].set_xlabel('模擬時間 sim (s)')
    axs[1].legend(fontsize=7, loc='upper right')
    axs[1].grid(alpha=0.25)
    ttl = '拉動期間的實測運動'
    if s.get('verdict') != 'computed':
        ttl += f"（同動判定：{s.get('verdict')}）"
    axs[0].set_title(ttl)
    _note(fig, synthetic, '接近／保持／退出不併入拉動同動占比')
    fig.tight_layout()
    p = os.path.join(out, 'fig2_pull_motion.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_compare(reps, out, synthetic=False):
    """圖 3：F／M 逐趟比較。**保留失敗趟次**，不只畫成功的。"""
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    groups = {}
    for r in reps:
        mode = (r.get('run') or {}).get('base_mode') or 'unknown'
        g = 'F 固定底盤' if mode == 'importer_fix_base' else (
            'M 開放底盤' if mode else 'unknown')
        groups.setdefault(g, []).append(r)
    xs = {g: i for i, g in enumerate(sorted(groups))}
    for g, rs in groups.items():
        for k, r in enumerate(rs):
            done = r.get('classification') == 'complete_data'
            style = dict(marker='o', ms=7,
                         mfc=('#2a9d8f' if done else 'none'),
                         mec=('#2a9d8f' if done else '#d00000'), mew=1.6,
                         ls='none')
            jitter = (k - (len(rs) - 1) / 2) * 0.12
            x = xs[g] + jitter
            # 追蹤誤差：pull 相位的 tcp_err RMS（缺則不畫）
            pp = (r.get('per_phase') or {}).get('pull') or [{}]
            e = (pp[0].get('tcp_err_parallel_m') or {}).get('rms')
            if e is not None:
                axs[0].plot([x], [e * 1000], **style)
            m = (pp[0].get('joint_limit_margin_min') or {}).get('rad')
            if m is not None:
                axs[1].plot([x], [m], **style)
            dur = (r.get('run') or {}).get('sim_time_s')
            if dur is not None:
                axs[2].plot([x], [float(dur)], **style)
            for ax in axs:
                ax.annotate(os.path.basename(
                    (r.get('inputs') or {}).get('run_dir', '')) [:18],
                    (x, ax.get_ylim()[0]), fontsize=5, rotation=90,
                    ha='center', va='bottom', color='#555')
    for ax, lab in zip(axs, ('拉動段 TCP 平行誤差 RMS (mm)',
                             '拉動段最小關節限位餘裕 (rad)',
                             '任務模擬時間 (s)')):
        ax.set_xticks(list(xs.values()))
        ax.set_xticklabels(list(xs), fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(alpha=0.25)
    axs[0].set_title('實心＝具備完整分析資料；空心＝未完成／資料不足（**保留**）',
                     fontsize=8, loc='left')
    _note(fig, synthetic, '不預設哪一組較好；各趟散點保留')
    fig.tight_layout()
    p = os.path.join(out, 'fig3_fixed_vs_free_base.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ------------------------------------------------------------- 合成驗證資料
def synth():
    """少量合成資料：涵蓋相位、連接事件、缺測空隙、失敗趟次。"""
    cols = ['t', 'phase', 'opening', 'e_par', 'e_perp', 'limit_margin',
            'base_x', 'base_y', 'base_vx', 'base_vy'] + \
        [f'joint{i}' for i in range(1, 7)]
    log, t = [], 0.0
    plan = [('approach', 1.0), ('engage', 0.5), ('pull', 2.0), ('hold', 2.0),
            ('release', 0.4), ('retreat', 1.0)]
    op = 0.0
    for ph, dur in plan:
        n = int(dur / 0.01)
        for k in range(n):
            t += 0.01
            if ph == 'pull':
                op = 0.020 * (k / max(n - 1, 1))
            moving = ph in ('pull', 'approach', 'retreat')
            row = [round(t, 4), ph, op,
                   0.001 * math.sin(t), 0.0005 * math.cos(t),
                   1.0 - 0.01 * k / max(n, 1),
                   10.5 + (0.02 * k / max(n, 1) if moving else 0.0), 8.09,
                   (0.008 if moving else 0.0), 0.0] + \
                [0.1 * math.sin(t + i) if moving else 0.3 for i in range(6)]
            # 合成一段缺測：pull 中段插入空隙（不補零）
            if ph == 'pull' and 0.9 < k / max(n, 1) < 0.95:
                continue
            log.append(row)
    return cols, log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis')
    ap.add_argument('--run')
    ap.add_argument('--compare', nargs='*', default=None)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    made = []

    if a.synthetic:
        cols, log = synth()
        rep = {'run': {'target_opening_used_m': 0.020, 'sim_time_s': 6.9,
                       'base_mode': None},
               'thresholds_from_spec': {'P1_final_opening_err_m_max': 0.0005},
               'phase_windows': [], 'classification': 'complete_data',
               'inputs': {'run_dir': 'SYNTHETIC'},
               'simultaneity_pull': {
                   'verdict': 'computed', 'attach_sim_t': 1.5,
                   'window_sim_t': [1.5, 3.5],
                   'thresholds': {'base_lin_min_mps': 0.005,
                                  'joint_rate_min_rps': 0.005}},
               'per_phase': {'pull': [{'tcp_err_parallel_m': {'rms': 0.001},
                                       'joint_limit_margin_min':
                                           {'rad': 0.9, 'joint': 'joint5'}}]}}
        # 相位窗由合成資料現算
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from coman_run_analyze import phase_windows
        ts = [r[0] for r in log]
        phs = [r[1] for r in log]
        rep['phase_windows'] = [
            {'phase': w['phase'], 't0': w['t0'], 't1': w['t1'],
             'dur_s': w['t1'] - w['t0'], 'n': w['n'], 'gap_splits': w['gaps']}
            for w in phase_windows(ts, phs, 0.15)]
        made += [x for x in (fig_opening(rep, cols, log, a.out, True),
                             fig_pull_motion(rep, cols, log, a.out, True))
                 if x]
        # 比較圖：一趟完整、一趟未完成、一趟未進入任務
        bad = dict(rep, classification='task_incomplete',
                   inputs={'run_dir': 'SYNTH_FAIL'},
                   run=dict(rep['run'], base_mode='importer_fix_base',
                            sim_time_s=3.1))
        ns = {'classification': 'not_started', 'inputs': {'run_dir': 'SYNTH_NS'},
              'run': {'base_mode': 'importer_fix_base'}}
        f = dict(rep, inputs={'run_dir': 'SYNTH_F'},
                 run=dict(rep['run'], base_mode='importer_fix_base'))
        m = dict(rep, inputs={'run_dir': 'SYNTH_M'},
                 run=dict(rep['run'], base_mode='free_base'))
        made.append(fig_compare([f, m, bad, ns], a.out, True))
        print(f'合成圖 {len(made)} 張（皆標明「{SYN}」）→ {a.out}')
        for p in made:
            print('  ', p)
        return 0

    if a.compare:
        reps = [json.load(open(p, encoding='utf-8')) for p in a.compare]
        p = fig_compare(reps, a.out)
        print('比較圖 →', p)
        return 0

    if not (a.analysis and a.run):
        print('需要 --analysis 與 --run，或 --compare，或 --synthetic',
              file=sys.stderr)
        return 2
    rep = json.load(open(a.analysis, encoding='utf-8'))
    sim = json.load(open(os.path.join(a.run, 'sim', 'drawer_run.json'),
                         encoding='utf-8'))
    made += [x for x in (fig_opening(rep, sim.get('log_cols'),
                                     sim.get('log') or [], a.out),
                         fig_pull_motion(rep, sim.get('log_cols'),
                                         sim.get('log') or [], a.out))
             if x]
    print(f'{len(made)} 張圖 → {a.out}')
    for p in made:
        print('  ', p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
