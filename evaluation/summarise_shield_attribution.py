"""Single source of truth for the CBF / shield attribution numbers.

Reads the bags directly and writes ONE machine-readable artefact that the
figure, the Markdown summary and the report all consume:

    bags/archive_<batch>/
        └─ summarise_shield_attribution.py
             ├─ results/shield_attribution.json      <- the numbers
             ├─ results/SHIELD_ATTRIBUTION_SUMMARY.md
             └─ plot_shield_attrib.py (reads the JSON)

Why this exists: the attribution numbers used to live as hand-copied constants
at the top of the plotting script. They drifted. The 71.9 / 28.1 split was
computed over 4030 classifiable cycles while the figure captioned it "of 4195
interventions", and a "CBF acts at 0.87 m" figure appeared in the write-up with
no script anywhere that produced it. Both are the same failure: a number
computed once, pasted somewhere, and never recomputable.

    python3 evaluation/summarise_shield_attribution.py [--batch archive_gmpc100]
                                                       [--prefix gmpc_cbf__scan_seed]

Bag selection
-------------
Bags are selected by EXACT prefix, never by a bare `*seed*` glob. The archives
hold bags from several arms at once -- archive_gmpc100 currently contains 293
directories across nine prefixes (gmpc__, gmpc_cbf__nocbf_, mppi__, rpp__, ...),
residue from a contamination bug whose fix stopped new archives being polluted
but never cleaned the old ones. A `*seed*` glob sweeps all of them and silently
mixes methods: doing exactly that moved the h>=0 share from 62% to 51%.

Note also that `gmpc_cbf__scan_` is a prefix of `gmpc_cbf__scan_nosm_`, so the
`seed` in the default prefix is load-bearing -- dropping it pulls in the
no-smoother arm.

Alignment
---------
/gmpc/diag and /shield/diag come from different nodes with independent cycle
counters, so they cannot be joined on their cycle fields. They are joined on
TIME instead: each shield cycle takes the most recent gmpc diag no older than
--max-age seconds. A shield cycle with no such partner is counted as
UNCLASSIFIED rather than silently dropped -- those cycles are the reason the
split's denominator is smaller than the intervention count, and the previous
write-up lost track of exactly that.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from std_msgs.msg import Float32MultiArray

# /gmpc/diag field indices (see gmpc_node.py, the d.data = [...] block)
G_CYCLE, G_NEAR_D, G_MIN_H, G_ACTIVE = 0, 7, 13, 14
G_LEN = 22
# /shield/diag field indices (see scan_safety_shield.py)
S_CYCLE, S_ACTIVE, S_DMIN, S_DV, S_FALLBACK = 0, 1, 3, 4, 16
S_LEN = 18

DANGER_H = 0.4          # cbf_danger_thresh as configured for the benchmark


def _read(bag: str, topics: list[str]) -> dict[str, list]:
    out = {t: [] for t in topics}
    rd = rosbag2_py.SequentialReader()
    try:
        rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    except Exception:
        return out
    have = {t.name for t in rd.get_all_topics_and_types()}
    want = [t for t in topics if t in have]
    if not want:
        return out
    rd.set_filter(rosbag2_py.StorageFilter(topics=want))
    while rd.has_next():
        tp, buf, t = rd.read_next()
        out[tp].append((t * 1e-9, deserialize_message(buf, Float32MultiArray).data))
    return out


def _pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else float('nan')


def collect(batch: str, prefix: str, limit: int, max_age: float) -> dict:
    root = f'evaluation/bags/{batch}'
    bags = sorted(glob.glob(f'{root}/{prefix}*'))[:limit]

    # What else is in here? Report it rather than letting it in silently.
    import re as _re
    others: dict[str, int] = {}
    for d in glob.glob(f'{root}/*seed*'):
        b = os.path.basename(d)
        if b.startswith(prefix):
            continue
        others[_re.sub(r'seed\d+$', '', b)] = others.get(_re.sub(r'seed\d+$', '', b), 0) + 1

    total_cycles = 0
    sh_active = sh_both = sh_alone = sh_unclassified = 0
    sh_dist: list[float] = []
    sh_dv: list[float] = []
    cbf_rows = cbf_danger = cbf_hneg = 0
    d_rows: list[float] = []
    d_danger: list[float] = []
    d_hneg: list[float] = []
    n_runs = 0

    for bag in bags:
        if not os.path.exists(f'{bag}/metadata.yaml'):
            continue
        d = _read(bag, ['/gmpc/diag', '/shield/diag'])
        g = [(t, v) for t, v in d['/gmpc/diag'] if len(v) >= G_LEN]
        s = [(t, v) for t, v in d['/shield/diag'] if len(v) >= S_LEN]
        if not g:
            continue
        n_runs += 1

        gt = np.array([t for t, _ in g])
        for _, v in g:
            total_cycles += 1
            near, h, act = float(v[G_NEAR_D]), float(v[G_MIN_H]), float(v[G_ACTIVE])
            if act <= 0:
                continue
            cbf_rows += 1
            good = np.isfinite(near) and 0.0 < near < 20.0
            if good:
                d_rows.append(near)
            if h < DANGER_H:
                cbf_danger += 1
                if good:
                    d_danger.append(near)
            if h < 0.0:
                cbf_hneg += 1
                if good:
                    d_hneg.append(near)

        # shield cycles, joined to the most recent gmpc diag within max_age
        for t, v in s:
            if float(v[S_ACTIVE]) <= 0.0:
                continue
            sh_active += 1
            dmin = float(v[S_DMIN])
            if np.isfinite(dmin) and dmin >= 0.0:
                sh_dist.append(dmin)
            dv = float(v[S_DV])
            if np.isfinite(dv):
                sh_dv.append(dv)
            i = int(np.searchsorted(gt, t) - 1)
            if i < 0 or (t - gt[i]) > max_age:
                sh_unclassified += 1
                continue
            h = float(g[i][1][G_MIN_H])
            if not np.isfinite(h):
                sh_unclassified += 1
            elif h < 0.0:
                sh_both += 1
            else:
                sh_alone += 1

    def q(v: list[float]) -> dict:
        if not v:
            return {'cycles': 0, 'median_m': None, 'p25_m': None, 'p75_m': None}
        a = np.array(v)
        return {'cycles': int(len(a)),
                'median_m': round(float(np.median(a)), 4),
                'p25_m': round(float(np.percentile(a, 25)), 4),
                'p75_m': round(float(np.percentile(a, 75)), 4)}

    classified = sh_both + sh_alone

    # ---- anti-drift invariants -------------------------------------------
    # These are the exact mistakes this file exists to prevent. Fail loudly
    # rather than emit a JSON that a figure will caption incorrectly.
    assert classified + sh_unclassified == sh_active, (
        f'{classified} + {sh_unclassified} != {sh_active}')
    assert sh_both + sh_alone == classified
    assert 0 <= sh_both <= classified and 0 <= sh_alone <= classified
    assert cbf_hneg <= cbf_danger <= cbf_rows <= total_cycles

    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'],
                                             text=True).strip())
    except Exception:
        head, dirty = None, None

    return {
        'schema_version': 1,
        'source_batch': batch,
        'bag_prefix': prefix,
        'n_runs': n_runs,
        'n_bags_matched': len(bags),
        # Foreign bags sharing this archive. Non-empty means the archive was
        # polluted by another arm; they are EXCLUDED, but recorded so the
        # exclusion is visible rather than assumed.
        'other_prefixes_excluded': dict(sorted(others.items())),
        'git_commit': head,
        'git_dirty': dirty,
        'danger_threshold_h': DANGER_H,
        'join_max_age_s': max_age,
        'total_cycles': total_cycles,

        'shield_active': sh_active,
        'shield_cbf_classifiable': classified,
        'shield_unclassified': sh_unclassified,
        'shield_with_h_negative': sh_both,
        'shield_with_h_nonnegative': sh_alone,
        # Percentages ALWAYS carry their denominator. The previous write-up
        # reported 28.1% without it and the figure then divided by the wrong
        # number.
        'shield_split_denominator': classified,
        'shield_h_negative_pct_of_classifiable': round(_pct(sh_both, classified), 2),
        'shield_h_nonnegative_pct_of_classifiable': round(_pct(sh_alone, classified), 2),
        'shield_h_nonnegative_pct_of_all_active': round(_pct(sh_alone, sh_active), 2),
        'shield_distance': q(sh_dist),
        'shield_dv_median_mps': (round(float(np.median(sh_dv)), 4)
                                 if sh_dv else None),

        'cbf_rows_cycles': cbf_rows,
        'cbf_danger_cycles': cbf_danger,
        'cbf_hneg_cycles': cbf_hneg,
        'cbf_rows_pct_of_total': round(_pct(cbf_rows, total_cycles), 2),
        'cbf_danger_pct_of_total': round(_pct(cbf_danger, total_cycles), 2),
        'shield_active_pct_of_total': round(_pct(sh_active, total_cycles), 2),

        'cbf_distance': {
            'row_exists': q(d_rows),
            'h_below_0_4': q(d_danger),
            'h_below_0': q(d_hneg),
        },
    }


def markdown(j: dict) -> str:
    cd, sd = j['cbf_distance'], j['shield_distance']

    def row(lbl, d):
        if not d['cycles']:
            return f'| {lbl} | – | – | – |'
        return (f'| {lbl} | {d["cycles"]:,} | {d["median_m"]:.3f} m | '
                f'{d["p25_m"]:.3f} – {d["p75_m"]:.3f} |')

    return f"""# CBF / Shield 歸因

由 `summarise_shield_attribution.py` 產生，資料來源 `{j['source_batch']}`
（{j['n_runs']} 趟，{j['total_cycles']:,} 個控制週期）。
**本檔與圖表、報告中的所有歸因數字皆來自同一份 `shield_attribution.json`。**

bag 前綴 `{j['bag_prefix']}`，比對到 {j['n_bags_matched']} 個。
commit `{(j['git_commit'] or '?')[:12]}`{'（工作區有未提交變更）' if j['git_dirty'] else ''}

{('> ⚠ 此 archive 另含其他實驗組的 bag，已排除：`' + str(j['other_prefixes_excluded']) + '`') if j['other_prefixes_excluded'] else ''}

## 誰在工作

| 事件 | 定義 | 週期數 | 佔全部 |
|---|---|---|---|
| CBF 建立約束 | QP 中至少一條屏障列 | {j['cbf_rows_cycles']:,} | {j['cbf_rows_pct_of_total']:.1f}% |
| CBF 進入危險區 | 最小屏障值 $h < {j['danger_threshold_h']}$ | {j['cbf_danger_cycles']:,} | {j['cbf_danger_pct_of_total']:.1f}% |
| CBF $h < 0$ | | {j['cbf_hneg_cycles']:,} | {100.0*j['cbf_hneg_cycles']/j['total_cycles']:.1f}% |
| Shield 介入 | 修改了輸出命令 | {j['shield_active']:,} | {j['shield_active_pct_of_total']:.2f}% |

## Shield 介入時 CBF 在想什麼

Shield 介入 **{j['shield_active']:,}** 次，其中 **{j['shield_cbf_classifiable']:,}** 次
可對齊到 {j['join_max_age_s']} s 內的 CBF 診斷值；其餘 **{j['shield_unclassified']:,}** 次
無可對齊診斷，未納入以下比例。

| CBF 狀態 | 週期數 | 佔可分類 |
|---|---|---|
| $h < 0$（兩層都認為危險） | {j['shield_with_h_negative']:,} | {j['shield_h_negative_pct_of_classifiable']:.1f}% |
| $h \\ge 0$（CBF 未判定為危險） | {j['shield_with_h_nonnegative']:,} | {j['shield_h_nonnegative_pct_of_classifiable']:.1f}% |

**分母是 {j['shield_split_denominator']:,}，不是 {j['shield_active']:,}。**
若誤用後者，$h \\ge 0$ 會變成 {j['shield_h_nonnegative_pct_of_all_active']:.1f}%。

$h \\ge 0$ 只表示兩層的危險判定不完全重合，**不足以**斷定障礙物未進入 CBF 集合 ——
那需要逐障礙物的集合成員診斷，目前未實作。

## 作用距離

「作用」的定義不同，距離差很多，因此三種定義一併列出。

| 狀態 | 週期數 | 距離中位 | p25 – p75 |
|---|---|---|---|
{row('CBF：QP 存在屏障列', cd['row_exists'])}
{row(f"CBF：且 $h < {j['danger_threshold_h']}$", cd['h_below_0_4'])}
{row('CBF：且 $h < 0$', cd['h_below_0'])}
{row('Shield：修改命令時', sd)}

CBF 在 {cd['h_below_0']['median_m']:.2f}–{cd['row_exists']['median_m']:.2f} m 就開始作用，
Shield 只在約 {sd['median_m']:.2f} m 介入 —— 無論採哪個定義，分工都一致。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', default='archive_gmpc100')
    ap.add_argument('--prefix', default='gmpc_cbf__scan_seed',
                    help='exact bag-name prefix; NOT a bare *seed* glob')
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--max-age', type=float, default=0.20,
                    help='max age of the gmpc diag joined to a shield cycle [s]')
    ap.add_argument('--out', default='evaluation/results/shield_attribution.json')
    a = ap.parse_args()

    j = collect(a.batch, a.prefix, a.limit, a.max_age)
    if not j['n_runs']:
        print(f'no usable bags in evaluation/bags/{a.batch}', file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(j, f, indent=2, ensure_ascii=False)
        f.write('\n')
    md = a.out.replace('.json', '').rsplit('/', 1)[0] + '/SHIELD_ATTRIBUTION_SUMMARY.md'
    with open(md, 'w') as f:
        f.write(markdown(j))
    print(f'  prefix {j["bag_prefix"]!r}: {j["n_bags_matched"]} bags matched, '
          f'{j["n_runs"]} usable, {j["total_cycles"]:,} cycles')
    if j['other_prefixes_excluded']:
        print(f'  EXCLUDED foreign bags in this archive: '
              f'{j["other_prefixes_excluded"]}')
    print(f'  shield {j["shield_active"]:,} active = '
          f'{j["shield_cbf_classifiable"]:,} classifiable + '
          f'{j["shield_unclassified"]:,} unclassified')
    print(f'  split over {j["shield_split_denominator"]:,}: '
          f'h<0 {j["shield_h_negative_pct_of_classifiable"]:.1f}% / '
          f'h>=0 {j["shield_h_nonnegative_pct_of_classifiable"]:.1f}%')
    print(f'  saved -> {a.out}\n  saved -> {md}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
