#!/usr/bin/env python3
"""瓶子主線交付物：讀 資料/瓶子主線總表.json 與 規格/瓶子主線_成果摘要.md，產生總表與來源 Markdown 及單頁 HTML。
    python3 第七次進度報告素材/產生程式/bottle_mainline_render.py <輸出.html>
"""
import html
import json
import os
import re
import sys

M = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(M, '資料', '瓶子主線總表.json')))
ROWS, PROV = D['rows'], {p['run']: p for p in D['provenance']}
OUT_MD = os.path.join(M, '規格', '瓶子主線_總表與來源.md')
LIMIT = {'bottle_s1_5': '位姿真值、路徑給定；s1_6、s1_7 逐位元相同，不算獨立樣本', 'bottle_s1_g1': '幾何小擾動，不是隨機重複', 'bottle_s1_g2': '同上', 'bottle_s1_g3': '同上', 'bottle_s1_g4': '同上',
         'bottle_s1_p1': '只有名目一趟；CBF 重解未能重現實際命令，未採用', 'bottle_s1_m1': '低調參移植基線；G 投入遠多於 M', 'bottle_s1_a1': '消融單趟；峰值減速度差異未判定',
         'bottle_n1_2': '掃描與命令走模擬 ROS 話題；位姿真值；牆鐘往返 p95 40.9 ms', 'bottle_n2_1': '起跑前規劃一次、不重規劃；路線不同，完成時間不比效率；牆鐘往返最高 156.7 ms',
         'bottle_s2_1': '瓶距未達 50 mm 目標；主要底盤讓路', 'bottle_s2_2': '方塊真值；主要底盤讓路；非夾爪連桿未入約束', 'bottle_s3_1': '方塊真值；底盤對牆為模型值；連桿列未貼作用邊界'}
GROUPS = [('', ['趟次', '條件']), ('任務', ['結果', '到出口 s', '完成 s', '滑移 mm／°']), ('距離', ['禁止接觸峰 N', '瓶淨空 mm', '懸空方塊 連桿／瓶 mm']),
          ('連續性', ['抓取段車速 m/s', '抓取段 s', '全程近停 s', '峰值減速 m/s²']), ('耗時', ['solve p50／p95／max ms', '超過 50 ms']), ('介入', ['CBF 改命令', 'Shield']), ('', ['限制'])]
NOTES = ['到出口：底盤首次進入半徑 0.3 m 的出口區。完成：在出口區內停下。',
         '滑移：持瓶相對 TCP 的最大漂移，判準 5 mm／3°。M 未抓到瓶，無此項。',
         '禁止接觸峰：`forbidden_pulses` 各通道的最大力。判定為 ≥0.05 N 連續 3 物理步，或穿透 ≥1 mm。',
         '瓶淨空：`clear_bottle.meas_min`，非手指機體對瓶的最小實測淨空。拒絕線 5 mm、保護帶 0.5 mm，實際觸發線 5.5 mm。',
         '懸空方塊：量測位姿重建的點雲／表面採樣最小距離，逐步計算，目標 50 mm。雙指都以 `finger1_q` 重建，不是實體幾何的精確最小距離。',
         '抓取段車速：閉爪到抬升開始的量測底盤速度，5 步平滑。抓取段：閉爪到離桌成立。',
         '全程近停：車速低於 0.01 m/s 的累計秒數，算到完成或停止，含起步與到站。',
         '峰值減速：車速在 0.10 s 差分窗的最大下降，排除前 0.2 s。',
         'solve：`solve_ms_all`，每控制週期的組裝與 QP 耗時，不含掃描、Shield、物理與渲染。M 的耦合求解是 GI 324、OSQP 103，其餘趟的耦合 QP 均採用 GI；每週期先解的底盤 hold 問題仍用 OSQP。',
         'CBF 改命令：同輸入移除 CBF 重解，命令差 ≥ 1e-3 且 CBF 列作用的週期數／總週期。未重解不等於零次，P 有 CBF，只是重解未能重現命令；S2、S3 的重解工具不含三維約束，所以未列。',
         'Shield：物理趟中非微量改寫命令的次數。',
         'N1、N2：掃描、障礙與底盤命令走模擬 ROS 話題，追蹤器與 Shield 是獨立節點；控制與物理同步，命令總延遲固定 8 物理步（實測往返 10 ms＋注入補足 70 ms，模擬時間）。n2_1 的 CBF 次數較少是因為規劃路線離方塊較遠（CRUISE 375→215），抓取各相位次數相同。',
         '地面方塊（掃描鏈會看到）對底盤中心的面距離，各趟皆 0.528 m（真值評估），未列欄。']


def n(v, f='.2f'):
    return '—' if v is None else format(v, f)


def cells(e):
    t = '成功' if e['ok'] else '失敗'
    box = f"{e['box']['link_mm']:.1f}／{e['box']['bottle_mm']:.1f}" if 'box' in e else '—'
    v = f"{e['v_grip'][0]:.3f}–{e['v_grip'][1]:.3f}" if e['v_grip'] else '—'
    if e['run'] == 'bottle_s1_p1':
        v = '0（停住）'
    slip = f"{e['slip_mm']:.2f}／{e['slip_deg']:.2f}" if e['slip_mm'] is not None else '—'
    ms = e['ms']
    cbf = f"{e['cbf']['changed']}／{e['cbf']['n']}" if 'cbf' in e and e['run'] != 'bottle_s1_p1' else ('重解未能重現，未採用' if e['run'] == 'bottle_s1_p1' else '未重解')
    return [e['run'].replace('bottle_', ''), e['label'], t, n(e['t_exit']), n(e['t_done']), slip, f"{e['forb_peak_n']:.2f}", f"{e['clear_bottle_mm']:.2f}", box, v, n(e['t_grip']), f"{e['near_run_s']:.2f}", f"{e['dec_peak']:.2f}",
            f"{ms['p50']:.1f}／{ms['p95']:.1f}／{ms['max']:.1f}", f"{ms['over50']}／{ms['n']}", cbf, str(e['shield']['active']), LIMIT[e['run']]]


def prov_cells(r):
    p = PROV[r]
    return [r.replace('bottle_', ''), p['time'][5:], p['args'] or '（無）', f"{p['first']}–{p['last']}", p['pick']['gmpc_wb.py'], p['pick']['bottle_s1_ctrl.py'], p['sha']['s1_steps.npz']]


PROV_HEAD = ['趟次', '時間', '實跑參數', 'git 區間', 'gmpc_wb', 'bottle_s1_ctrl', 's1_steps.npz']
PROV_NOTES = ['資料位置：`evaluation/runs/<趟次全名>/`（`s1_report.json`、`s1_steps.npz`、`s1_cbf_replay.json`、`versions.sha256`）。雜湊皆取 sha256 前段。',
              'git 區間：`versions.sha256` 列的檔案內容在這段 commit 內都與實跑相同，起點是該內容首次提交的版本。',
              'G 控制配置凍結於 8719b41d9。`gmpc_wb` 29ae4fa4＋`bottle_s1_ctrl` 3af3e4fa 在 s1_5、g1–g4、p1、m1 相同。a1 的控制檔不同（加消融開關）。S2、S3 的核心與控制檔都不同，是候選配置。',
              '封存導航 `gmpc.py`（2a1e2148b0d4bc22）全部 11 趟相同；`gmpc.py`、`gmpc_node.py` 最後修改於 a10568326（2026-10-30），早於本主線。',
              's1_5、s1_6、s1_7 的逐步紀錄逐位元相同，只算一個樣本。bottle_n1_1 與 n1_2 的逐步紀錄也逐位元相同（s1_steps.npz 雜湊 f13bbf6d），不是兩個樣本。',
              'N1、N2 用的 gmpc_wb 052b80bf、bottle_s1_ctrl 69df3798 是 S3 時期的版本（三維約束未啟用），不是凍結的 G 版本；和 s1_5 的比較跨版本。',
              '環境：Isaac Sim 6.0.1，物理步 0.01 s，TGS。run 紀錄未存完整機器指紋；溫度來源 k10temp，與目前機器 Ryzen 7 9800X3D＋RTX 5070 相符。',
              'N1、N2 跑法：`evaluation/run_bottle_ros.sh`（N2 加 `PLAN=1 ISAAC_SCRIPT=evaluation/isaac_bottle_n2.py`）；規劃參數 `evaluation/bottle_n2_planner.yaml`，完整路徑在 `s1_report.json` 的 plan 欄。bottle_n1_1 任務逐步紀錄完整，但報告寫檔崩潰、rc 1，原因未定位，不算完整驗收。',
              '未進總表的趟次：s1_1、s1_2 手臂姿態已否決；s1_3 障礙真值輸入、未接掃描鏈；s1_4 掃描鏈首趟（s1_5 的前一版）；s0a、s0b 桌瓶物理夾持前置；probe、smoke 開發測試。',
              '距離更正：`瓶子S1_總表.md` 舊值 s2_1 52.1、s2_2 61.5、s3_1 63.2 mm 是每 5 步取樣、手指開度固定算的。本表逐步重算並用量測手指開度。',
              '總表資料：`資料/瓶子主線總表.json`。產生程式：`產生程式/bottle_mainline_table.py`、`產生程式/bottle_mainline_render.py`。']


def write_md():
    head = [h for _, hs in GROUPS for h in hs]
    L = ['# 瓶子主線總表與來源', '', '由 `產生程式/bottle_mainline_render.py` 產生，請勿手改。各條件 n＝1，Isaac 物理。', '', '## 總表', '', '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    L += ['| ' + ' | '.join(cells(e)) + ' |' for e in ROWS]
    L += ['', '欄位定義：', ''] + ['- ' + x for x in NOTES]
    L += ['', '## 程式版本、run 與資料來源', '', '| ' + ' | '.join(PROV_HEAD) + ' |', '|' + '---|' * len(PROV_HEAD)]
    L += ['| ' + ' | '.join(x.replace('|', '｜') for x in prov_cells(e['run'])) + ' |' for e in ROWS]
    L += [''] + ['- ' + x for x in PROV_NOTES]
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)


def sections():
    txt = open(os.path.join(M, '規格', '瓶子主線_成果摘要.md'), encoding='utf-8').read().strip()
    parts = re.split(r'^## ', txt, flags=re.M)
    lines = [x for x in parts[0].split('\n') if x.strip()]
    sec = [(p.split('\n')[0], [x[2:] for x in p.split('\n')[1:] if x.startswith('- ')]) for p in parts[1:]]
    return lines[0][2:], lines[1], lines[2], sec


CSS = """
/* 版面：一欄結論，三段成果與限制並排，寬表自捲；測試單的語氣，藍灰底配一個儀表藍 */
:root{--bg:#eef1f4;--sheet:#fbfcfd;--ink:#17212b;--muted:#566472;--rule:#d3dae1;--accent:#0a6a86;--good:#1b7a4c;--bad:#b4342b;--warn:#8f6200;--band:#e4eaef;
--disp:"Noto Serif TC","Songti TC",serif;--body:"IBM Plex Sans","Noto Sans TC","PingFang TC",sans-serif;--mono:"IBM Plex Mono","Noto Sans TC",monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#10161c;--sheet:#172029;--ink:#e6ebf0;--muted:#9eabb7;--rule:#2d3944;--accent:#67bdd8;--good:#68c999;--bad:#f08e84;--warn:#e2b75f;--band:#1f2a34;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#10161c;--sheet:#172029;--ink:#e6ebf0;--muted:#9eabb7;--rule:#2d3944;--accent:#67bdd8;--good:#68c999;--bad:#f08e84;--warn:#e2b75f;--band:#1f2a34;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:16px;line-height:1.7;padding-block:32px 56px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.wrap{max-width:1120px;margin:0 auto;display:grid;gap:32px}
header{display:grid;gap:8px}
.meta{font-family:var(--mono);font-size:13px;color:var(--muted);letter-spacing:.02em}
h1{font-family:var(--disp);font-size:clamp(28px,4vw,38px);line-height:1.25;font-weight:700;margin:0;text-wrap:balance}
.lead{font-size:18px;line-height:1.75;max-width:44em;margin:0}
.grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(0,1fr);gap:32px;align-items:start}
.main{display:grid;gap:28px;min-width:0}
section h2,aside h2,.block h2{font-family:var(--disp);font-size:20px;margin:0 0 10px;text-wrap:balance}
ul{margin:0;padding-left:1.15em;display:grid;gap:8px}
li{max-width:40em}
aside{background:var(--sheet);border:1px solid var(--rule);padding:20px 22px;min-width:0}
aside li{font-size:15px}
code{font-family:var(--mono);font-size:.9em}
.block{display:grid;gap:12px;min-width:0}
.scroll{overflow-x:auto;background:var(--sheet);border:1px solid var(--rule)}
table{border-collapse:collapse;font-size:13px;line-height:1.45;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:7px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--rule);white-space:nowrap}
thead th{background:var(--band);font-weight:600;color:var(--ink)}
thead tr.g th{font-family:var(--disp);font-size:14px;color:var(--accent);border-bottom:0}
td.wrap-cell{white-space:normal;min-width:15em}
td.mono{font-family:var(--mono);font-size:12px}
tbody tr:last-child td{border-bottom:0}
.st{font-weight:600}.st.ok{color:var(--good)}.st.no{color:var(--bad)}
.under{color:var(--warn);font-weight:600}
.notes{margin:0;padding-left:1.15em;font-size:13.5px;color:var(--muted);display:grid;gap:4px}
.notes li{max-width:62em}
@media (max-width:820px){.grid{grid-template-columns:minmax(0,1fr)}}
"""


def table_html():
    g = ''.join(f'<th colspan="{len(hs)}">{html.escape(t)}</th>' for t, hs in GROUPS)
    h = ''.join(f'<th>{html.escape(x)}</th>' for _, hs in GROUPS for x in hs)
    body = []
    for e in ROWS:
        c = [html.escape(x) for x in cells(e)]
        c[2] = f'<span class="st {"ok" if e["ok"] else "no"}">{c[2]}</span>'
        if 'box' in e and e['box']['bottle_mm'] < 50:
            c[8] = f'<span class="under">{c[8]} 未達標</span>'
        body.append('<tr>' + ''.join(f'<td>{x}</td>' for x in c[:-1]) + f'<td class="wrap-cell">{c[-1]}</td></tr>')
    return f'<div class="scroll" tabindex="0" role="region" aria-label="逐趟總表"><table><thead><tr class="g">{g}</tr><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def prov_html():
    h = ''.join(f'<th>{x}</th>' for x in PROV_HEAD)
    body = []
    for e in ROWS:
        c = [html.escape(x) for x in prov_cells(e['run'])]
        body.append('<tr>' + ''.join(f'<td class="mono">{x}</td>' if i in (3, 4, 5, 6) else f'<td>{x}</td>' for i, x in enumerate(c)) + '</tr>')
    return f'<div class="scroll" tabindex="0" role="region" aria-label="來源表"><table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def write_html(path):
    title, meta, lead, sec = sections()
    main = ''.join(f'<section><h2>{html.escape(t)}</h2><ul>{"".join(f"<li>{inline(x)}</li>" for x in it)}</ul></section>' for t, it in sec[:-1])
    t, it = sec[-1]
    aside = f'<aside><h2>{html.escape(t)}</h2><ul>{"".join(f"<li>{inline(x)}</li>" for x in it)}</ul></aside>'
    notes = ''.join(f'<li>{inline(x)}</li>' for x in NOTES)
    pn = ''.join(f'<li>{inline(x)}</li>' for x in PROV_NOTES)
    page = f"""<title>{title}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&family=Noto+Sans+TC:wght@400;600&family=Noto+Serif+TC:wght@600;700&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header><div class="meta">{html.escape(meta)}</div><h1>{title}</h1><p class="lead">{inline(lead)}</p></header>
<div class="grid"><div class="main">{main}</div>{aside}</div>
<div class="block"><h2>逐趟總表</h2>{table_html()}<ul class="notes">{notes}</ul></div>
<div class="block"><h2>程式版本、run 與資料來源</h2>{prov_html()}<ul class="notes">{pn}</ul></div>
</div>
"""
    open(path, 'w', encoding='utf-8').write(page)


def main():
    assert all(e['fallback'] == 0 for e in ROWS)
    assert all(e['shield']['active'] == 0 for e in ROWS)
    assert all(PROV[e['run']]['gmpc'] == '2a1e2148b0d4bc22' for e in ROWS)
    write_md()
    write_html(sys.argv[1])
    print('寫出', OUT_MD, sys.argv[1])


if __name__ == '__main__':
    sys.exit(main())
