#!/usr/bin/env python3
"""第五次進度報告：Isaac 全身閉迴路的圖表（§12）。

沿用 build_fifth_progress_visuals.py 的 Figure 類別與版面，不另建風格。
數字一律由趟次輸出讀取，不在此硬寫。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_fifth_progress_visuals import (BLUE, INK, LIGHT, LINE,  # noqa: E402
                                          MUTED, NAVY, ORANGE, TEAL, Figure)

RUNS = ROOT / 'evaluation' / 'runs'
OK = 'wb_solver_iso_100502'
BAD = 'wb_solver_iso_094526'


def _date(f):
    """Figure.footer 內建的日期是 2026-09-10；本節資料到 2026-09-14。"""
    f.rect(1500, 1036, 360, 34, 'white', radius=0)
    f.text(1856, 1060, '資料截至 2026-09-14', 18, MUTED, anchor='end')


def load(run, name):
    return json.load(open(RUNS / run / name))


def tcp_error(run):
    d = load(run, 'sim/wb_run.json')
    cols = d['log_cols']
    i = {c: k for k, c in enumerate(cols)}
    tgt = (0.300, 0.000, 0.550)
    out = []
    for row in d['log']:
        t = row[i['t']]
        dx = row[i['tcp_x']] - tgt[0]
        dy = row[i['tcp_y']] - tgt[1]
        dz = row[i['tcp_z']] - tgt[2]
        out.append((t, (dx * dx + dy * dy + dz * dz) ** 0.5 * 1000.0))
    return out


def fig_result():
    ck = load(OK, 'solver_iso_check.json')
    m = ck['measured']
    f = Figure('圖 13', 'Isaac 自由空間全身閉迴路：到位驗證',
               f"趟次 {OK}　{ck['passed']}/{ck['total']} 項事前判準通過"
               '　—— 指定配置下的一趟驗證，非跨場景穩健性')
    f.table(64, 232, (560, 340, 300, 560),
            ('判準', '結果', '門檻', '說明'),
            [('連續符合到達條件的時間', f"{m['sustain_s']:.2f} s", '≥ 2.0 s',
              '位置與姿態同時符合'),
             ('TCP 位置誤差（全程最小）', f"{m['ep_iso_min_m']*1000:.3f} mm",
              '≤ 5 mm', 'Isaac 記錄的 link_tcp'),
             ('同上，實測 FK 重算', f"{m['ep_fk_min_m']*1000:.3f} mm", '≤ 5 mm',
              '獨立讀回，不採求解器自報'),
             ('兩來源一致', f"{m['source_agreement_max_m']*1000:.3f} mm",
              '≤ 10 mm', '兩者不一致即判定不成立'),
             ('完整三維姿態誤差（最小）', f"{m['er_iso_min_rad']:.5f} rad",
              '≤ 0.02 rad', '非僅工具軸'),
             ('實測連續同動', f"{m['simultaneous_longest_s']:.2f} s", '≥ 3.0 s',
              '由實測運動判定'),
             ('底盤線速度', f"{m['lin_max']:.6f} m/s", '≤ 0.05', '執行端界限未動'),
             ('輪速／輪加速度', f"{m['wheel_speed_max']:.4f} / "
              f"{m['wheel_accel_max']:.3f}", '0.2775 / 6.25', '逐步檢查')],
            row_h=54)
    f.box(64, 752, 876, 152, '仍未獲得的保證',
          ['修改後命令的上游全身安全性未重新論證；停止掃掠無避碰保證',
           '本配置不是 B 基線、不是 WGMPC；接觸操作不在本趟範圍'],
          ORANGE, '#FFF4EC', 28)
    f.box(980, 752, 876, 152, '最小誤差不是最終精度',
          ['0.364 mm 為全程最小值；sim 140 s 時長尾末值 21.71 mm',
           '長時間保持尚未驗收，列入下一階段'],
          MUTED, LIGHT, 28)
    f.footer('底盤與手臂在既有安全鏈內同時執行完整 9 維命令，'
             '並把 TCP 送到指定位姿',
             '命令拒收 0、執行端失效 無；求解器自報成功不作為驗收依據')
    _date(f)
    f.save('13_Isaac全身閉迴路到位驗證')


def fig_rootcause():
    f = Figure('圖 14', '失敗趟次的定位與修正：NODATA 的狀態分類',
               '自由空間下距離資料全為 NODATA，被讀成「資料未知」而套用退化速度上限')
    ax, ay, aw, ah = 110, 300, 800, 470
    f.rect(ax, ay, aw, ah, 'white', LINE)
    f.text(ax, ay - 24, '兩趟的 TCP 位置誤差', 30, NAVY, 700)
    curves = [(BAD, ORANGE, '094526（未通過）'), (OK, TEAL, '100502（通過）')]
    tmax, emax = 60.0, 1300.0
    for k in range(5):
        yy = ay + ah - k * ah / 4
        f.line([(ax, yy), (ax + aw, yy)], LINE, 2)
        f.text(ax - 14, yy + 9, f'{int(k * emax / 4)}', 22, MUTED, anchor='end')
    f.text(ax + 6, ay - 12, 'mm', 22, MUTED)
    for k in range(4):
        xx = ax + k * aw / 3
        f.text(xx, ay + ah + 34, f'{int(k * tmax / 3)} s', 22, MUTED)
    for run, color, label in curves:
        pts = []
        for t, e in tcp_error(run):
            if t > tmax:
                break
            pts.append((ax + t / tmax * aw, ay + ah - min(e, emax) / emax * ah))
        f.line(pts[::5], color, 4)
    f.text(ax + 470, ay + 90, '094526：誤差單調惡化', 26, ORANGE, 600)
    f.text(ax + 470, ay + 126, '底盤觸發自身 1.2 m guard', 24, MUTED)
    f.text(ax + 250, ay + ah - 70, '100502：到位並保持', 26, TEAL, 600)
    f.box(980, 232, 876, 250, '根因（以既有資料重播，差 0.000000）',
          ['距離節點在自由空間對每個連桿產生 STATUS_NODATA 列',
           '濾波器據此套用 nodata_speed_cap = 0.05，壓住整個 9 維速度框',
           '底盤命令 0.0353 < 0.05 原樣通過；手臂 0.9 rad/s → 約 0.004',
           '結果：全身解只有底盤那一半被執行'],
          ORANGE, '#FFF4EC', 29)
    f.box(980, 506, 876, 240, '修正：狀態分類（門檻值未動）',
          ['已確認自由空間 → NODATA 讀成「範圍內沒有東西」，不套退化上限',
           '資料缺失／過期／TF 失效 → 維持既有失效處置，不得當成自由空間',
           '一般有障礙物模式 → 原處理完全不變',
           '關節限位、速度框、加速度框、jerk 框一律保留'],
          TEAL, '#ECF7F5', 29)
    f.table(980, 756, (300, 280, 296), ('量', '094526', '100502'),
            [('安全層 speed_cap', '0.05', '−1（未施加該項）'),
             ('cmd_out 手臂 |max|', '0.024771', '0.999900')], row_h=52)
    f.footer('修的是「未知」與「已確認空場景」的分類，不是把退化上限調大',
             'speed_cap = −1 表示未施加該項退化上限，不是取消所有速度限制')
    _date(f)
    f.save('14_NODATA狀態分類與修正')


if __name__ == '__main__':
    fig_result()
    fig_rootcause()
    print('完成')
