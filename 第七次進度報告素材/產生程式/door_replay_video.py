#!/usr/bin/env python3
"""門場景實錄重播影片：逐格疊上標註後以 ffmpeg 編碼（Howard／Codex 20261023_234952：重要動作實驗都要交影片，成功失敗都保留）。

輸入：重播輸出 runs/<REPLAY>/frames/f_NNNNN.png 與 frames.csv（run_door_replay.sh）＋來源趟 runs/<SRC>/task.json、room_run.json。
標註：RUN_ID、「該趟實錄重播」（不是即時錄影）、模擬時間、任務相位、門角、閂鎖後底盤位移、關鍵時刻橫幅
      （對準＝ALIGN_PRE／ALIGN_HOLD、夾持＝ENGAGE／HOLD、OPEN、失接觸／中止（策略中止）、閂鎖、停止後漂移＝閂鎖後）。
另寫 <out>.keymoments.json：關鍵時刻（模擬時間、影片時間）供 README 引用。
播放（r2；Codex 20261023_235904）：每幀顯示時長＝（下一幀模擬時間 − 本幀模擬時間）÷ 倍速，依**實際影格**的模擬時間編碼（ffmpeg concat），
缺圖時沿用前一幀並在頂欄標「缺圖」，不把缺幀壓縮成固定倍速；關鍵時刻的影片時間依實際影格累計。
完整性取自同目錄 replay_check.json（door_replay_check.py）；不完整時頂欄標「重播不完整」並寫入 keymoments.json。

    python3 evaluation/door_replay_video.py <replay_run_id> <src_run_id> <out.mp4> [--fps 20] [--title 文字]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
PHASE_ZH = {'WAIT_HANDOVER': '等待交棒', 'UNFOLD': '展開手臂', 'ALIGN_PRE': '對準', 'ALIGN_HOLD': '對準保持', 'APPROACH': '接近',
            'ENGAGE': '夾持', 'HOLD': '夾持保持', 'OPEN': '開門', 'SETTLE': '穩定', 'END_HOLD': '結束保持', 'DONE': '完成', 'ABORTED': '已中止'}


def key_moments(task, room):
    ev = [(e['t'], e['to']) for e in task.get('policy_events', []) if 'to' in e]
    km = []
    for t, ph in ev:
        if ph in ('ALIGN_PRE', 'ENGAGE', 'HOLD', 'OPEN'):
            km.append({'sim_t': t, 'label': {'ALIGN_PRE': '對準開始', 'ENGAGE': '夾持開始', 'HOLD': '夾持保持', 'OPEN': '開門（OPEN）開始'}[ph]})
    ab = next((e for e in task.get('policy_events', []) if 'abort' in e), None)
    if ab is not None:
        km.append({'sim_t': ab['t'], 'label': f"策略中止：{ab['abort']}"})
    lat = [e['k2_abort_latched'] for e in room.get('events', []) if 'k2_abort_latched' in e]
    if lat:
        km.append({'sim_t': lat[0]['sim_t'], 'label': '執行端閂鎖（命令歸零）'})
        km.append({'sim_t': lat[0]['sim_t'] + 1.0, 'label': '停止後漂移（閂鎖後）'})
    return sorted(km, key=lambda x: x['sim_t']), ev, (lat[0] if lat else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('replay')
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--speed', type=float, default=2.0, help='倍速（模擬時間 ÷ 影片時間）')
    ap.add_argument('--title', default='')
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} 已存在，不覆寫')
    RD = os.path.join(HERE, 'runs', a.replay)
    SD = os.path.join(HERE, 'runs', a.src)
    task = json.load(open(os.path.join(SD, 'task.json')))
    room = json.load(open(os.path.join(SD, 'room_run.json')))
    km, ev, lat = key_moments(task, room)
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    by_step = {int(s[I['step']]): s for s in room['steps']}
    b_lat = None if lat is None else by_step[int(lat['step'])][I['base_xyth']]
    rows = [r for r in csv.reader(open(os.path.join(RD, 'frames.csv'))) if r and not r[0].startswith('#') and r[0] != 'frame']
    chk_p = os.path.join(RD, 'replay_check.json')
    chk = json.load(open(chk_p)) if os.path.exists(chk_p) else {'complete': False, 'reasons': ['replay_check_missing']}
    f_big, f_mid = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 24)
    speed = a.speed
    tmp = tempfile.mkdtemp(prefix='doorvid_')
    try:
        n = 0
        t_first = float(rows[0][1]) if rows else 0.0
        concat, last_png, n_missing = [], None, 0
        for i, r in enumerate(rows):
            fr, st, step = int(r[0]), float(r[1]), int(r[2])
            src = os.path.join(RD, 'frames', f'f_{fr:05d}.png')
            t_next = float(rows[i + 1][1]) if i + 1 < len(rows) else st + 0.1
            dur = max(1e-3, (t_next - st) / speed)
            if not os.path.exists(src):                          # 缺圖：沿用前一幀（若有），時間照實保留
                n_missing += 1
                if last_png is not None:
                    concat.append((last_png, dur))
                continue
            im = Image.open(src).convert('RGB')
            d = ImageDraw.Draw(im, 'RGBA')
            ph = None
            for t, p in ev:
                if t <= st + 1e-9:
                    ph = p
            if lat is not None and st >= lat['sim_t'] - 1e-9:
                ph = 'ABORTED'
            s = by_step.get(step)
            q = None if s is None else math.degrees(s[I['door_q_rad']])
            line2 = f"模擬 {st:6.2f} s　相位 {PHASE_ZH.get(ph, ph or '—')}　門角 {q:.2f}°" if q is not None else f'模擬 {st:6.2f} s'
            if b_lat is not None and s is not None and st >= lat['sim_t'] - 1e-9:
                bx = s[I['base_xyth']]
                line2 += f"　閂鎖後底盤位移 {math.hypot(bx[0] - b_lat[0], bx[1] - b_lat[1]) * 1e3:.1f} mm"
            d.rectangle([0, 0, im.width, 92], fill=(0, 0, 0, 150))
            flag = '' if chk.get('complete') else f"　重播不完整（{'、'.join(chk.get('reasons', []))[:40]}）"
            if n_missing:
                flag += f'　缺圖 {n_missing}'
            d.text((16, 8), f"{a.src}　該趟實錄重播（非即時錄影；{speed:g} 倍速）{('　' + a.title) if a.title else ''}{flag}", font=f_mid, fill=(255, 255, 255))
            d.text((16, 46), line2, font=f_mid, fill=(255, 235, 160))
            cur = [k for k in km if k['sim_t'] <= st + 1e-9 and st - k['sim_t'] < 1.5]
            if cur:
                d.rectangle([0, im.height - 64, im.width, im.height], fill=(150, 30, 30, 170))
                d.text((16, im.height - 54), cur[-1]['label'], font=f_big, fill=(255, 255, 255))
            n += 1
            last_png = os.path.join(tmp, f'{n:05d}.png')
            im.save(last_png)
            concat.append((last_png, dur))
        if n == 0:
            raise SystemExit('沒有任何幀')
        lst = os.path.join(tmp, 'concat.txt')
        with open(lst, 'w') as f:
            for pth, dur in concat:
                f.write(f"file '{pth}'\nduration {dur:.6f}\n")
            f.write(f"file '{concat[-1][0]}'\n")                    # concat 規格：最後一格重複以套用時長
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-vsync', 'vfr',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', a.out], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for k in km:
        k['video_t_s'] = max(0.0, (k['sim_t'] - t_first) / speed)
    json.dump({'replay': a.replay, 'src': a.src, 'n_frames': n, 'n_missing_images': n_missing, 'speed_x': speed,
               'replay_complete': chk.get('complete'), 'replay_check_reasons': chk.get('reasons'),
               'timing': '每幀時長＝相鄰影格模擬時間差 ÷ 倍速（依實際影格）',
               'kind': '該趟實錄重播（render-only replay of the same run）', 'key_moments': km},
              open(a.out + '.keymoments.json', 'w'), ensure_ascii=False, indent=1)
    print(f'{a.out}：{n} 幀；關鍵時刻 {[(k["label"], round(k["video_t_s"], 1)) for k in km]}')


if __name__ == '__main__':
    main()
