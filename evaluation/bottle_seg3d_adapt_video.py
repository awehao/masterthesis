#!/usr/bin/env python3
"""有界自適應觀察對照影片（Codex 20261101_232935 第 3 項、234651）：低桌換視角後接受 vs 高桌兩次調整後拒絕，並列；感知採集，非自主抓放。
素材只有兩趟保存的手腕相機影格（觀察段 5 Hz、其餘每 1 s 一張；無全景，採集紀錄沒有逐步全身狀態可供重播，不補造）。
時間：以原始擷取時刻（任務時刻）對齊，固定 10 fps、1 倍速；每個輸出影格顯示該時刻之前最新的一張擷取影格（保持時間比例）。
疊圖：同一模型離線重算的遮罩（標「離線重算疊圖」），只供看圖；旗標、缺項、視點與決策一律取自當趟旁路紀錄（seg3d_w1.npz）與 decisions.jsonl，不用重算結果改寫。
佈置（桌高、瓶位）只作標題說明，策略沒有讀它。
    .venv-dl0/bin/python evaluation/bottle_seg3d_adapt_video.py OUT.mp4
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_replay_video as RV                        # noqa: E402
import bottle_seg_depth3d as S3                         # noqa: E402
from bottle_seg_train import acquire_lock               # noqa: E402

RUNS = (('seg3d_ad_z045_p1b', '低桌 0.45 m／瓶位 P1'), ('seg3d_ad_z075_p2b', '高桌 0.75 m／瓶位 P2'))
W, H, FPS, CW, CH = 1920, 1080, 10, 880, 660
VIEW_ZH = dict(V0='V0 固定俯視', V_LOW='V_LOW 瞄準較低', V_HIGH='V_HIGH 相機較高')
DEF_ZH = dict(bottom_out_of_frame='瓶底出畫', top_cut='瓶頂出畫', top_background_unknown='瓶頂外側無深度', top_unconfirmed='瓶頂未確認', support_unobserved='支撐不可觀測',
              no_target='無有效實例', xy_invalid='xy 無效', incomplete_rows='同列不完整', stale='資料逾期')
STATE_ZH = ('接近', '移動視點', '觀察中', '回收', '結束')


def load(run, model, dev):
    D = os.path.join(HERE, 'runs', run)
    F = np.load(os.path.join(D, 'w1_frames.npz'))
    L = np.load(os.path.join(D, 'w1_log.npz'))
    lc = {str(k): L['rows'][:, i] for i, k in enumerate(L['cols'])}
    off = float(np.median(lc['stamp'] - lc['task_t']))
    t = np.array(F['stamp'], float) - off
    ov = []
    for i in range(len(t)):
        img = F['rgb'][i].copy()
        masks, recs, _ = S3.SI.predict(model, Image.fromarray(img), dev)
        if len(masks):
            m = masks[int(np.argmax([r['confidence'] for r in recs]))]
            img[m] = (img[m] * 0.45 + np.array([0, 255, 0]) * 0.55).astype(np.uint8)
        ov.append(Image.fromarray(img).resize((CW, CH)))
    Z = np.load(os.path.join(D, 'seg3d_w1.npz'))
    fs = ['recv_t'] + str(Z['label']).split('fields=')[1].split(',')
    seg = sorted(((r_['stamp'] - off, r_) for r_ in (dict(zip(fs, r)) for r in Z['rows'])), key=lambda x: x[0])
    dec = [json.loads(l) for l in open(os.path.join(D, 'decisions.jsonl'))]
    C = np.load(os.path.join(D, 'collect_log.npz'))
    cl = {str(k): C['rows'][:, i] for i, k in enumerate(C['cols'])}
    rep = json.load(open(os.path.join(D, 'collect_report.json')))
    return dict(t=t, ov=ov, seg=seg, dec=[d for d in dec if 'final' not in d], final=next((d['final'] for d in dec if 'final' in d), None), cl=cl, rep=rep)


def panel(F, d, x0, y0, title, tn):
    k = int(np.searchsorted(d['t'], tn, side='right')) - 1
    if k >= 0:
        F.paste(d['ov'][k], (x0, y0 + 40))
        age = tn - d['t'][k]
    else:
        age = None
    dr = ImageDraw.Draw(F, 'RGBA')
    dr.text((x0, y0), title, font=RV.F_M, fill=RV.INK)
    j = int(np.searchsorted(d['cl']['t'], tn, side='right')) - 1
    st = STATE_ZH[int(d['cl']['phase'][j])] if j >= 0 else '—'
    vi = int(d['cl']['view'][j]) if j >= 0 else -1
    view = ('V0', 'V_LOW', 'V_HIGH')[vi] if vi >= 0 else None
    vz = VIEW_ZH.get(view, '') if view else ''
    lab = f'{st}　{vz}' if st == '觀察中' else (f'{st}（自 {vz}）' if vz else st)   # 移動／回收時標出發視點，避免誤讀成目的地
    b = dr.textbbox((x0 + 10, y0 + 50), lab, font=RV.F_S)
    dr.rectangle([b[0] - 6, b[1] - 4, b[2] + 6, b[3] + 4], fill=(16, 22, 28, 200))
    dr.text((x0 + 10, y0 + 50), lab, font=RV.F_S, fill=RV.C2)
    if age is not None and age > 0.35:
        dr.text((x0 + CW - 10, y0 + 50), f'影格保持 {age:.1f} s（移動段每 1 s 一張）', font=RV.F_L, fill=RV.MUTED, anchor='ra')
    yy = y0 + 40 + CH + 10
    rows = [r for ts, r in d['seg'] if ts <= tn + 1e-9]
    if rows:
        r = rows[-1]
        fl = lambda key: '✓' if r[key] > 0.5 else '×'                                     # noqa: E731  （✗ 字型沒有）
        txt = (f"旁路（擷取 {r['stamp'] - (d['seg'][0][1]['stamp'] - d['seg'][0][0]):.1f} s）：實例 {fl('valid')}　xy {fl('xy_valid')}　瓶頂 {fl('top_observed')}　支撐 {fl('support_observed')}　"
               f"高度 {fl('height_valid')}　上緣截斷 {fl('top_cut')}　下緣截斷 {fl('bottom_cut')}　上緣外無深度 {r['edge_top_unknown']:.2f}" if r['valid'] > 0.5 else '旁路：無有效實例')
        dr.text((x0, yy), txt, font=RV.F_L, fill=RV.INK)
    on = [x for x in d['dec'] if x['t_decide'] <= tn < x['t_decide'] + 3.0]
    if on:
        x = on[-1]
        miss = '、'.join(DEF_ZH.get(z, z) for z in (x['deficits'] or []))
        if x['action'] == 'move':
            act = f"缺 {miss} ⇒ 移到 {VIEW_ZH.get(x['arg'], x['arg'])}"
        elif x['action'] == 'accept':
            act = f"同列完整 {x['n_complete']}/{x['n']} ⇒ 接受"
        else:
            act = f"缺 {miss}，已調整兩次 ⇒ 拒絕"
        s = f"決策（{VIEW_ZH.get(x['view'], x['view'])}，{x['t_decide']:.2f} s）：{act}"
        b = dr.textbbox((x0, yy + 30), s, font=RV.F_M)
        dr.rectangle([b[0] - 8, b[1] - 6, b[2] + 8, b[3] + 6], fill=(16, 22, 28, 230))
        dr.text((x0, yy + 30), s, font=RV.F_M, fill=RV.GOOD if x['action'] == 'accept' else (RV.C2 if x['action'] == 'move' else (240, 120, 110)))
    elif d['final'] and tn >= d['dec'][-1]['t_decide'] + 3.0:
        fz = '接受（取得符合條件的觀察）' if d['final']['action'] == 'accept' else '拒絕（明確不採用）'
        dr.text((x0, yy + 30), f"最終：{fz}；視點 {' → '.join(d['rep']['views_tried'])}", font=RV.F_M, fill=RV.MUTED)


def main():
    out = sys.argv[1]
    lock = acquire_lock()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = S3.load_model(os.path.join(HERE, 'dl0_ckpt', 'bottle_seg_v1_continue3', 'best.pth'), dev)
    data = [(lab, load(run, model, dev), run) for run, lab in RUNS]
    lock.close()
    t0 = 14.0
    t1 = max(max(d['dec'][-1]['t_decide'] for _, d, _ in data) + 6.0, 0.0)
    pipe = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '20',
                             '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    n = 0
    for tn in np.arange(t0, t1, 1.0 / FPS):
        F = Image.new('RGB', (W, H), RV.BG)
        dr = ImageDraw.Draw(F)
        dr.rectangle([0, 0, W, 64], fill=RV.BAR)
        dr.text((24, 14), '有界自適應觀察：低桌換視角後接受／高桌兩次調整後拒絕', font=RV.F_T, fill=RV.INK)
        dr.text((W - 24, 20), f'感知採集，非自主抓放　任務時刻 {tn:5.1f} s　1 倍速　各 n＝1', font=RV.F_S, fill=RV.MUTED, anchor='ra')
        for i, (lab, d, run) in enumerate(data):
            panel(F, d, 40 + i * (CW + 80), 84, f'{lab}（{run}）', float(tn))
        dr.text((24, H - 56), '手腕相機實際擷取影格（觀察段 5 Hz、其餘每 1 s 一張）；綠色遮罩＝同模型離線重算疊圖，只供看圖；旗標與決策取自當趟旁路紀錄與 decisions.jsonl。', font=RV.F_L, fill=RV.MUTED)
        dr.text((24, H - 30), '無全景：採集紀錄沒有逐步全身狀態可供重播，不補造。佈置只作標題說明，策略沒有讀桌高或佈置編號。接受≠抓取成功；中心含已知半徑先驗。', font=RV.F_L, fill=RV.MUTED)
        pipe.stdin.write(F.tobytes())
        n += 1
    pipe.stdin.close()
    rc = pipe.wait()
    json.dump(dict(runs=[r for r, _ in RUNS], fps=FPS, n_frames=n, t_range=[t0, float(t1)], kind='手腕相機擷取影格＋離線重算遮罩疊圖；非 Isaac 重播',
                   decisions={run: [dict(view=x['view'], t=x['t_decide'], action=x['action'], arg=x['arg'], deficits=x['deficits']) for x in d['dec']] for _, d, run in data}),
              open(out + '.keymoments.json', 'w'), ensure_ascii=False, indent=1)
    print(f'寫出 {out}，{n} 幀，ffmpeg rc {rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
