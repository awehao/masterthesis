#!/usr/bin/env python3
"""相機診斷的核對：影格有效、取景、封存。

**不**判定控制或抽屜 —— 這趟沒有命令來源。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('diag_dir')
    ap.add_argument('--min-frames', type=int, default=20)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    d = a.diag_dir
    rep, bad = {'dir': d}, []

    sim = os.path.join(d, 'sim/wb_run.json')
    if not os.path.exists(sim):
        bad.append('無 sim/wb_run.json ⇒ 模擬未正常封存')
        rep['stop_reason'] = None
    else:
        j = json.load(open(sim))
        rep['stop_reason'] = j.get('stop_reason')
        rep['sim_time_s'] = j.get('sim_time_s')
        rep['cpu_temp_max_c'] = j.get('cpu_temp_max_c')
        rep['record_frames'] = j.get('record_frames')
        if j.get('stop_reason') != 'sim_limit':
            bad.append(f'收尾不正常：stop_reason={j.get("stop_reason")!r}')
        if j.get('drawer') is None:
            bad.append('紀錄裡沒有抽屜資訊 ⇒ 場景不含抽屜')

    frames = sorted(glob.glob(os.path.join(d, 'frames', '*')))
    rep['n_frames'] = len(frames)
    if len(frames) < a.min_frames:
        bad.append(f'影格只有 {len(frames)} 張，少於 {a.min_frames}')
    else:
        # 影格要能讀、尺寸正確、**不是全黑或全白**
        # **讀圖的套件要擇一並記下用了哪個**：系統 python 沒有 imageio
        #（它在 Isaac 的虛擬環境裡），所以依序試 imageio → PIL；
        # 兩者都沒有就退回只讀 PNG 檔頭的尺寸 ＋ 檔案大小，並**明標**
        # 那樣只能支持「尺寸正確、不是純色」而不能支持像素統計。
        import numpy as np
        reader = None
        try:
            import imageio.v2 as iio
            reader = 'imageio'

            def _read(q):
                return np.asarray(iio.imread(q))
        except Exception:
            try:
                from PIL import Image
                reader = 'PIL'

                def _read(q):
                    return np.asarray(Image.open(q).convert('RGB'))
            except Exception:
                reader = None
        rep['frame_reader'] = reader
        try:
            picks = [frames[0], frames[len(frames) // 2], frames[-1]]
            info = []
            if reader is not None:
                for q in picks:
                    im = _read(q)
                    info.append({'file': os.path.basename(q),
                                 'shape': list(im.shape),
                                 'mean': float(im[..., :3].mean()),
                                 'std': float(im[..., :3].std())})
            else:
                # PNG 檔頭：簽章 8 bytes + IHDR 長度 4 + 'IHDR' 4 + 寬 4 + 高 4
                for q in picks:
                    with open(q, 'rb') as fh:
                        hdr = fh.read(24)
                    w = int.from_bytes(hdr[16:20], 'big')
                    h = int.from_bytes(hdr[20:24], 'big')
                    sz = os.path.getsize(q)
                    info.append({'file': os.path.basename(q),
                                 'shape': [h, w, 4], 'bytes': sz,
                                 'mean': None, 'std': None,
                                 'note': '無讀圖套件 ⇒ 只核尺寸與檔案大小'})
                rep['frame_check_limited'] = (
                    '**無讀圖套件**：只核了 PNG 檔頭尺寸與檔案大小，'
                    '沒有像素統計。純色畫面會壓得很小，所以用大小當代理，'
                    '但那不是像素證據')
            rep['frame_samples'] = info
            if any(i['shape'][0] != 720 or i['shape'][1] != 1280
                   for i in info):
                bad.append(f'影格尺寸不是 1280x720：{[i["shape"] for i in info]}')
            if reader is not None:
                if all(i['std'] < 1.0 for i in info):
                    bad.append('取樣影格的像素標準差全部 <1 ⇒ 畫面可能全黑／'
                               '全白（相機取景或算繪有問題）')
            else:
                # 1280x720 的純色 PNG 通常 <20 KB；有內容的場景遠大於此
                if all(i.get('bytes', 0) < 20000 for i in info):
                    bad.append(f'取樣影格都小於 20 KB '
                               f'（{[i.get("bytes") for i in info]}）⇒ '
                               f'可能是純色畫面。**這是檔案大小的代理證據，'
                               f'不是像素證據**')
        except Exception as e:
            bad.append(f'影格讀取失敗：{e!r}')

    rep['violations'] = bad
    rep['verdict'] = 'pass' if not bad else 'fail'
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1,
                  default=str)
    print(f'  相機診斷：{rep["verdict"]}；影格 {rep["n_frames"]} 張；'
          f'收尾 {rep.get("stop_reason")}')
    print(f'    讀圖套件：{rep.get("frame_reader")}')
    for sm in rep.get('frame_samples', []):
        if sm.get('std') is not None:
            print(f'    {sm["file"]} {sm["shape"]} 均值 {sm["mean"]:.1f} '
                  f'標準差 {sm["std"]:.1f}')
        else:
            print(f'    {sm["file"]} {sm["shape"]} {sm.get("bytes")} bytes '
                  f'（{sm.get("note")}）')
    for b in bad:
        print(f'    - {b}')
    return 0 if not bad else 1


if __name__ == '__main__':
    raise SystemExit(main())
