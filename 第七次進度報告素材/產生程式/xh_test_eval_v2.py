#!/usr/bin/env python3
"""【v2，2026-10-21】XH3 測試評估的入口複本（attempt2 用）。逐格評估、彙總、閉合與完整性**直接匯入** xh_test_eval（凍結 r5，不改、不複製）。

差異只有入口：
  * 開封紀錄、趟次根目錄、輸出路徑都必須明示（--opening／--runs-root／--out），不再用模組常數，也不接受任意 opened 紀錄：
    紀錄須 state=opened、attempt=2，且紀錄內的 runs_root 與 test_eval_out 與本次參數一致；任一不符 ⇒ 拒絕（結束碼 2，不寫輸出）。
  * 環境指紋加機器欄位（DMI 產品名／廠商、CPU 型號、實際選用的 CPU 溫度來源），並與開發同環境參照逐欄對照（只記錄、不阻擋）。

    .venv-dl0/bin/python evaluation/xh_test_eval_v2.py --ckpt <ckpt> --opening <attempt2.json> --runs-root <根目錄> --out <輸出.json>
"""
import argparse
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import xh_test_eval as TE                              # noqa: E402  凍結版的 run／summarize／env_fingerprint

DEV_REF_ENV = os.path.join(HERE, 'results', 'vision', 'XH3_dev_rerender_msi_env.json')


def _read1(p):
    try:
        return open(p).readline().strip()
    except OSError as e:
        return f'NA（{type(e).__name__}）'


def machine_fingerprint():
    import cpu_temp
    cpu = 'NA'
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    except OSError as e:
        cpu = f'NA（{type(e).__name__}）'
    return {'dmi_vendor': _read1('/sys/class/dmi/id/sys_vendor'), 'dmi_product': _read1('/sys/class/dmi/id/product_name'),
            'cpu_model': cpu, 'cpu_temp_source': cpu_temp.read()[1]}


def env_fingerprint_v2(torch):
    return {**TE.env_fingerprint(torch), **machine_fingerprint()}


def check_record(rec, runs_root, out):
    """回傳拒絕原因清單（空＝通過）。路徑一律以工作區相對路徑比較。"""
    why = []
    if not isinstance(rec, dict):
        return ['record_unreadable']
    if rec.get('state') != 'opened':
        why.append(f"state:{rec.get('state')}")
    if rec.get('attempt') != 2:
        why.append(f"attempt:{rec.get('attempt')}")
    if rec.get('runs_root') != os.path.relpath(os.path.abspath(runs_root), WS):
        why.append(f"runs_root_mismatch:{rec.get('runs_root')}")
    if rec.get('test_eval_out') != os.path.relpath(os.path.abspath(out), WS):
        why.append(f"out_mismatch:{rec.get('test_eval_out')}")
    return why


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--opening', required=True)
    ap.add_argument('--runs-root', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print(f'{a.out} 已存在，不覆寫（測試只開封一次）')
        return 2
    try:
        rec = json.load(open(a.opening))
    except Exception as e:                               # noqa: BLE001
        rec = None
        print(f'開封紀錄讀取失敗：{type(e).__name__}: {e}')
    why = check_record(rec, a.runs_root, a.out)
    if why:
        print('開封紀錄與本次參數不一致 ⇒ 拒絕：', why)
        return 2
    res = {'schema': 'xh3_test_eval/2+entry_v2', 'ckpt': a.ckpt, 'opening': os.path.relpath(os.path.abspath(a.opening), WS),
           'runs_root': rec['runs_root'], 'complete': False}
    try:
        import hashlib
        import numpy as np
        import torch
        import yaml
        import xh_train as TR
        dev = torch.device('cuda')
        model = TR.build_model(pretrained=False)
        model.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
        model.to(dev).eval()
        res.update({'ckpt_sha256': hashlib.sha256(open(a.ckpt, 'rb').read()).hexdigest(), 'score_t': TR.SCORE_T, 'bin_t': TR.BIN_T})
        res['env'] = env_fingerprint_v2(torch)
        res['env_vs_dev_ref'] = TE.env_compare(res['env'], DEV_REF_ENV)

        def infer(img):
            with torch.no_grad():
                o = model([torch.from_numpy(img.copy()).permute(2, 0, 1).float().to(dev) / 255.0])[0]
            keep = o['scores'] >= TR.SCORE_T
            pm = (o['masks'][keep][0, 0] >= TR.BIN_T).cpu().numpy() if keep.any() else np.zeros(img.shape[:2], bool)
            return bool(keep.any()), pm
        reg = yaml.safe_load(open(os.path.join(HERE, 'results', 'vision', 'XH1_registration.yaml')))
        assets = {x['id']: x for x in reg['assets']}
        TE.run(infer, assets, TE.TEST, os.path.abspath(a.runs_root), os.path.join(WS, 'src/my_omnibot_description/config/xh'), res=res)
        res['labels_note'] = {'R_x25_150': '名目開口篩選內；直徑未見、長度外插', 'R_x40_300': '名目開口篩選外；直徑與長度皆外插',
                              'N_k40': '干擾物（未見尺寸）'}
    except Exception as e:                               # noqa: BLE001  共享容器：已累積的 rows／fates 一併保存
        res['complete'] = False
        res['fatal'] = f'{type(e).__name__}: {e}'
        res['traceback'] = traceback.format_exc()
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({'complete': res.get('complete'), 'closure': res.get('closure'), 'n_issues': len(res.get('evidence_issues', [])),
                      'fatal': res.get('fatal')}, ensure_ascii=False, default=str))
    return 0 if res.get('complete') else 1


if __name__ == '__main__':
    sys.exit(main())
