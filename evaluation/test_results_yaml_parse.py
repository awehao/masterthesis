"""evaluation/results/ 下的每一份 YAML 都必須能解析。

為什麼需要：規格檔是紀錄，不是散文。解析不了的紀錄沒有人會發現 ——
它在被引用時才出錯，而那通常是幾週之後。最常見的成因是以 `*` 開頭的值
（`**強調**` 寫在值的開頭時，YAML 把它當成 alias）。
"""
import glob, sys, yaml

OK, BAD = [], []
for p in sorted(glob.glob('evaluation/results/*.yaml')):
    try:
        d = yaml.safe_load(open(p))
        if d is None:
            BAD.append((p, '解析結果為空')); continue
        if not isinstance(d, dict):
            BAD.append((p, f'頂層不是映射而是 {type(d).__name__}')); continue
        OK.append((p, len(d)))
    except Exception as e:
        BAD.append((p, str(e).splitlines()[-2].strip()))
for p, n in OK:
    print(f'ok    {p.split("/")[-1]:48s} 頂層鍵 {n}')
for p, why in BAD:
    print(f'FAIL  {p.split("/")[-1]:48s} {why}')
print(f'\n{len(OK)}/{len(OK)+len(BAD)} 份可解析')
sys.exit(0 if not BAD else 1)
