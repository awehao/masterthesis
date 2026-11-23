"""把渲染幀加上字幕並編成影片。

只做疊字與編碼：**不碰任何量測值**，動態讀數全部來自渲染時寫下的
replay_check.csv（其值又直接取自原趟次的 log）。
"""
import argparse, csv, json, os, glob
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument('--frames', required=True)
ap.add_argument('--spec', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--fps', type=float, default=30.0)
ap.add_argument('--title-s', type=float, default=4.0)
a = ap.parse_args()

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
if not os.path.exists(FONT):
    FONT = FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc'

S = json.load(open(a.spec))
META = json.load(open(os.path.join(a.frames, 'render_meta.json')))
ROWS = list(csv.DictReader(open(os.path.join(a.frames, 'replay_check.csv'))))
PNG = sorted(glob.glob(os.path.join(a.frames, 'f*.png')))
assert len(PNG) == len(ROWS), f'幀數 {len(PNG)} 與核對表 {len(ROWS)} 不符'

PHASE = {'idle': '待命（尚未收到命令）', 'reach': '移動到起始姿態', 'approach': '接近把手',
         'engage': '建立理想固定連接', 'postengage': '連接後保持',
         'pull': '拉開', 'hold': '保持', 'release': '解除連接／張開手指',
         'retreat': '退出', 'settle': '靜置', '': ''}

W, H = Image.open(PNG[0]).size
f_t = ImageFont.truetype(FONTB, int(H * 0.052))
f_s = ImageFont.truetype(FONT, int(H * 0.032))
f_i = ImageFont.truetype(FONT, int(H * 0.0265))
f_d = ImageFont.truetype(FONTB, int(H * 0.036))
f_f = ImageFont.truetype(FONT, int(H * 0.023))

OUTD = os.path.join(a.frames, '_annotated')
os.makedirs(OUTD, exist_ok=True)


def band(d, xy, wh, alpha=170):
    o = Image.new('RGBA', wh, (12, 14, 18, alpha))
    return o, xy


def paste(img, o, xy):
    img.alpha_composite(o, xy)


# ---------- 標題卡 ----------
n_title = int(round(a.title_s * a.fps))
tc = Image.new('RGBA', (W, H), (14, 16, 20, 255))
d = ImageDraw.Draw(tc)
y = int(H * 0.15)
d.text((int(W * 0.08), y), S['title'], font=f_t, fill=(245, 246, 248))
y += int(H * 0.075)
d.text((int(W * 0.08), y), S['subtitle'], font=f_s, fill=(236, 160, 48))
y += int(H * 0.085)
for ln in S['info']:
    d.text((int(W * 0.08), y), ln, font=f_i, fill=(198, 204, 212))
    y += int(H * 0.042)
d.text((int(W * 0.08), int(H * 0.92)), S['footer'], font=f_f, fill=(150, 156, 166))
for k in range(n_title):
    tc.convert('RGB').save(os.path.join(OUTD, f'a{k:05d}.png'))

# ---------- 逐幀 ----------
IS_DRAWER = META['kind'] == 'drawer'
EV = META.get('events_sim_t', {})
hdr = S.get('running_header', S['subtitle'])
for i, (p, r) in enumerate(zip(PNG, ROWS)):
    img = Image.open(p).convert('RGBA')
    d = ImageDraw.Draw(img)
    # 上方：固定標示
    o, xy = band(d, (0, 0), (W, int(H * 0.105)), 150)
    paste(img, o, xy)
    d.text((int(W * 0.022), int(H * 0.016)), S['title'], font=f_d,
           fill=(245, 246, 248))
    d.text((int(W * 0.022), int(H * 0.064)), hdr, font=f_f, fill=(236, 160, 48))
    # 下方：動態讀數
    o, xy = band(d, (0, int(H * 0.86)), (W, int(H * 0.14)), 150)
    paste(img, o, xy)
    t = float(r['log_t'])
    ph = PHASE.get(r['phase'], r['phase'])
    left = f"模擬時間 {t:6.2f} s"
    if IS_DRAWER:
        # 只是顯示取兩位小數：−0.001 mm 會印成「-0.00」，讀起來像有號的零，
        # 所以小於半個顯示位的值直接顯示 0.00（四捨五入，不是改數據）。
        ov = float(r['opening_mm'])
        if abs(ov) < 0.005:
            ov = 0.0
        left += f"    開度 {ov:6.2f} mm"
    d.text((int(W * 0.022), int(H * 0.875)), left, font=f_d, fill=(245, 246, 248))
    if ph:
        d.text((int(W * 0.022), int(H * 0.932)), f"階段：{ph}", font=f_f,
               fill=(198, 204, 212))
    fs = S.get('footer_short', S['footer'])
    d.text((W - int(d.textlength(fs, font=f_f)) - int(W * 0.022),
            int(H * 0.932)), fs, font=f_f, fill=(150, 156, 166))
    # 事件提示（前後 0.6 s）
    for name, lbl in (('engage', '建立理想固定連接'), ('release', '解除固定連接')):
        et = EV.get(name)
        if et is not None and abs(t - float(et)) <= 0.6:
            tw = d.textlength(lbl, font=f_d)
            bx = int((W - tw) / 2) - 18
            o2 = Image.new('RGBA', (int(tw) + 36, int(H * 0.075)), (200, 60, 40, 200))
            img.alpha_composite(o2, (bx, int(H * 0.74)))
            d.text((bx + 18, int(H * 0.752)), lbl, font=f_d, fill=(255, 255, 255))
    img.convert('RGB').save(os.path.join(OUTD, f'a{n_title + i:05d}.png'))

print(f'[annotate] 疊字完成 {n_title + len(PNG)} 幀 -> {OUTD}')
rc = os.system(
    f'ffmpeg -y -loglevel error -framerate {a.fps} -i "{OUTD}/a%05d.png" '
    f'-c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "{a.out}"')
if rc != 0:
    raise SystemExit(f'ffmpeg 失敗 rc={rc}')
sz = os.path.getsize(a.out) / 1e6
print(f'[annotate] 影片 {a.out}  {sz:.1f} MB  {n_title + len(PNG)} 幀 @ {a.fps} fps')
