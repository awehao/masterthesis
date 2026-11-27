#!/usr/bin/env python3
"""Build one editable slide using the existing scene figure and native shapes.

This is an overview illustration, not a measured trajectory or result figure.
Run from any directory; all source images remain unchanged.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "第八次進度報告素材/第八次進度報告簡報"
IMAGE = ROOT / "預口試論文/overleaf/figsrc/ch02/simulation_environment_map.png"
TEMPLATE = ROOT / "第七次進度報告素材/第七次進度報告簡報/第七次進度報告_工作稿_v1.pptx"
NAME = "本次進度01_住宅任務整合_圖文版"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
FONT = "Noto Sans CJK TC"
NAVY, BLUE, DARK, MUTED = "17324D", "236D91", "263545", "566575"


def emu(v):
    return str(round(v * 914400))


def xfrm(x, y, w, h):
    return f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'


def group():
    return ('<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>')


parts, next_id, drawing = [], 2, []


def box(x, y, w, h, fill, line=None, geometry="rect"):
    global next_id
    i = next_id
    next_id += 1
    drawing.append(('box', x, y, w, h, fill, line, geometry))
    stroke = f'<a:solidFill><a:srgbClr val="{line}"/></a:solidFill>' if line else '<a:noFill/>'
    parts.append(f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="Shape {i}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
                 f'<p:spPr>{xfrm(x,y,w,h)}<a:prstGeom prst="{geometry}"><a:avLst/></a:prstGeom>'
                 f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln w="9525">{stroke}</a:ln></p:spPr></p:sp>')


def text(x, y, w, h, lines, size=16, color=DARK, bold=False, align="l"):
    global next_id
    i = next_id
    next_id += 1
    drawing.append(('text', x, y, w, h, lines, size, color, bold, align))
    ps = []
    for line in lines.split("\n"):
        ps.append(f'<a:p><a:pPr algn="{align}"><a:lnSpc><a:spcPct val="120000"/></a:lnSpc></a:pPr>'
                  f'<a:r><a:rPr lang="zh-TW" sz="{round(size*100)}" b="{int(bold)}">'
                  f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
                  f'<a:latin typeface="{FONT}"/><a:ea typeface="{FONT}"/><a:cs typeface="{FONT}"/>'
                  f'</a:rPr><a:t>{escape(line)}</a:t></a:r><a:endParaRPr lang="zh-TW"/></a:p>')
    parts.append(f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="Text {i}"/>'
                 f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>{xfrm(x,y,w,h)}'
                 '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
                 '<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t"/>'
                 f'<a:lstStyle/>{"".join(ps)}</p:txBody></p:sp>')


def picture(x, y, w, h):
    global next_id
    i = next_id
    next_id += 1
    drawing.append(('picture', x, y, w, h))
    parts.append(f'<p:pic><p:nvPicPr><p:cNvPr id="{i}" name="住宅配置示意" '
                 'descr="本專案住宅配置示意；不含實測路徑或逐趟行人配置。"/>'
                 '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
                 '<p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
                 f'<p:spPr>{xfrm(x,y,w,h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')


def rels(items):
    return f'<Relationships xmlns="{REL}">' + ''.join(
        f'<Relationship Id="{i}" Type="{R}/{kind}" Target="{target}"/>' for i,kind,target in items
    ) + '</Relationships>'


def render_preview():
    """Render the same content and geometry as standalone drawing artifacts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    from matplotlib import image as mpimg
    from matplotlib import font_manager
    from fontTools.ttLib import TTFont
    import tempfile
    # Matplotlib indexes only the first face of a TTC. Extract TC face 3.
    font_dir=Path(tempfile.gettempdir())/'m8_intro_fonts'
    font_dir.mkdir(exist_ok=True)
    for style in ['Regular','Bold']:
        dst=font_dir/f'NotoSansCJKtc-{style}.otf'
        if not dst.exists():
            ft=TTFont(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{style}.ttc',fontNumber=3)
            ft.save(dst)
            ft.close()
        font_manager.fontManager.addfont(str(dst))
    plt.rcParams.update({'font.family': FONT, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(40/3, 7.5), dpi=144, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 40/3), ylim=(7.5, 0))
    ax.axis('off')
    text_bounds = []
    for item in drawing:
        kind, x, y, w, h, *args = item
        if kind == 'box':
            fill, line, geometry = args
            if geometry == 'chevron':
                patch = Polygon([(x,y),(x+w*.55,y),(x+w,y+h/2),(x+w*.55,y+h),(x,y+h),(x+w*.45,y+h/2)],
                                closed=True, facecolor='#'+fill, edgecolor='none', zorder=2)
            else:
                patch = Rectangle((x,y),w,h,facecolor='#'+fill,
                                  edgecolor='#'+line if line else 'none',linewidth=.65,zorder=0)
            ax.add_patch(patch)
        elif kind == 'picture':
            ax.imshow(mpimg.imread(IMAGE), extent=(x,x+w,y+h,y), aspect='auto', zorder=1)
        elif kind == 'text':
            lines, size, color, bold, align = args
            tx = x+w/2 if align == 'ctr' else x
            t = ax.text(tx,y,lines,fontsize=size,color='#'+color,
                        weight='bold' if bold else 'normal',va='top',
                        ha='center' if align=='ctr' else 'left',linespacing=1.2,zorder=3)
            text_bounds.append((t,x,y,w,h))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for t,x,y,w,h in text_bounds:
        bounds=t.get_window_extent(renderer)
        if bounds.width>144*(w+.05) or bounds.height>144*(h+.05):
            raise RuntimeError(f'Text overflow: {t.get_text()}')
    for suffix in ['png','pdf','svg']:
        fig.savefig(OUT/f'{NAME}.{suffix}',dpi=144,facecolor='white')
    plt.close(fig)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    box(0, 0, 13.333333, 7.5, "FFFFFF")
    box(.43, .36, .07, .47, BLUE)
    text(.65, .33, 12.1, .55, "本次進度（一）視覺引導的住宅任務整合", 28, NAVY, True)
    text(.66, 1.00, 12.0, .32, "將單一物件抓放延伸至住宅場景，整合導航、操作與返回流程", 15.5, MUTED)
    box(.45, 1.49, 7.38, 4.40, "F8FAFC", "E0E6EC")
    # Preserve the complete original figure and its aspect ratio.
    from PIL import Image
    with Image.open(IMAGE) as im:
        iw, ih = im.size
    h = 4.07
    w = h * iw / ih
    picture(.45 + (7.38-w)/2, 1.60, w, h)
    text(.60, 5.70, 7.07, .21, "住宅配置示意｜門、茶几與抽屜串接多階段操作", 10.5, MUTED, False, "ctr")

    cards = [
        ("01", "視覺估測與操作選位", "影像與深度 → 操作目標\n可達性與淨空 → 底盤站位"),
        ("02", "動態導航與全身協同", "障礙預測 → 調速與側移參考\n聯合求解底盤與手臂命令"),
        ("03", "機構操作與流程銜接", "門沿鉸鏈旋轉；抽屜沿滑軌移動\n承接狀態、設定點與在途命令"),
    ]
    for j,(number,title,body) in enumerate(cards):
        y = 1.51 + j*1.47
        box(8.10, y, 4.76, 1.33, "F3F7FA")
        text(8.24, y+.13, .42, .33, number, 17, BLUE, True)
        text(8.77, y+.13, 3.92, .33, title, 17.5, NAVY, True)
        text(8.77, y+.58, 3.92, .65, body, 13.0, DARK)

    text(.47, 6.08, 1.4, .30, "完整任務流程", 14.5, NAVY, True)
    labels = ["開門通行", "茶几取物入盒", "抽屜取筆入盒", "關閉抽屜", "回程關門", "返回起點"]
    start, gap, width = .47, .25, 1.85
    for j,label in enumerate(labels):
        x = start+j*(width+gap)
        box(x, 6.48, width, .48, NAVY if j==0 else "EAF1F5")
        text(x+.03, 6.575, width-.06, .30, label, 13.7, "FFFFFF" if j==0 else NAVY, j==0, "ctr")
        if j<5:
            box(x+width+.073, 6.65, .105, .115, BLUE, geometry="chevron")
    text(.48, 7.15, 12.38, .20,
         "任務先驗：地圖、家具區域與物件類型　｜　精確操作目標：當次視覺量測與幾何估測", 10.5, MUTED)

    xml = f'<p:sld xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree>{group()}{"".join(parts)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
    master = f'<p:sldMaster xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree>{group()}</p:spTree></p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
    layout = f'<p:sldLayout xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>{group()}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    pres = f'<p:presentation xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
    overrides = {
        'ppt/presentation.xml':'application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml',
        'ppt/slides/slide1.xml':'application/vnd.openxmlformats-officedocument.presentationml.slide+xml',
        'ppt/slideLayouts/slideLayout1.xml':'application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml',
        'ppt/slideMasters/slideMaster1.xml':'application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml',
        'ppt/theme/theme1.xml':'application/vnd.openxmlformats-officedocument.theme+xml',
        'docProps/core.xml':'application/vnd.openxmlformats-package.core-properties+xml',
        'docProps/app.xml':'application/vnd.openxmlformats-officedocument.extended-properties+xml',
    }
    ct = '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>' + ''.join(f'<Override PartName="/{path}" ContentType="{content}"/>' for path,content in overrides.items()) + '</Types>'
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    core = f'<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>本次進度第一頁 住宅任務整合</dc:title><dc:description>住宅配置為示意圖，不含實測軌跡。圖片來源：預口試論文/overleaf/figsrc/ch02/simulation_environment_map.png；方法依第三章。全部文字與流程框均可編輯。</dc:description><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created></cp:coreProperties>'
    content = {
        '[Content_Types].xml':ct,
        '_rels/.rels':f'<Relationships xmlns="{REL}"><Relationship Id="rId1" Type="{R}/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="{R}/extended-properties" Target="docProps/app.xml"/></Relationships>',
        'docProps/core.xml':core,
        'docProps/app.xml':'<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Codex</Application><Slides>1</Slides></Properties>',
        'ppt/presentation.xml':pres,
        'ppt/_rels/presentation.xml.rels':rels([('rId1','slideMaster','slideMasters/slideMaster1.xml'),('rId2','slide','slides/slide1.xml')]),
        'ppt/slides/slide1.xml':xml,
        'ppt/slides/_rels/slide1.xml.rels':rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','image','../media/scene.png')]),
        'ppt/slideLayouts/slideLayout1.xml':layout,
        'ppt/slideLayouts/_rels/slideLayout1.xml.rels':rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')]),
        'ppt/slideMasters/slideMaster1.xml':master,
        'ppt/slideMasters/_rels/slideMaster1.xml.rels':rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','theme','../theme/theme1.xml')]),
    }
    for value in content.values():
        ET.fromstring(value)
    with ZipFile(TEMPLATE) as src:
        theme = src.read('ppt/theme/theme1.xml')
    with ZipFile(OUT/f'{NAME}.pptx','w',ZIP_DEFLATED) as z:
        for path,value in content.items():
            z.writestr(path,'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'+value)
        z.writestr('ppt/theme/theme1.xml',theme)
        z.write(IMAGE,'ppt/media/scene.png')
    print(OUT/f'{NAME}.pptx')
    render_preview()


if __name__ == '__main__':
    build()
