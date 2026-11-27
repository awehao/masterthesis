#!/usr/bin/env python3
"""Create method slides 02–05, reusing pre-oral figures without changing them.

Outputs editable PPTX, per-page PNG/SVG/PDF, and a combined deck with page 01.
Renderings share the same text, image placement, and geometry as the PPTX.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
import hashlib
import json
import posixpath
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import build_m8_intro_slide as base

ROOT, OUT = base.ROOT, base.OUT
P, A, R, REL = base.P, base.A, base.R, base.REL
FONT = base.FONT
NAVY, BLUE, DARK, MUTED = base.NAVY, base.BLUE, base.DARK, base.MUTED
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'
FIG = ROOT/'預口試論文/overleaf/figsrc'
INTRO = OUT/'本次進度01_住宅任務整合_圖文版.pptx'
METHODS = '第八次進度報告_本次進度02至05_圖文版'
COMBINED = '第八次進度報告_本次進度01至05_圖文版'
ET.register_namespace('p', P)
ET.register_namespace('a', A)
ET.register_namespace('r', R)


class Slide:
    def __init__(self, number, title, subtitle, stem, footer):
        self.number, self.title, self.stem = number, title, stem
        self.parts, self.drawing, self.images = [], [], []
        self.next_id = 2
        self.box(0,0,40/3,7.5,'FFFFFF')
        self.box(.43,.36,.07,.47,BLUE)
        self.text(.65,.33,12.1,.55,title,28,NAVY,True)
        self.text(.66,1.00,12.0,.32,subtitle,15.5,MUTED)
        self.text(.48,7.14,11.94,.24,footer,9.3,MUTED)
        self.text(12.52,7.12,.34,.25,f'{number:02}',10.5,BLUE,True,'ctr')

    def ident(self):
        i=self.next_id
        self.next_id+=1
        return i

    def box(self,x,y,w,h,fill,line=None,geometry='rect'):
        i=self.ident()
        stroke=f'<a:solidFill><a:srgbClr val="{line}"/></a:solidFill>' if line else '<a:noFill/>'
        self.parts.append(f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="Shape {i}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
                          f'<p:spPr>{base.xfrm(x,y,w,h)}<a:prstGeom prst="{geometry}"><a:avLst/></a:prstGeom>'
                          f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln w="9525">{stroke}</a:ln></p:spPr></p:sp>')
        self.drawing.append(('box',x,y,w,h,fill,line,geometry))

    def text(self,x,y,w,h,lines,size=16,color=DARK,bold=False,align='l'):
        i=self.ident()
        ps=[]
        for line in lines.split('\n'):
            ps.append(f'<a:p><a:pPr algn="{align}"><a:lnSpc><a:spcPct val="120000"/></a:lnSpc></a:pPr>'
                      f'<a:r><a:rPr lang="zh-TW" sz="{round(size*100)}" b="{int(bold)}">'
                      f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="{FONT}"/>'
                      f'<a:ea typeface="{FONT}"/><a:cs typeface="{FONT}"/></a:rPr><a:t>{escape(line)}</a:t>'
                      '</a:r><a:endParaRPr lang="zh-TW"/></a:p>')
        self.parts.append(f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="Text {i}"/><p:cNvSpPr txBox="1"/>'
                          f'<p:nvPr/></p:nvSpPr><p:spPr>{base.xfrm(x,y,w,h)}'
                          '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln>'
                          '</p:spPr><p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" '
                          f'anchor="t"/><a:lstStyle/>{"".join(ps)}</p:txBody></p:sp>')
        self.drawing.append(('text',x,y,w,h,lines,size,color,bold,align))

    def image(self,path,x,y,w,h,description):
        """Fit a complete original figure into a box; no cropping or stretching."""
        from PIL import Image
        path=Path(path)
        with Image.open(path) as im:
            iw,ih=im.size
        ratio=min(w/iw,h/ih)
        pw,ph=ratio*iw,ratio*ih
        px,py=x+(w-pw)/2,y+(h-ph)/2
        i=self.ident()
        rid=f'rId{len(self.images)+2}'
        self.images.append({'path':path,'rid':rid,'description':description})
        self.parts.append(f'<p:pic><p:nvPicPr><p:cNvPr id="{i}" name="{escape(path.stem)}" '
                          f'descr="{escape(description)}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/>'
                          '</p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill>'
                          f'<a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
                          f'<p:spPr>{base.xfrm(px,py,pw,ph)}<a:prstGeom prst="rect"><a:avLst/>'
                          '</a:prstGeom></p:spPr></p:pic>')
        self.drawing.append(('picture',px,py,pw,ph,path))

    def card(self,x,y,w,h,number,title,body,body_size=13.4):
        self.box(x,y,w,h,'F3F7FA')
        self.text(x+.14,y+.13,.42,.32,number,16,BLUE,True)
        self.text(x+.66,y+.13,w-.82,.34,title,16.2,NAVY,True)
        self.text(x+.66,y+.57,w-.82,h-.66,body,body_size)

    def banner(self,body,y=6.49):
        self.box(.47,y,12.39,.46,'EAF1F5')
        self.text(.68,y+.105,11.97,.30,body,14,NAVY,True,'ctr')

    def xml(self):
        return f'<p:sld xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree>{base.group()}{"".join(self.parts)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'


def make_slides():
    s2=Slide(2,'本次進度（二）視覺定位與操作選位',
             '由目標分割與深度建立操作幾何，再選擇底盤站位與抓取構型',
             '本次進度02_視覺定位與操作選位_圖文版',
             '圖片：預口試第二、三章｜模擬分割示例與選位示意；物件尺寸、鉸鏈與滑軌等先驗仍保留')
    s2.text(.49,1.49,5.6,.30,'六類目標的 RGB 與預測遮罩',16,NAVY,True)
    s2.image(FIG/'ch02/mask_rcnn_class_examples.png',.49,1.97,5.56,3.55,
             '六類模擬測試影像與模型預測遮罩；非同一趟線上連續觀測。')
    s2.text(6.36,1.49,6.49,.30,'由目標位置選擇可行操作構型',16,NAVY,True)
    s2.image(FIG/'ch03/visual_operation_reference.png',6.36,1.97,6.49,2.69,
             '操作目標導引之底盤選位示意；(a)選位前，(b)選位後接近目標。')
    s2.text(6.48,4.88,6.22,.32,'操作目標：遮罩＋有效深度＋觀測時刻位姿',14.4,DARK)
    s2.text(6.48,5.34,6.22,.32,'候選檢查：導航可達、淨空及關節構型',14.4,DARK)
    labels=['RGB 實例分割','深度與幾何估測','觀測時間對齊','選位與接近參考']
    for j,label in enumerate(labels):
        x=.48+j*3.16
        s2.box(x,6.30,2.88,.53,NAVY if j==3 else 'EAF1F5')
        s2.text(x+.04,6.435,2.80,.30,label,14,'FFFFFF' if j==3 else NAVY,j==3,'ctr')
        if j<3:s2.box(x+2.98,6.49,.10,.13,BLUE,geometry='chevron')

    s3=Slide(3,'本次進度（三）底盤與手臂的移動中接近',
             '依剩餘距離安排展臂與後段併線，讓末端接近進度配合底盤前進',
             '本次進度03_移動中接近與全身協同_圖文版',
             '圖片：預口試第三章｜曲線為設計規則示例；圖中速度與門檻不是實測運動或性能結果')
    s3.text(.50,1.53,7.0,.30,'剩餘距離同時影響速度上限與展臂進度',16,NAVY,True)
    s3.image(FIG/'ch03/mobile_approach_distance_profile.png',.49,2.02,7.08,4.19,
             '平移逐軸速度上限、手臂距離展開進度及後段併線門檻的設計示例；非實測曲線。')
    s3.card(7.91,1.57,4.95,1.34,'01','先形成接近構型',
            '起始關節向選位構型平滑展開\n產生對應的全身末端參考')
    s3.card(7.91,3.07,4.95,1.34,'02','後段銜接抓取線',
            '展開參考逐步併入世界抓取線\n底盤與末端共用同一時間基準')
    s3.card(7.91,4.57,4.95,1.34,'03','依到達狀態銜接',
            '末端與底盤接近條件成立後閉爪\n超出可達窗口時停止並回報')
    s3.banner('接近尚未完成時，降低底盤速度或延後操作進度')

    s4=Slide(4,'本次進度（四）動態障礙預測與導航調整',
             '在共同未來時刻比較候選運動，選擇速度與偏移後更新導航參考',
             '本次進度04_動態障礙預測與導航調整_圖文版',
             '圖片：預口試第三章｜抓取接近段限制為不加速、不新增偏移；候選仍須經後續防護核對')
    s4.image(FIG/'ch03/dynamic_obstacle_prediction.png',.51,1.58,12.32,3.26,
             '動態障礙候選運動示意；(b)(c)(d)為同一未來時刻，虛線圓為目前底盤位置。')
    captions=['(a) 目前觀測','(b) 維持原速度','(c) 降低速度','(d) 側向避讓']
    for j,caption in enumerate(captions):
        s4.text(.55+j*3.08,4.88,3.00,.30,caption,14,NAVY,True,'ctr')
    s4.card(.48,5.48,4.00,1.24,'01','比較預測間距',
            '由歷史運動外推障礙位置\n在共同時間軸比較候選',12.6)
    s4.card(4.68,5.48,4.00,1.24,'02','兼顧前進進度',
            '評估調速、側移與切換成本\n更新速度與偏移參考',12.6)
    s4.card(8.88,5.48,3.98,1.24,'03','銜接執行防護',
            '全身 QP／CBF、原始掃描\n與執行端核對持續作用',12.6)

    s5=Slide(5,'本次進度（五）門與抽屜的機構操作',
             '依物件的旋轉或平移幾何生成末端參考，底盤與手臂共同配合',
             '本次進度05_門與抽屜機構操作_圖文版',
             '圖片：預口試第三章構型示意｜抓持後採固定相對位姿前提；接觸力與滑移未納入最佳化')
    s5.text(.52,1.57,5.98,.30,'門：把手繞鉸鏈作圓弧運動',17,NAVY,True)
    s5.text(6.83,1.57,5.98,.30,'抽屜：把手沿滑軌方向平移',17,NAVY,True)
    s5.image(FIG/'ch03/door_operation_sequence.png',.49,2.06,6.04,3.06,
             '門扇操作起始與開啟後之全身構型示意；非時序實測影格。')
    s5.image(FIG/'ch03/drawer_operation_sequence.png',6.81,2.06,6.04,3.06,
             '抽屜初始與拉出後之全身構型示意；非時序實測影格。')
    s5.box(.49,5.30,6.04,.91,'F3F7FA')
    s5.text(.65,5.47,5.73,.62,
            '門角參考 → 把手圓弧 → 末端位姿\n依回報門角調整進度，底盤配合把手運動',13.3)
    s5.box(6.81,5.30,6.04,.91,'F3F7FA')
    s5.text(6.97,5.47,5.73,.62,
            '位移參考 → 沿滑軌平移 → 末端位姿\n由 TCP 軸向位移估計進度，同步調整全身',13.3)
    s5.banner('共用全身控制器，依任務階段更新目標、追蹤權重與啟用限制')
    return [s2,s3,s4,s5]


def write_deck(slides,filename,include_intro=False):
    with ZipFile(INTRO) as z:
        content={name:z.read(name) for name in z.namelist()}
    pres=ET.fromstring(content['ppt/presentation.xml'])
    ids=pres.find(f'{{{P}}}sldIdLst')
    ids.clear()
    rel=[('rId1','slideMaster','slideMasters/slideMaster1.xml')]
    types=ET.fromstring(content['[Content_Types].xml'])
    for child in list(types):
        if child.get('PartName','').startswith('/ppt/slides/'):
            types.remove(child)
    offset=1 if include_intro else 0
    if include_intro:
        ET.SubElement(ids,f'{{{P}}}sldId',{'id':'256',f'{{{R}}}id':'rId2'})
        rel.append(('rId2','slide','slides/slide1.xml'))
        ET.SubElement(types,f'{{{CT}}}Override',{'PartName':'/ppt/slides/slide1.xml',
            'ContentType':'application/vnd.openxmlformats-officedocument.presentationml.slide+xml'})
    for j,slide in enumerate(slides,1+offset):
        ET.SubElement(ids,f'{{{P}}}sldId',{'id':str(255+j),f'{{{R}}}id':f'rId{j+1}'})
        rel.append((f'rId{j+1}','slide',f'slides/slide{j}.xml'))
        content[f'ppt/slides/slide{j}.xml']=slide.xml().encode()
        rs=[('rId1','slideLayout','../slideLayouts/slideLayout1.xml')]
        for k,img in enumerate(slide.images,1):
            part=f'm8_{slide.number}_{k}.png'
            content[f'ppt/media/{part}']=img['path'].read_bytes()
            rs.append((img['rid'],'image',f'../media/{part}'))
        content[f'ppt/slides/_rels/slide{j}.xml.rels']=base.rels(rs).encode()
        ET.SubElement(types,f'{{{CT}}}Override',{'PartName':f'/ppt/slides/slide{j}.xml',
            'ContentType':'application/vnd.openxmlformats-officedocument.presentationml.slide+xml'})
    content['ppt/presentation.xml']=ET.tostring(pres,encoding='utf-8',xml_declaration=True)
    content['ppt/_rels/presentation.xml.rels']=base.rels(rel).encode()
    content['[Content_Types].xml']=ET.tostring(types,encoding='utf-8',xml_declaration=True)
    app=ET.fromstring(content['docProps/app.xml'])
    app.find('{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Slides').text=str(len(slides)+offset)
    content['docProps/app.xml']=ET.tostring(app,encoding='utf-8',xml_declaration=True)
    core=ET.fromstring(content['docProps/core.xml'])
    core.find('{http://purl.org/dc/elements/1.1/}title').text=filename
    core.find('{http://purl.org/dc/elements/1.1/}description').text='第八次進度報告方法頁。沿用預口試圖片；圖中設計曲線與構型示意不作為實測性能證據。文字與流程圖形可編輯。來源見同目錄圖片來源 JSON。'
    content['docProps/core.xml']=ET.tostring(core,encoding='utf-8',xml_declaration=True)
    with ZipFile(OUT/f'{filename}.pptx','w',ZIP_DEFLATED) as z:
        for path,data in content.items():
            if path.endswith(('.xml','.rels')):ET.fromstring(data)
            z.writestr(path,data)


def setup_rendering():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from fontTools.ttLib import TTFont
    font_dir=Path(tempfile.gettempdir())/'m8_intro_fonts'
    font_dir.mkdir(exist_ok=True)
    for style in ['Regular','Bold']:
        dst=font_dir/f'NotoSansCJKtc-{style}.otf'
        if not dst.exists():
            ft=TTFont(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{style}.ttc',fontNumber=3)
            ft.save(dst)
            ft.close()
        font_manager.fontManager.addfont(str(dst))
    plt.rcParams.update({'font.family':FONT,'svg.fonttype':'none','pdf.fonttype':42})
    return plt


def render_slide(slide,plt,pdf):
    from matplotlib.patches import Rectangle, Polygon
    from matplotlib import image as mpimg
    fig=plt.figure(figsize=(40/3,7.5),dpi=144,facecolor='white')
    ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,40/3),ylim=(7.5,0))
    ax.axis('off')
    checks=[]
    for item in slide.drawing:
        kind,x,y,w,h,*args=item
        assert x>=0 and y>=0 and x+w<=40/3+.01 and y+h<=7.51,item
        if kind=='box':
            fill,line,geometry=args
            if geometry=='chevron':
                shape=Polygon([(x,y),(x+w*.55,y),(x+w,y+h/2),(x+w*.55,y+h),(x,y+h),(x+w*.45,y+h/2)],
                              closed=True,facecolor='#'+fill,edgecolor='none',zorder=1)
            else:
                shape=Rectangle((x,y),w,h,facecolor='#'+fill,edgecolor='#'+line if line else 'none',linewidth=.65,zorder=0)
            ax.add_patch(shape)
        elif kind=='picture':
            ax.imshow(mpimg.imread(args[0]),extent=(x,x+w,y+h,y),aspect='auto',zorder=1)
        elif kind=='text':
            lines,size,color,bold,align=args
            t=ax.text(x+w/2 if align=='ctr' else x,y,lines,fontsize=size,color='#'+color,
                      weight='bold' if bold else 'normal',va='top',ha='center' if align=='ctr' else 'left',
                      linespacing=1.2,zorder=3)
            checks.append((t,x,y,w,h))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for t,x,y,w,h in checks:
        bound=t.get_window_extent(renderer)
        if bound.width>144*(w+.04) or bound.height>144*(h+.04):
            raise RuntimeError(f'Page {slide.number} text overflow: {t.get_text()}')
    for suffix in ['png','pdf','svg']:
        fig.savefig(OUT/f'{slide.stem}.{suffix}',dpi=144,facecolor='white')
    pdf.savefig(fig,facecolor='white')
    plt.close(fig)


def validate_deck(path,expected):
    with ZipFile(path) as z:
        assert z.testzip() is None
        names=set(z.namelist())
        for name in names:
            if name.endswith('.rels'):
                parent='' if name=='_rels/.rels' else name.rsplit('/_rels/',1)[0]
                for rel in ET.fromstring(z.read(name)):
                    if rel.get('TargetMode')!='External':
                        target=posixpath.normpath(posixpath.join(parent,rel.get('Target')))
                        assert target in names,(name,target)
        ids=ET.fromstring(z.read('ppt/presentation.xml')).find(f'{{{P}}}sldIdLst')
        assert len(ids)==expected
        for j in range(1,expected+1):
            s=ET.fromstring(z.read(f'ppt/slides/slide{j}.xml'))
            shape_ids=[x.get('id') for x in s.findall(f'.//{{{P}}}cNvPr')]
            assert len(shape_ids)==len(set(shape_ids))


def main():
    slides=make_slides()
    paths={img['path'] for slide in slides for img in slide.images}
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    plt=setup_rendering()
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(OUT/f'{METHODS}.pdf') as pdf:
        for slide in slides:render_slide(slide,plt,pdf)
    write_deck(slides,METHODS)
    write_deck(slides,COMBINED,include_intro=True)
    validate_deck(OUT/f'{METHODS}.pptx',4)
    validate_deck(OUT/f'{COMBINED}.pptx',5)
    subprocess.run(['pdfunite',str(INTRO.with_suffix('.pdf')),str(OUT/f'{METHODS}.pdf'),str(OUT/f'{COMBINED}.pdf')],check=True)
    fig,axes=plt.subplots(2,2,figsize=(40/3,7.5),dpi=144)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=.02,hspace=.02)
    for ax,slide in zip(axes.flat,slides):
        ax.imshow(plt.imread(OUT/f'{slide.stem}.png'))
        ax.axis('off')
    fig.savefig(OUT/'本次進度02至05_四頁預覽.png',dpi=144,facecolor='#E6EBEF')
    plt.close(fig)
    sources=[]
    for slide in slides:
        sources.append({'page':slide.number,'title':slide.title,'output_stem':slide.stem,
                        'method_source':'預口試論文/第三章_精簡重整版/03_ResearchMethodology.tex',
                        'figures':[{'path':str(img['path'].relative_to(ROOT)),
                                    'sha256':before[str(img['path'])],'scope':img['description']}
                                   for img in slide.images]})
    for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==before[str(p)]
    (OUT/'本次進度02至05_圖片來源.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2))
    with ZipFile(OUT/'本次進度02至05_逐頁圖片.zip','w',ZIP_DEFLATED) as z:
        for slide in slides:z.write(OUT/f'{slide.stem}.png',f'{slide.stem}.png')
        z.write(OUT/'本次進度02至05_圖片來源.json','本次進度02至05_圖片來源.json')
    print(json.dumps({'new_slides':4,'combined_slides':5,'source_images':len(paths),
                      'source_images_unchanged':True,'pptx':str(OUT/f'{COMBINED}.pptx')},ensure_ascii=False))


if __name__=='__main__':
    main()
