#!/usr/bin/env python3
"""Standalone annotated diagrams, without slide headings, prose or results."""
from pathlib import Path
from html import escape
import math
import zipfile
from PIL import Image, ImageDraw, ImageFont
import build_fifth_progress_visuals as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '第五次進度報告素材' / '純示意圖'
BLUE, TEAL, ORANGE = base.BLUE, base.TEAL, base.ORANGE
INK, MUTED, NAVY, LINE = base.INK, base.MUTED, base.NAVY, base.LINE


class Diagram(base.Figure):
    def __init__(self, title, w=1600, h=1000):
        self.w,self.h = w,h
        self.s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',f'<title>{escape(title)}</title>','<defs>']
        for color in (BLUE,TEAL,ORANGE,MUTED):
            self.s.append(f'<marker id="a{color[1:]}" markerWidth="14" markerHeight="14" refX="12" refY="7" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L14 7 L0 14 Z" fill="{color}"/></marker>')
        self.s.append('</defs>')

    def label(self,x,y,txt,size=34,color=INK,anchor='start'):
        self.text(x,y,txt,size,color,600,anchor)

    def ellipse(self,x,y,rx,ry,fill,stroke=BLUE,width=3):
        self.s.append(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def path(self,d,color=BLUE,width=4,dashed=False,arrow=False):
        opts=' stroke-dasharray="12 10"' if dashed else ''
        if arrow:opts+=f' marker-end="url(#a{color[1:]})"'
        self.s.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{opts}/>')

    def node(self,x,y,w,h,label,color=BLUE,small=None):
        self.rect(x,y,w,h,'#F2F7FB' if color==BLUE else '#EFF8F5',color,18)
        self.text(x+w/2,y+h/2+(12 if small is None else -10),label,33,color,700,'middle')
        if small:self.text(x+w/2,y+h/2+37,small,26,MUTED,400,'middle')

    def save(self,name):
        svg='\n'.join(self.s+['</svg>'])
        p=OUT/name
        p.with_suffix('.svg').write_text(svg)
        base.W,base.H=self.w,self.h
        base.render_files(svg,p)
        with Image.open(p.with_suffix('.png')) as src:
            white=Image.new('RGBA',src.size,'white')
            white.alpha_composite(src.convert('RGBA'))
            white.convert('RGB').save(OUT/(name+'_白底.png'))
        return p


def robot():
    f=Diagram('底盤與六軸手臂的全身運動學示意')
    # Chassis in schematic oblique projection.
    f.ellipse(652,770,287,92,'#C2D6E8')
    f.rect(365,678,574,94,'#DCE9F4',BLUE,4)
    f.ellipse(652,678,287,92,'#EAF2F9')
    for x,y,angle in [(402,739,-20),(880,739,20),(465,815,-20),(836,815,20)]:
        f.s.append(f'<g transform="translate({x} {y}) rotate({angle})">')
        f.rect(-38,-26,76,52,NAVY,'none',13)
        for dx in [-24,-8,8,24]: f.line([(dx,-20),(dx,20)],'#8BA3BA',4)
        f.s.append('</g>')
    f.ellipse(652,637,70,23,'#BDD7CD',TEAL)
    f.rect(582,587,140,50,'#D5E8E0',TEAL,4)
    joints=[(652,587),(652,490),(522,338),(756,215),(902,320),(1050,320)]
    f.line(joints,'#BFD9CE',57)
    f.line(joints,TEAL,36)
    for i,(x,y) in enumerate(joints):
        f.circle(x,y,23,'white',TEAL,5)
        offsets=[(-87,15),(-87,7),(-73,-15),(-16,-52),(-10,-48),(-12,-48)]
        dx,dy=offsets[i]
        f.label(x+dx,y+dy,'J'+str(i+1),28,TEAL)
    f.rect(1080,285,58,70,NAVY,radius=8)
    f.line([(1138,294),(1198,294),(1198,311)],NAVY,11)
    f.line([(1138,346),(1198,346),(1198,329)],NAVY,11)
    f.circle(1200,320,7,ORANGE,ORANGE,1)
    f.line([(1200,320),(1294,320)],ORANGE,4,True)
    f.line([(1200,320),(1200,226)],TEAL,4,True)
    f.label(1310,330,'zₜ',28,ORANGE)
    f.label(1215,238,'xₜ',28,TEAL)
    f.label(1231,404,'TCP／夾爪',35,ORANGE)
    f.line([(1208,390),(1200,332)],ORANGE,2)
    f.rect(959,261,47,27,'#7990A5',NAVY,5)
    f.circle(974,274,6,'#E3F1FC',NAVY,1)
    f.label(1020,162,'腕部相機',33,MUTED)
    f.line([(1057,180),(1040,228),(984,260)],MUTED,2)
    f.label(210,216,'Lite 6・6 DOF',39,TEAL)
    f.line([(330,238),(392,274),(491,327)],TEAL,2)
    f.label(110,667,'全向底盤',39,BLUE)
    f.label(110,714,'3 DOF',31,BLUE)
    f.line([(294,681),(355,693)],BLUE,2)
    f.line([(1105,813),(1328,813)],BLUE,5,True)
    f.line([(1105,813),(1026,748)],BLUE,5,True)
    f.label(1347,824,'x',31,BLUE)
    f.label(996,746,'y',31,BLUE)
    f.path('M 1141 872 C 1250 906 1375 850 1313 757',BLUE,4,arrow=True)
    f.label(1350,881,'θ',33,BLUE)
    f.label(650,944,'q = [ x, y, θ, q₁, …, q₆ ]ᵀ',37,NAVY,'middle')
    f.text(1540,975,'幾何示意・不依比例',23,MUTED,anchor='end')
    return f.save('01_全身機器人與自由度')


def clearance():
    f=Diagram('連桿表面取樣與障礙距離',1600,900)
    f.rect(1200,110,275,648,'#FBEADC',ORANGE,8)
    f.label(1337,817,'障礙物表面',36,ORANGE,'middle')
    f.line([(1100,100),(1100,758)],ORANGE,3,dashed=True)
    f.label(1113,66,'安全餘裕',31,ORANGE)
    f.line([(1100,91),(1200,91)],ORANGE,3)
    for x in [1100,1200]: f.line([(x,83),(x,99)],ORANGE,3)
    f.s.append('<g transform="translate(195 571) rotate(-26)">')
    f.rect(0,-80,610,160,'#DFEEE8',TEAL,80)
    for x,y in [(35,-65),(98,-80),(163,-80),(228,-80),(293,-80),(358,-80),(423,-80),(488,-80),(553,-64),(600,-32),(610,0),(600,32),(553,64),(488,80),(423,80),(358,80),(293,80),(228,80),(163,80),(98,80),(35,65),(6,30),(0,0),(6,-30)]:
        f.circle(x,y,7,TEAL,TEAL,1)
    f.s.append('</g>')
    f.circle(253,542,29,'white',TEAL,5)
    # Selected surface sample; distance to the planar obstacle is normal.
    px,py=745,304
    f.circle(px,py,12,ORANGE,ORANGE,2)
    f.line([(px+18,py),(1190,py)],ORANGE,5,True)
    f.label(960,267,'dᵢ',42,ORANGE,'middle')
    f.label(869,344,'法向 nᵢ',30,ORANGE)
    f.line([(px,py-20),(px,145)],BLUE,5,True)
    f.label(584,139,'切向速度',32,BLUE)
    f.line([(px+10,py+12),(941,437)],TEAL,5,True)
    f.label(929,487,'表面點速度 Jᵢν',31,TEAL,'middle')
    f.label(154,764,'連桿表面取樣點',36,TEAL)
    f.line([(352,735),(403,663),(441,540)],TEAL,2)
    f.label(713,586,'表面點 pᵢ',34,ORANGE)
    f.line([(793,554),(814,518),(749,321)],ORANGE,2)
    f.text(1540,867,'幾何示意・不依比例',23,MUTED,anchor='end')
    return f.save('02_連桿表面距離與速度')


def self_filter():
    f=Diagram('LiDAR 動態自體濾波示意',1700,900)
    for cx,after in [(390,False),(1300,True)]:
        cy=685
        f.label(cx,85,'濾波後' if after else '濾波前',40,TEAL if after else BLUE,'middle')
        f.line([(cx-270,177),(cx+270,177)],ORANGE,16)
        f.label(cx,139,'外部牆面',32,ORANGE,'middle')
        f.circle(cx,cy,89,'#EAF2F9',BLUE,4)
        # Return geometry is constructed by ray intersections with a cross-section.
        armx,army,r=cx+55,405,79
        a0=math.atan2(army-cy,armx-cx)
        spread=math.asin(r/math.hypot(armx-cx,army-cy))
        if after:
            aa,bb=a0-spread,a0+spread
            ya=177
            xa=cx+(ya-cy)/math.sin(aa)*math.cos(aa)
            xb=cx+(ya-cy)/math.sin(bb)*math.cos(bb)
            f.s.append(f'<path d="M{cx} {cy} L{xa} {ya} L{xb} {ya} Z" fill="#E7EBEF" opacity="0.8"/>')
        for k in range(23):
            a=math.radians(-118+k*2.55)
            dx,dy=math.cos(a),math.sin(a)
            ox,oy=cx-armx,cy-army
            b=2*(ox*dx+oy*dy)
            disc=b*b-4*(ox*ox+oy*oy-r*r)
            t=(-b-math.sqrt(disc))/2 if disc>=0 else -1
            hit=t>0
            if hit:
                xx,yy=cx+t*dx,cy+t*dy
                f.line([(cx,cy),(xx,yy)],'#C4D4E2',1.7)
                if not after:f.circle(xx,yy,6,ORANGE,ORANGE,1)
            else:
                t=(177-cy)/dy
                xx=cx+t*dx
                f.line([(cx,cy),(xx,177)],'#B4C9DB',1.7)
                f.circle(xx,177,6,BLUE,BLUE,1)
        f.circle(armx,army,r,'#D8EADF',TEAL,4)
        f.label(cx-272,354,'手臂截面',31,TEAL)
        f.line([(cx-123,363),(armx-r-8,army-28)],TEAL,2)
        f.circle(cx,cy,13,BLUE,BLUE,1)
        f.label(cx,820,'2D LiDAR',34,BLUE,'middle')
        if after:
            f.label(cx+149,503,'自體回波移除',28,TEAL)
            f.line([(cx+174,477),(armx+79,army+28)],TEAL,2)
            f.label(cx+155,260,'遮擋盲區',29,MUTED)
        else:
            f.label(cx+144,509,'自體回波',29,ORANGE)
            f.line([(cx+165,479),(armx+52,army+61)],ORANGE,2)
    f.line([(775,470),(912,470)],TEAL,5,True)
    f.label(847,414,'姿態＋幾何',27,TEAL,'middle')
    f.text(1630,874,'俯視示意・不依比例',23,MUTED,anchor='end')
    return f.save('03_LiDAR自體濾波')


def qp():
    f=Diagram('分離式控制與整合式全身 QP',1800,920)
    f.label(80,100,'分離式架構',36,BLUE)
    f.label(80,514,'整合式架構',36,TEAL)
    # All text labels name nodes or signals; no explanatory paragraph.
    for x,w,txt,small in [(80,280,'末端任務','位置＋姿態'),(437,365,'任務速度解算',None),(879,335,'安全濾波',None),(1291,429,'底盤＋手臂',None)]:
        f.node(x,154,w,132,txt,BLUE,small)
    for a,b in [(360,437),(802,879),(1214,1291)]: f.line([(a+9,220),(b-13,220)],BLUE,4,True)
    f.path('M 1505 301 V 366 H 610 V 303',MUTED,3,arrow=True)
    f.label(1080,405,'狀態回授',28,MUTED,'middle')
    f.node(879,34,335,79,'安全限制',ORANGE)
    f.line([(1046,120),(1046,140)],ORANGE,3,True)
    f.node(80,601,280,132,'末端任務',TEAL,'位置＋姿態')
    f.node(437,601,365,132,'全身 QP',TEAL,'底盤＋手臂共同求解')
    f.node(879,601,335,132,'下游安全濾波',TEAL)
    f.node(1291,601,429,132,'底盤＋手臂',TEAL)
    for a,b in [(360,437),(802,879),(1214,1291)]: f.line([(a+9,667),(b-13,667)],TEAL,4,True)
    f.node(390,450,460,93,'安全與運動限制',ORANGE)
    f.line([(620,550),(620,587)],ORANGE,3,True)
    f.path('M 1505 748 V 834 H 620 V 750',MUTED,3,arrow=True)
    f.label(1080,882,'狀態回授',28,MUTED,'middle')
    return f.save('04_全身QP控制架構')


def pregrasp():
    f=Diagram('底盤與手臂協同預抓取的任務幾何示意',1600,1050)
    # World coordinates from the known static task; path is illustrative.
    sc=645
    def xy(x,y):return (700+x*sc,570-y*sc)
    def world_box(x,y,w,h,col):
        xx,yy=xy(x-w/2,y+h/2)
        f.rect(xx,yy,w*sc,h*sc,col,ORANGE,6)
    world_box(.74,0,.40,1.,'#FBEADC')
    world_box(-.225,-.320,.16,.20,'#FBEADC')
    f.label(1390,573,'目標物',35,ORANGE)
    f.label(511,946,'底盤障礙物',33,ORANGE)
    f.line([(620,906),(591,850)],ORANGE,2)
    sx,sy=xy(-.70,0)
    ex,ey=xy(-.15,.20)
    r=.3*sc
    f.s.append(f'<circle cx="{sx}" cy="{sy}" r="{r}" fill="#F4F7FA" stroke="{MUTED}" stroke-width="3" stroke-dasharray="12 10"/>')
    f.circle(sx,sy,7,MUTED,MUTED,1)
    f.label(sx,842,'起始底盤',33,MUTED,'middle')
    f.path(f'M {sx} {sy} C {sx+25} {sy-186} {ex-155} {ey-139} {ex-15} {ey-22}',BLUE,5,dashed=True,arrow=True)
    f.label(225,213,'底盤移動',34,BLUE)
    f.circle(ex,ey,r,'#E7F0F8',BLUE,4)
    for a in [0,90,180,270]:
        xx=ex+r*.96*math.cos(math.radians(a))
        yy=ey+r*.96*math.sin(math.radians(a))
        f.s.append(f'<g transform="translate({xx} {yy}) rotate({a})">')
        f.rect(-20,-33,40,66,NAVY,radius=9)
        f.s.append('</g>')
    tx,ty=xy(.30,0)
    arm=[(ex+55,ey),(ex+100,ey-73),(ex+208,ey-22),(tx-49,ty)]
    f.line(arm,'#BFD9CE',32)
    f.line(arm,TEAL,22)
    for x,y in arm:f.circle(x,y,13,'white',TEAL,3)
    f.line([(tx-37,ty-20),(tx+3,ty-20)],NAVY,7)
    f.line([(tx-37,ty+20),(tx+3,ty+20)],NAVY,7)
    f.circle(tx,ty,8,ORANGE,ORANGE,1)
    f.line([(tx+9,ty),(tx+93,ty)],ORANGE,4,True)
    f.label(1030,693,'預抓取 TCP',33,ORANGE,'middle')
    f.line([(980,663),(tx+10,ty+32)],ORANGE,2)
    f.label(620,172,'手臂構形調整',34,TEAL)
    f.line([(767,198),(786,264),(arm[2][0],arm[2][1]-20)],TEAL,2)
    f.label(887,806,'預抓取距離',31,ORANGE)
    surface_x,_=xy(.54,0)
    f.line([(tx,736),(surface_x,736)],ORANGE,3)
    for xx in [tx,surface_x]:f.line([(xx,724),(xx,748)],ORANGE,3)
    f.label(616,88,'可行底盤位姿',34,BLUE,'middle')
    f.line([(616,112),(616,ey-r-15)],BLUE,2)
    f.line([(1390,896),(1490,896)],BLUE,3,True)
    f.line([(1390,896),(1390,805)],BLUE,3,True)
    f.label(1504,906,'x',27,BLUE)
    f.label(1375,787,'y',27,BLUE)
    f.text(1540,1013,'俯視任務示意・虛線非實測軌跡',23,MUTED,anchor='end')
    return f.save('05_底盤手臂協同預抓取')


def interfaces():
    f=Diagram('ROS 2 控制系統與 Gazebo、Isaac Sim 的介面',1650,950)
    f.node(70,360,350,166,'ROS 2 控制系統',BLUE)
    f.node(680,360,310,166,'模擬介面',TEAL)
    f.node(1220,166,350,149,'Gazebo',BLUE)
    f.node(1220,609,350,149,'Isaac Sim',TEAL)
    f.line([(436,395),(666,395)],BLUE,4,True)
    f.label(550,345,'速度命令',31,BLUE,'middle')
    f.line([(666,497),(436,497)],TEAL,4,True)
    f.label(550,558,'scan / odom / TF',26,TEAL,'middle')
    f.line([(1006,404),(1100,404),(1100,210),(1204,210)],BLUE,4,True)
    f.line([(1204,275),(1147,275),(1147,447),(1006,447)],TEAL,4,True)
    f.line([(1006,467),(1065,467),(1065,650),(1204,650)],BLUE,4,True)
    f.line([(1204,708),(1018,708),(1018,510),(1006,510)],TEAL,4,True)
    f.node(680,92,310,100,'/clock',ORANGE)
    f.line([(835,207),(835,345)],ORANGE,3,True)
    f.label(835,52,'模擬時間',30,ORANGE,'middle')
    f.node(70,755,350,116,'影像檢視／記錄',MUTED)
    f.line([(1395,774),(1395,813),(436,813)],TEAL,4,True)
    f.label(902,869,'底盤 RGB 相機',30,TEAL,'middle')
    return f.save('06_模擬平台介面')


def package(paths):
    cw,ch=770,540
    sheet=Image.new('RGB',(1600,1780),'#EAF0F5')
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',25)
    for i,p in enumerate(paths):
        x,y=20+(i%2)*790,20+(i//2)*585
        sheet.paste('white',(x,y,x+cw,y+ch))
        with Image.open(p.with_suffix('.png')) as im:
            im=im.convert('RGBA')
            im.thumbnail((cw-30,ch-25),Image.Resampling.LANCZOS)
            sheet.paste(im,(x+(cw-im.width)//2,y+(ch-im.height)//2),im)
        draw.text((x+8,y+546),p.stem.replace('_','  '),font=font,fill=NAVY)
    sheet.save(OUT/'00_示意圖總覽.jpg',quality=94)
    (OUT/'使用說明.md').write_text('''# 純示意圖

只包含圖形、箭頭、物件名稱與必要圖例，無投影片標題、成果數據或報告段落。

- `.png`：透明背景，高解析，可直接放進簡報。
- `_白底.png`：白色背景版本。
- `.svg`：向量原稿，使用 Noto Sans CJK TC。
- `.pdf`：向量輸出。
- `00_示意圖總覽.jpg`：選圖用縮圖。

01 全身機器人為結構示意，非精確 CAD。
02 距離與速度方向的幾何示意，不是完整安全性證明。
03 俯視光線示意，濾除自身回波不會恢復被遮住的外部觀測。
04 兩種控制架構，下游安全濾波保留在整合式基線中。
05 根據預抓取案例物件位置示意，虛線與手臂構形不是實測軌跡。
06 表示兩個模擬平台的介面分工，並非同時控制或已完成全身跨平台對照。

產生程式：tools/build_fifth_standalone_diagrams.py
沿用的向量繪圖與輸出程式：tools/build_fifth_progress_visuals.py
''')
    zpath=OUT.parent/'第五次進度報告_純示意圖.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():z.write(p,arcname='純示意圖/'+p.name)
        for name in ['build_fifth_standalone_diagrams.py','build_fifth_progress_visuals.py']:
            z.write(ROOT/'tools'/name,arcname='產生程式/'+name)
    print(zpath)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    package([robot(),clearance(),self_filter(),qp(),pregrasp(),interfaces()])
