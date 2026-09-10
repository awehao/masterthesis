#!/usr/bin/env python3
"""Restrained technical illustrations with real URDF geometry and vector labels."""
from pathlib import Path
import base64,json,math,zipfile
from html import escape
from PIL import Image,ImageDraw,ImageFont
import build_fifth_progress_visuals as renderer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'第五次進度報告素材/專業示意圖'
INK='#26333D';GRAY='#78828A';LIGHT='#F1F3F4';BLUE='#30647E';AMBER='#A97745';LINE='#AAB2B8'


class Fig:
    def __init__(self,name,w=1500,h=950):
        self.name,self.w,self.h=name,w,h
        self.s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',f'<title>{escape(name)}</title>','<defs>']
        for c in [INK,GRAY,BLUE,AMBER]:
            self.s.append(f'<marker id="m{c[1:]}" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10 Z" fill="{c}"/></marker>')
        self.s.append('</defs>')
    def text(self,x,y,t,size=28,color=INK,weight=400,anchor='start'):
        self.s.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-family="Noto Sans CJK TC, sans-serif" font-weight="{weight}" text-anchor="{anchor}">{escape(t)}</text>')
    def rect(self,x,y,w,h,fill='white',stroke=LINE,r=3,dash=False):
        self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.6" rx="{r}"'+(' stroke-dasharray="6 6"' if dash else '')+'/>')
    def line(self,pts,c=INK,width=1.8,arrow=False,dash=False):
        self.s.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" stroke="{c}" stroke-width="{width}" stroke-linejoin="round"'+(f' marker-end="url(#m{c[1:]})"' if arrow else '')+(' stroke-dasharray="7 6"' if dash else '')+'/>')
    def circle(self,x,y,r,fill='white',stroke=BLUE,width=1.8):
        self.s.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')
    def path(self,d,c=INK,width=2,fill='none',dash=False,arrow=False):
        self.s.append(f'<path d="{d}" fill="{fill}" stroke="{c}" stroke-width="{width}"'+(' stroke-dasharray="7 6"' if dash else '')+(f' marker-end="url(#m{c[1:]})"' if arrow else '')+'/>')
    def node(self,x,y,w,h,label,small=None,accent=False):
        self.rect(x,y,w,h,'#F4F8FA' if accent else 'white',BLUE if accent else LINE)
        self.text(x+w/2,y+h/2+(10 if small is None else -4),label,28,BLUE if accent else INK,500,'middle')
        if small:self.text(x+w/2,y+h/2+31,small,22,GRAY,400,'middle')
    def leader(self,x,y,label,end,elbow=None,color=INK,anchor='start',size=29):
        self.text(x,y,label,size,color,500,anchor)
        ex,ey=end
        if elbow is None:elbow=(x,y+14)
        self.line([elbow,(elbow[0]+(30 if anchor=='start' else -30),elbow[1]),(ex,ey)],GRAY,1.5)
        self.circle(ex,ey,3,color,color,1)
    def bitmap(self,path,x,y,w,h):
        data=base64.b64encode(Path(path).read_bytes()).decode()
        self.s.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/png;base64,{data}"/>')
    def save(self):
        p=OUT/self.name;svg='\n'.join(self.s+['</svg>'])
        p.with_suffix('.svg').write_text(svg)
        renderer.W,renderer.H=self.w,self.h
        renderer.render_files(svg,p)
        with Image.open(p.with_suffix('.png')) as im:
            canvas=Image.new('RGBA',im.size,'white');canvas.alpha_composite(im)
            canvas.convert('RGB').save(OUT/(self.name+'_白底.png'))
        return p


def robot():
    f=Fig('01_全身機器人_URDF',1500,1030)
    f.bitmap(OUT/'assets/robot_urdf.png',300,35,910,910)
    pts=json.loads((OUT/'assets/robot_urdf.json').read_text())['projected_labels']
    def p(k):return (300+pts[k][0]*910/2200,35+pts[k][1]*910/2200)
    f.leader(72,205,'Lite 6 機械臂',p('arm'),(270,221))
    f.text(72,241,'6 自由度',23,GRAY)
    f.leader(1213,279,'腕部相機',p('camera'),(1194,290))
    f.leader(1224,495,'夾爪／TCP',p('tcp'),(1205,485),BLUE)
    f.leader(72,656,'全向底盤',p('base'),(242,671))
    f.text(72,692,'x、y、θ：3 自由度',23,GRAY)
    f.leader(72,859,'全向輪',p('wheel'),(213,873))
    f.text(750,1002,'q = [ x, y, θ, q₁, …, q₆ ]ᵀ',31,INK,400,'middle')
    return f.save()


def geometry():
    f=Fig('02_連桿距離與速度',1500,880)
    # A two-joint link, gray body and blue measured surface samples.
    f.path('M 263 574 C 196 582 165 516 205 470 L 650 250 C 713 217 784 266 781 326 C 779 362 755 390 725 405 Z',INK,2,'#E7EAEC')
    f.circle(239,517,32,'#F7F8F9',GRAY,1.6)
    f.circle(710,326,37,'#F7F8F9',GRAY,1.6)
    samples=[(260,566),(320,539),(380,511),(440,482),(500,453),(560,425),(620,398),(683,410),(730,394),(767,367),(781,326),(771,286),(737,258),(690,242),(643,256),(583,286),(523,316),(463,345),(403,375),(343,405),(283,434),(224,465)]
    for x,y in samples:f.circle(x,y,4.3,BLUE,BLUE,.8)
    f.rect(1140,170,245,465,'#F4F1ED',GRAY)
    f.rect(1035,170,105,465,'#FAF8F5','none')
    f.line([(1035,142),(1035,662)],AMBER,1.6,dash=True)
    f.line([(1035,126),(1140,126)],AMBER,1.6)
    for x in [1035,1140]:f.line([(x,116),(x,136)],AMBER,1.6)
    f.text(1087,97,'d_stop',27,AMBER,400,'middle')
    f.text(1262,699,'障礙物表面',29,INK,400,'middle')
    px,py=781,326
    f.circle(px,py,7,BLUE,BLUE,1)
    f.line([(px+10,py),(1128,py)],BLUE,2.5,True)
    f.text(952,299,'距離 dᵢ／法向 nᵢ',26,BLUE,400,'middle')
    f.line([(px,py-12),(px,151)],INK,2,True)
    f.text(674,125,'切向速度',27,INK)
    f.line([(px+7,py+10),(940,462)],INK,2,True)
    f.text(952,497,'Jᵢ(q)ν',29,INK)
    f.leader(591,632,'表面點 pᵢ',(px,py),(753,620),BLUE)
    f.leader(154,701,'連桿表面取樣點',(410,497),(374,684),BLUE)
    f.text(1347,810,'幾何示意',20,GRAY,anchor='end')
    return f.save()


def lidar():
    f=Fig('03_自體濾波與遮擋盲區',1600,850)
    for cx,after in [(420,False),(1210,True)]:
        cy=641;arm=(cx+42,364);r=67
        f.text(cx,69,'（b）濾波後' if after else '（a）濾波前',29,INK,500,'middle')
        f.line([(cx-238,159),(cx+238,159)],INK,6)
        f.text(cx,124,'外部牆面',25,INK,400,'middle')
        a0=math.atan2(arm[1]-cy,arm[0]-cx);spread=math.asin(r/math.hypot(arm[0]-cx,arm[1]-cy))
        xa=cx+(159-cy)/math.sin(a0-spread)*math.cos(a0-spread)
        xb=cx+(159-cy)/math.sin(a0+spread)*math.cos(a0+spread)
        if after:f.path(f'M {cx} {cy} L {xa} 159 L {xb} 159 Z','none',0,'#EFF1F2')
        for k in range(25):
            a=math.radians(-115+k*2.07);dx,dy=math.cos(a),math.sin(a)
            ox,oy=cx-arm[0],cy-arm[1];b=2*(ox*dx+oy*dy);disc=b*b-4*(ox*ox+oy*oy-r*r)
            t=(-b-math.sqrt(disc))/2 if disc>=0 else -1
            if t>0:
                xx,yy=cx+t*dx,cy+t*dy
                f.line([(cx,cy),(xx,yy)],LINE,1)
                if not after:f.circle(xx,yy,4,AMBER,AMBER,.7)
            else:
                xx=cx+(159-cy)/dy*dx
                f.line([(cx,cy),(xx,159)],'#CBD2D7',1)
                f.circle(xx,159,3.6,BLUE,BLUE,.7)
        f.circle(*arm,r,'#E1E5E7',GRAY,1.6)
        f.circle(cx,cy,47,'white',INK,1.8)
        f.circle(cx,cy,6,BLUE,BLUE,1)
        f.text(cx,738,'2D LiDAR',27,INK,400,'middle')
        f.leader(cx-294,354,'手臂截面',(arm[0]-r+4,arm[1]-10),(cx-151,366),INK,size=25)
        if after:
            f.leader(cx+169,257,'盲區',((xa+xb)/2,223),(cx+154,267),GRAY,size=25)
            f.leader(cx+133,464,'移除自體回波',(arm[0]+28,arm[1]+59),(cx+118,451),BLUE,size=24)
        else:f.leader(cx+141,464,'自體回波',(arm[0]+28,arm[1]+59),(cx+126,451),AMBER,size=25)
    return f.save()


def architecture():
    f=Fig('04_全身QP架構對照',1700,880)
    for yy,new in [(167,False),(570,True)]:
        f.text(62,yy-83,'（b）整合式全身控制' if new else '（a）任務解算與安全修正分離',30,INK,500)
        f.node(62,yy,260,111,'末端任務','位置與姿態')
        f.node(411,yy,330,111,'全身 QP' if new else '任務速度解算',accent=new)
        f.node(830,yy,310,111,'下游安全濾波' if new else '安全濾波')
        f.node(1229,yy,365,111,'底盤＋機械臂')
        for a,b in [(322,411),(741,830),(1140,1229)]:f.line([(a+7,yy+55),(b-12,yy+55)],INK,2,True)
        cx=576 if new else 985
        f.text(cx,yy-33,'安全與運動限制',25,BLUE,400,'middle')
        f.line([(cx,yy-21),(cx,yy-10)],BLUE,1.8,True)
        f.line([(1411,yy+124),(1411,yy+183),(576,yy+183),(576,yy+124)],GRAY,1.6,True)
        f.text(1020,yy+220,'狀態回授',24,GRAY,400,'middle')
    return f.save()


def task():
    f=Fig('05_預抓取任務_URDF',1600,1040)
    f.bitmap(OUT/'assets/pregrasp_urdf.png',240,45,980,980)
    pts=json.loads((OUT/'assets/pregrasp_urdf.json').read_text())['projected_labels']
    def p(k):return (240+pts[k][0]*980/2200,45+pts[k][1]*980/2200)
    f.leader(69,209,'機械臂',p('arm'),(218,224))
    f.leader(69,794,'全向底盤',p('base'),(230,780))
    f.leader(1280,193,'目標物',p('target_box'),(1257,207))
    f.leader(1280,477,'預抓取 TCP',p('tcp'),(1257,465),BLUE)
    f.leader(1280,805,'底盤障礙物',p('obstacle'),(1257,791),AMBER)
    f.text(1454,1008,'依 URDF 與既有完成案例姿態重建',21,GRAY,anchor='end')
    return f.save()


def simulator():
    f=Fig('06_模擬平台介面',1550,880)
    f.node(65,338,320,137,'ROS 2 控制系統')
    f.node(645,338,260,137,'模擬介面',accent=True)
    f.rect(1070,127,414,565,'none',LINE,3,True)
    f.text(1277,97,'模擬平台',27,INK,400,'middle')
    f.node(1127,194,300,111,'Gazebo')
    f.node(1127,514,300,111,'Isaac Sim')
    f.line([(398,367),(632,367)],INK,2,True)
    f.text(515,337,'速度命令',25,INK,400,'middle')
    f.line([(632,445),(398,445)],BLUE,2,True)
    f.text(515,503,'scan · odom · TF',24,BLUE,400,'middle')
    f.line([(918,365),(975,365),(975,229),(1114,229)],INK,1.8,True)
    f.line([(1114,273),(1018,273),(1018,405),(918,405)],BLUE,1.8,True)
    f.line([(918,420),(975,420),(975,549),(1114,549)],INK,1.8,True)
    f.line([(1114,593),(1018,593),(1018,461),(918,461)],BLUE,1.8,True)
    f.text(775,207,'/clock',28,INK,400,'middle')
    f.line([(775,226),(775,325)],GRAY,1.8,True)
    f.node(65,693,320,111,'影像檢視／記錄')
    f.line([(1277,638),(1277,748),(398,748)],BLUE,1.8,True)
    f.text(789,799,'底盤 RGB 影像',25,BLUE,400,'middle')
    return f.save()


def package(paths):
    sheet=Image.new('RGB',(1700,1725),'#EEF0F2');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',25)
    for i,p in enumerate(paths):
        x,y=20+(i%2)*840,20+(i//2)*566
        sheet.paste('white',(x,y,x+820,y+512))
        with Image.open(p.with_suffix('.png')) as src:
            im=src.copy();im.thumbnail((790,490),Image.Resampling.LANCZOS)
            sheet.paste(im,(x+(820-im.width)//2,y+(512-im.height)//2),im)
        draw.text((x+9,y+524),p.stem.replace('_','  '),font=font,fill=INK)
    sheet.save(OUT/'00_新版示意圖預覽.jpg',quality=94)
    (OUT/'圖檔說明.md').write_text('''# 技術示意圖新版

PNG 為透明背景；同名「_白底.png」可直接放入白底投影片；SVG 與 PDF 保留向量文字與線條。
01 與 05 的機器人為 URDF 真實視覺網格的正投影 CPU 渲染，SVG 內嵌渲染圖，模型不是純向量。
其餘四張為純向量示意，僅保留圖形與標註。

01 幾何來源：omni_bot_wholebody.urdf.xacro 展開；工具姿態由 pregrasp_reference 產生。
05 使用 evaluation/results/baseline_repeat/mu0p03_run1.json 的最後一筆底盤與關節姿態，
搭配 arm_barrier_test 場景中的目標箱與底盤障礙位置重建，不是模擬器截圖。
02 為距離與速度幾何概念圖，03 為自體濾波俯視概念圖，均不按實機比例。
04 保留實際基線的下游安全濾波；05 不繪製虛構運動軌跡；06 僅表示介面關係。

生成：tools/build_fifth_technical_figures.py
網格渲染：tools/render_urdf_diagram_asset.py
SVG／PNG／PDF 輸出：tools/build_fifth_progress_visuals.py
''')
    archive=OUT.parent/'第五次進度報告_技術示意圖新版.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,arcname='技術示意圖/'+str(p.relative_to(OUT)))
        for name in ['build_fifth_technical_figures.py','render_urdf_diagram_asset.py','build_fifth_progress_visuals.py']:
            z.write(ROOT/'tools'/name,arcname='產生程式/'+name)
    print(archive)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    package([robot(),geometry(),lidar(),architecture(),task(),simulator()])
