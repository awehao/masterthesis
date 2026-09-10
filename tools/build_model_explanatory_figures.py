#!/usr/bin/env python3
"""Model-based explanatory figures: preserve the accepted first illustration."""
from pathlib import Path
from functools import lru_cache
import sys,json,math,shutil,zipfile
import numpy as np
import xml.etree.ElementTree as ET
from scipy.optimize import least_squares
from PIL import Image,ImageDraw,ImageFont
import render_urdf_diagram_asset as mesh
import build_fifth_technical_figures as layout
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics,rpy_to_rot

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'第五次進度報告素材/模型解說圖'
ASSETS=OUT/'assets'
URDF=ROOT/'第五次進度報告素材/專業示意圖/assets/wholebody_expanded.urdf'
K=WholeBodyKinematics.from_urdf_file(str(URDF))
TREE=ET.parse(URDF).getroot()
BLUE='#30647E';INK='#26333D';GRAY='#78828A';AMBER='#A97745'
layout.OUT=OUT


@lru_cache(None)
def local_geoms():
    result=[]
    for link in TREE.findall('link'):
        name=link.get('name')
        for v in link.findall('visual'):
            tri=mesh.primitive(v.find('geometry'))
            if not len(tri):continue
            o=v.find('origin')
            xyz=np.fromstring(o.get('xyz','0 0 0') if o is not None else '0 0 0',sep=' ')
            rpy=np.fromstring(o.get('rpy','0 0 0') if o is not None else '0 0 0',sep=' ')
            tri=tri@rpy_to_rot(*rpy).T+xyz
            color=[.27,.39,.48] if name=='base_link' else [.22,.25,.28] if name.startswith(('rim','roller')) else [.37,.4,.42] if name=='link_eef' or 'gripper' in name or 'finger' in name else [.79,.82,.84]
            result.append((name,tri,np.array(color)))
    return result


def scene(q,only=None):
    triangles=[];colors=[];names=[]
    for name,tri,col in local_geoms():
        if only and name not in only:continue
        T=K.fk(q,name);p=tri@T[:3,:3].T+T[:3,3]
        triangles.append(p);colors.append(np.tile(col,(len(tri),1)));names.extend([name]*len(tri))
    return np.concatenate(triangles),np.concatenate(colors),np.array(names)


class View:
    def __init__(self,tri,eye=(1.7,-2.6,1.55),bounds=None):
        view=np.array(eye,dtype=float);view/=np.linalg.norm(view)
        right=np.cross([0,0,1],view);right/=np.linalg.norm(right)
        self.basis=np.array([right,-np.cross(view,right),view]);self.view=view
        p=tri@self.basis.T
        if bounds is None:
            lo=p[:,:,:2].reshape(-1,2).min(0);hi=p[:,:,:2].reshape(-1,2).max(0)
        else:
            ps=np.array(bounds)@self.basis.T;lo=ps[:,:2].min(0);hi=ps[:,:2].max(0)
        self.scale=min(1700/(hi-lo)[0],1700/(hi-lo)[1])
        self.offset=np.array([950,950])-(lo+hi)/2*self.scale
    def project(self,p):
        a=np.asarray(p)@self.basis.T
        return a[...,:2]*self.scale+self.offset
    def render(self,tri,colors,path):
        pp=self.project(tri);depth=(tri@self.basis.T)[:,:,2].mean(1)
        n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
        light=np.array([-1,-2,4.]);light/=np.linalg.norm(light)
        shade=.46+.42*np.abs(n@light)+.12*np.abs(n@self.view)
        col=np.clip(colors*shade[:,None]*255,0,255).astype(np.uint8)
        im=Image.new('RGBA',(1900,1900),(255,255,255,0));d=ImageDraw.Draw(im)
        for i in np.argsort(depth):d.polygon([tuple(v) for v in pp[i]],fill=tuple(col[i])+(255,))
        im.save(path)
    def point(self,q,link):return self.project(K.fk(q,link)[:3,3])


def inset(f,path,view,x,y,w):
    f.bitmap(path,x,y,w,w)
    return lambda p: tuple(np.array([x,y])+view.project(p)*w/1900)


def coverage():
    q=np.array([0,0,0,0,.3641,.5263,0,-1.4086,0])
    tri,col,names=scene(q,only={'link4','link5','link6','link_eef','uflite_gripper_link','uflite_finger1','uflite_finger2'})
    view=View(tri,eye=(1.7,-2.6,1.55));path=ASSETS/'wrist.png';view.render(tri,col,path)
    f=layout.Fig('02_連桿表面安全_模型近照',1700,1000)
    pr=inset(f,path,view,100,120,770)
    # The selected 3D surface points are illustrative, not controller telemetry.
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);visible=(n@view.view)>0
    ctr=tri.mean(1)
    candidates=ctr[visible & np.isin(names,['link5','link6','uflite_gripper_link','uflite_finger1','uflite_finger2'])]
    projected=view.project(candidates)
    chosen=[]
    for i in np.argsort(projected[:,0])[::-1]:
        pp=projected[i]
        if not chosen or min(np.linalg.norm(pp-projected[j]) for j in chosen)>65:
            chosen.append(i)
        if len(chosen)>=65:break
    for i in chosen:
        xx,yy=pr(candidates[i]);f.circle(xx,yy,3.3,BLUE,BLUE,.7)
    # Wireframe boundary instead of a giant solid box.
    f.path('M 1200 180 L 1500 275 L 1500 925 L 1200 830 Z',GRAY,1.5,'#F6F3EF')
    pi=candidates[np.argmax(candidates[:,0])];p=pr(pi)
    target=(1200,p[1])
    f.circle(*p,6,AMBER,AMBER,1)
    f.line([p,target],AMBER,2.8,True)
    f.text((p[0]+target[0])/2,(p[1]+target[1])/2-27,'表面距離 d',30,AMBER,400,'middle')
    f.leader(142,107,'腕部與夾爪',pr(ctr.mean(0)),(342,125),size=31)
    f.leader(92,913,'連桿表面取樣',pr(candidates[chosen[len(chosen)//2]]),(324,890),BLUE,size=30)
    f.text(1360,969,'障礙物表面',30,INK,400,'middle')
    f.text(82,972,'表面取樣與障礙面為幾何示意',20,GRAY)
    return f.save()


def scan():
    # Construct a model pose that places the gripper across the documented scan plane.
    def fun(a):
        q=np.r_[np.zeros(3),a]
        return K.fk(q,'uflite_gripper_link')[:3,3]-[.43,0,.2652]
    sol=least_squares(fun,[0,.8,1.2,0,-.6,0],bounds=([-2.93,-2.44,-.061,-2.93,-1.98,-2.93],[2.93,2.44,2.93,2.93,1.98,2.93]),max_nfev=500)
    q=np.r_[np.zeros(3),sol.x]
    tri,col,names=scene(q)
    view=View(tri,eye=(2.2,-3.2,2.1));path=ASSETS/'scan_pose.png';view.render(tri,col,path)
    f=layout.Fig('03_雷射自體遮蔽_模型示意',1700,1100)
    pr=inset(f,path,view,180,160,840)
    # Scanner plane is a schematic ring at the documented sensor height.
    z=.2652
    ring=np.array([[.47*np.cos(a),.47*np.sin(a),z] for a in np.linspace(0,2*np.pi,120)])
    points=[pr(p) for p in ring]
    f.line(points,BLUE,1.4,dash=True)
    # Current posture's gripper cross-section and a magnified echo strip.
    center=K.fk(q,'uflite_gripper_link')[:3,3]
    cx,cy=pr(center)
    f.circle(cx,cy,43,'none',AMBER,2)
    f.line([(cx+42,cy),(1270,cy),(1270,393)],GRAY,1.5)
    f.rect(1202,203,397,378,'#FAFBFC','#D5DBDF',5)
    f.text(1400,250,'掃描平面交會處',28,INK,400,'middle')
    f.path('M 1280 358 C 1320 296 1480 301 1516 365 C 1500 422 1350 453 1280 358 Z',GRAY,1.6,'#E0E5E8')
    for a in np.linspace(.13*np.pi,.84*np.pi,16):
        xx=1398+113*np.cos(a);yy=365+57*np.sin(a)
        f.circle(xx,yy,4,AMBER,AMBER,1)
    f.text(1400,500,'手臂自體回波',28,AMBER,400,'middle')
    f.text(1400,544,'依關節姿態與網格截面移除',21,GRAY,400,'middle')
    f.leader(102,86,'手臂進入雷射掃描平面',(cx,cy),(437,105),size=31)
    ringpoint=pr([-.37,-.29,z])
    f.leader(99,999,'2D LiDAR 掃描平面',ringpoint,(406,976),BLUE,size=29)
    f.text(1600,1064,'構形與回波為幾何示意，非實測掃描',20,GRAY,anchor='end')
    (ASSETS/'scan_pose.json').write_text(json.dumps({'q':q.tolist(),'plane_z':z,'construction_error_m':float(np.linalg.norm(fun(sol.x)))},indent=2))
    return f.save()


def load_runs():
    a=json.loads((ROOT/'evaluation/results/wholebody_pregrasp.json').read_text())
    b=json.loads((ROOT/'evaluation/results/baseline_repeat/mu0p03_run1.json').read_text())
    return a,b


def qrow(r):return np.array(r['base']+r['q'])


def ghost_scene(f,pr):
    # Only the near target surface and the smaller base obstacle are illustrated.
    face=[pr([.54,y,z]) for y,z in [(-.18,.32),(.18,.32),(.18,.78),(-.18,.78),(-.18,.32)]]
    f.line(face,'#78828A',1.1,dash=True)
    for z in [0,.6]:
        p=[pr([x,y,z]) for x,y in [(-.305,-.42),(-.145,-.42),(-.145,-.22),(-.305,-.22),(-.305,-.42)]]
        f.line(p,AMBER,1.6)
    for x,y in [(-.305,-.42),(-.145,-.42),(-.145,-.22),(-.305,-.22)]:f.line([pr([x,y,0]),pr([x,y,.6])],AMBER,1.6)


def compare():
    a,b=load_runs();rows=[a['log'][-1],b['log'][-1]]
    bundles=[scene(qrow(r)) for r in rows]
    alltri=np.concatenate([s[0] for s in bundles])
    view=View(alltri,eye=(1.5,-3.5,2.0))
    f=layout.Fig('04_任務停滯與完成_模型對照',2100,1120)
    for i,((tri,col,names),r) in enumerate(zip(bundles,rows)):
        path=ASSETS/f'compare_{i}.png';view.render(tri,col,path)
        x=90+i*1040
        pr=inset(f,path,view,x,150,840)
        ghost_scene(f,pr)
        f.text(x+440,78,'分離式控制' if i==0 else '整合式全身 QP',36,INK,500,'middle')
        target=pr([.3,0,.55]);tcp=pr(r['tcp'])
        f.circle(*target,11,'white',BLUE,2)
        f.line([(target[0]-16,target[1]),(target[0]+16,target[1])],BLUE,1.7)
        f.line([(target[0],target[1]-16),(target[0],target[1]+16)],BLUE,1.7)
        f.circle(*tcp,6,AMBER if i==0 else BLUE,AMBER if i==0 else BLUE,1)
        if i==0:
            f.line([tcp,target],AMBER,2,dash=True)
            f.leader(x+650,220,'末端位置偏差',tcp,(x+629,237),AMBER,size=28)
        else:f.leader(x+650,220,'到達預抓取位姿',tcp,(x+629,237),BLUE,size=28)
        f.text(x+440,1056,'安全停住・末端未到達' if i==0 else '底盤與手臂共同完成任務',29,INK,400,'middle')
    f.text(2030,1100,'依兩組既有實驗終態重建；構形權重設定亦不同',19,GRAY,anchor='end')
    return f.save()


def sequence():
    a,b=load_runs();logs=b['log'];rows=[logs[0],logs[len(logs)//2],logs[-1]]
    qs=[qrow(r) for r in rows];bundles=[scene(q) for q in qs]
    # Align each model to its own base only for the three pose views.
    for i,q in enumerate(qs):
        qq=q.copy();qq[:2]=0
        bundles[i]=scene(qq)
    view=View(np.concatenate([s[0] for s in bundles]),eye=(1.8,-3.3,1.8))
    f=layout.Fig('05_全身預抓取動作序列',2400,1200)
    for i,(tri,col,names) in enumerate(bundles):
        path=ASSETS/f'sequence_{i}.png';view.render(tri,col,path)
        f.bitmap(path,35+i*790,160,750,750)
        f.text(410+i*790,99,['起始構形','底盤與手臂同動','預抓取構形'][i],34,INK,500,'middle')
    for x in [778,1568]:f.line([(x,553),(x+41,553)],BLUE,2.5,True)
    # Actual recorded base trajectory provides motion context without invented paths.
    base=np.array([r['base'][:2] for r in logs])
    def xy(v):return (700+(v[0]+.7)*1650,1060-v[1]*550)
    f.line([xy(v) for v in base],BLUE,3)
    for r,label in zip(rows,['起點','同動','終點']):
        pp=xy(r['base'][:2]);f.circle(*pp,5,BLUE,BLUE,1);f.text(pp[0],pp[1]+43,label,23,GRAY,400,'middle')
    f.text(606,1080,'底盤軌跡',28,INK,400,'end')
    f.text(2325,1165,'模型圖已按底盤位置對齊；下方為同次實驗的底盤軌跡',19,GRAY,anchor='end')
    return f.save()


def package(paths):
    # The accepted figure is copied byte-for-byte, not re-rendered.
    src=ROOT/'第五次進度報告素材/專業示意圖'
    first='01_全身機器人_URDF'
    for ext in ['.png','.svg','.pdf','_白底.png']:shutil.copyfile(src/(first+ext),OUT/(first+ext))
    paths=[OUT/first]+paths
    sheet=Image.new('RGB',(1700,1190),'#EEF0F2');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',23)
    for i,p in enumerate(paths[1:]):
        x,y=20+(i%2)*840,20+(i//2)*585
        sheet.paste('white',(x,y,x+820,y+520))
        with Image.open(p.with_suffix('.png')) as im:
            im=im.copy();im.thumbnail((790,500),Image.Resampling.LANCZOS)
            sheet.paste(im,(x+(820-im.width)//2,y+(520-im.height)//2),im)
        draw.text((x+7,y+533),p.stem.replace('_','  '),font=font,fill=INK)
    sheet.save(OUT/'00_新繪四張示意圖預覽.jpg',quality=95)
    old_preview=OUT/'00_模型解說图預覽.jpg'
    if old_preview.exists():old_preview.unlink()
    (OUT/'圖檔說明.md').write_text('''# 模型解說圖

第一張保持已接受版本不變。其餘四張以機器人實際網格、局部放大、實驗姿態對照與動作序列重做。
PNG 有透明版與白底版；SVG/PDF 內的文字和標註為向量，模型為內嵌渲染圖。

02 局部近照取自真實視覺網格；藍色取樣點及障礙面為概念示意，不是控制器紀錄。
03 以 FK 求得交會掃描平面的示意姿態，掃描高度為 base_link 0.2152 m＋離地 0.05 m。
回波局部示意不是實測掃描，也不構成該姿態自碰撞驗收。
04 左側取 evaluation/results/wholebody_pregrasp.json 終態；右側取 baseline_repeat/mu0p03_run1.json 終態。
兩者同任務但控制架構與構形權重皆不同，不是單一變因消融。
05 三個姿態與下方底盤軌跡都來自 baseline_repeat/mu0p03_run1.json。
模型三視圖為方便比較，以各自底盤為中心對齊；並非三個模型的實際世界相對位置。
所有渲染都是離線重建，不是模擬器截圖。

不再提供已被否定的抽象模擬平台方框圖；第一張加上本輪四張可配合四頁進度內容使用。
''')
    archive=OUT.parent/'第五次進度報告_模型解說圖.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,arcname='模型解說圖/'+str(p.relative_to(OUT)))
        for name in ['build_model_explanatory_figures.py','build_fifth_technical_figures.py','render_urdf_diagram_asset.py','build_fifth_progress_visuals.py']:
            z.write(ROOT/'tools'/name,arcname='產生程式/'+name)
    print(archive)


if __name__=='__main__':
    ASSETS.mkdir(parents=True,exist_ok=True)
    package([coverage(),scan(),compare(),sequence()])
