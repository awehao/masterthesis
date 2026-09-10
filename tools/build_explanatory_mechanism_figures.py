#!/usr/bin/env python3
"""Mechanism diagrams with explicit signals, legends, and matched scene views."""
from pathlib import Path
import json,math,shutil,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import build_model_explanatory_figures as model
import build_fifth_technical_figures as art

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'第五次進度報告素材/機制解說圖'
ASSETS=OUT/'assets'
art.OUT=OUT
BLUE='#246B91';ORANGE='#B46A30';INK='#253540';GRAY='#79848C';GREEN='#257968'


def words(f,x,y,lines,size=30,color=INK,step=43,anchor='start'):
    for i,line in enumerate(lines):f.text(x,y+i*step,line,size,color,400,anchor)


def badge(f,x,y,n,color=BLUE):
    f.circle(x,y,20,color,color,1)
    f.text(x,y+10,str(n),27,'white',500,'middle')


def distance():
    q=np.array([0,0,0,0,.3641,.5263,0,-1.4086,0])
    tri,col,names=model.scene(q,only={'link4','link5','link6','link_eef','uflite_gripper_link','uflite_finger1','uflite_finger2'})
    v=model.View(tri,eye=(0,-1,0));p=ASSETS/'wrist_side.png';v.render(tri,col,p)
    f=art.Fig('02_連桿安全限制如何作用',1900,1270)
    pr=model.inset(f,p,v,80,70,800)
    xyz=tri.reshape(-1,3);near=xyz[np.argmax(xyz[:,0])];px,py=pr(near)
    wallx=1450
    f.rect(wallx,190,198,653,'#EEE9E2','#8F8B84')
    f.rect(wallx-146,190,146,653,'#FAF1E6','none')
    f.line([(wallx-146,162),(wallx-146,866)],ORANGE,2,dash=True)
    f.text(1549,139,'障礙物',36,INK,500,'middle')
    f.circle(px,py,9,BLUE,BLUE,2)
    f.line([(px+12,py),(wallx-12,py)],BLUE,3,True)
    f.text((px+wallx)/2,py-25,'量測距離 d',35,BLUE,500,'middle')
    # Distinguish the sampled-point distance from stopping allowance.
    f.line([(wallx-146,902),(wallx,902)],ORANGE,2)
    for xx in [wallx-146,wallx]:f.line([(xx,889),(xx,915)],ORANGE,2)
    words(f,wallx-73,957,['停止距離 d_stop','基礎餘裕＋延遲＋制動'],28,ORANGE,39,'middle')
    f.leader(82,85,'連桿表面',pr(tri.mean((0,1))),(268,104),size=34)
    f.leader(466,848,'取樣點 pᵢ',(px,py),(661,833),BLUE,size=32)
    # One geometric point, three meaningful velocity directions.
    f.line([(px+23,py+75),(px+193,py+75)],ORANGE,3,True)
    f.text(px+202,py+85,'接近：受限制',30,ORANGE)
    f.line([(px+4,py-31),(px+4,py-198)],INK,2.5,True)
    f.text(px-14,py-225,'切向',29,INK,400,'middle')
    f.line([(px-18,py+116),(px-181,py+116)],INK,2.5,True)
    f.text(px-185,py+166,'退離',29,INK,400,'middle')
    f.line([(90,1038),(1809,1038)],'#CFD5DA',1)
    for x,n,heading,detail in [(122,1,'取得距離','表面取樣距離 d'),(670,2,'扣除取樣誤差','有效距離 d_eff = d − ρ'),(1272,3,'限制接近速度','v_app ≤ α (d_eff − d_stop)')]:
        badge(f,x,1102,n)
        f.text(x+38,1113,heading,32,INK,500)
        f.text(x+38,1167,detail,29,BLUE)
    for x in [563,1164]:f.line([(x,1110),(x+72,1110)],BLUE,2,True)
    f.text(1808,1238,'靜態障礙幾何示意；切向／退離仍受其他關節與安全限制',22,GRAY,anchor='end')
    return f.save()


def cross_section(q,z):
    # Use the actual link collision meshes at the specified plane.
    sections=[]
    for link in model.TREE.findall('link'):
        name=link.get('name')
        if name not in {'link_base','link1','link2','link3','link4','link5','link6','link_eef','uflite_gripper_link','uflite_finger1','uflite_finger2'}:continue
        T=model.K.fk(q,name)
        for col in link.findall('collision'):
            tri=model.mesh.primitive(col.find('geometry'))
            if not len(tri):continue
            o=col.find('origin')
            xyz=np.fromstring(o.get('xyz','0 0 0') if o is not None else '0 0 0',sep=' ')
            rpy=np.fromstring(o.get('rpy','0 0 0') if o is not None else '0 0 0',sep=' ')
            tri=tri@model.rpy_to_rot(*rpy).T+xyz
            tri=tri@T[:3,:3].T+T[:3,3]
            for t in tri[(tri[:,:,2].min(1)<=z)&(tri[:,:,2].max(1)>=z)]:
                hits=[]
                for a,b in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]:
                    if (a[2]-z)*(b[2]-z)<0:
                        hits.append((a+(b-a)*(z-a[2])/(b[2]-a[2]))[:2])
                if len(hits)==2:sections.append(hits)
    return np.array(sections)


def lidar():
    pose=json.loads((ROOT/'第五次進度報告素材/模型解說圖/assets/scan_pose.json').read_text())
    q=np.array(pose['q']);z=pose['plane_z'];seg=cross_section(q,z)
    assert len(seg)>0
    tri,col,names=model.scene(q);v=model.View(tri,eye=(0,-1,.08))
    p=ASSETS/'scan_side.png';v.render(tri,col,p)
    f=art.Fig('03_自體濾波保留什麼移除什麼',2100,1240)
    pr=model.inset(f,p,v,0,163,820)
    source=np.array([0,0,z]);sp=pr(source);cp=pr([.43,0,z])
    f.line([sp,cp],ORANGE,4,True)
    f.circle(*sp,10,BLUE,BLUE,2)
    f.circle(*cp,24,'none',ORANGE,2)
    f.text(81,97,'手臂進入掃描高度',35,INK,500)
    f.leader(54,969,'LiDAR',sp,(179,947),BLUE,size=31)
    f.leader(501,239,'掃到自己的手臂',cp,(727,258),ORANGE,size=29)
    f.text(799,1050,'模型側視',23,GRAY,anchor='end')
    # Matched before/after scan views, exact collision-section ray intersections.
    for i,cx in enumerate([1140,1738]):
        by=785;sc=690
        def xy(p):return (cx-p[1]*sc,by-p[0]*sc)
        f.text(cx,96,'濾波前' if i==0 else '濾波後',35,INK,500,'middle')
        wall=.73
        f.line([xy([wall,-.36]),xy([wall,.36])],INK,6)
        f.text(cx,by-wall*sc-35,'外部牆面',29,INK,400,'middle')
        rays=[]
        for a in np.linspace(-.42,.42,43):
            u=np.array([math.cos(a),math.sin(a)])
            wdist=wall/u[0];best=wdist;hit=False
            for s in seg:
                b=s[1]-s[0];den=u[0]*b[1]-u[1]*b[0]
                if abs(den)<1e-10:continue
                t=(s[0,0]*b[1]-s[0,1]*b[0])/den
                ss=(s[0,0]*u[1]-s[0,1]*u[0])/den
                if t>0 and 0<=ss<=1 and t<best:best=t;hit=True
            rays.append((u,best,wdist,hit))
        blocked=[u for u,t,wt,hit in rays if hit]
        if blocked:
            a0,a1=blocked[0],blocked[-1]
            aa=xy(a0*(wall/a0[0]));bb=xy(a1*(wall/a1[0]))
            f.path(f'M {cx} {by} L {aa[0]} {aa[1]} L {bb[0]} {bb[1]} Z','none',0,'#F1F2F3')
        for u,t,wt,hit in rays:
            ep=xy(u*t)
            f.line([(cx,by),ep],'#CBD2D7',1.1)
            if hit:
                if i==0:f.circle(*ep,5,ORANGE,ORANGE,1)
            else:f.circle(*ep,5,BLUE,BLUE,1)
        for a,b in seg:f.line([xy(a),xy(b)],GRAY,1.5)
        f.circle(cx,by,10,BLUE,BLUE,1)
        f.text(cx,by+51,'LiDAR',28,BLUE,400,'middle')
        f.text(cx,935,'橙點：手臂自體回波' if i==0 else '橙點移除，藍點保留',29,ORANGE if i==0 else BLUE,400,'middle')
        f.text(cx,978,'藍點：外部牆面回波' if i==0 else '遮擋後方仍是未知區域',27,BLUE if i==0 else GRAY,400,'middle')
    f.line([(1430,612),(1469,612)],BLUE,3,True)
    f.line([(82,1090),(2020,1090)],'#CED5DA',1)
    words(f,1050,1147,['依「當下關節姿態＋連桿截面」辨識自體回波；移除回波不等於看穿手臂。'],30,INK,anchor='middle')
    f.text(2018,1210,'剖面由模型計算；姿態、牆面與掃描為幾何示意',21,GRAY,anchor='end')
    (ASSETS/'scan_section.json').write_text(json.dumps({'segments':seg.tolist(),'q':q.tolist(),'plane_z':z},indent=2))
    return f.save()


def comparison():
    a,b=model.load_runs();rows=[a['log'][-1],b['log'][-1]]
    bundles=[model.scene(model.qrow(r)) for r in rows]
    view=model.View(np.concatenate([s[0] for s in bundles]),eye=(1.0,-3.0,1.7))
    f=art.Fig('04_為何停住與如何完成',2300,1450)
    for i,(bundle,r) in enumerate(zip(bundles,rows)):
        x=45+i*1150
        tri,col,names=bundle;p=ASSETS/f'control_{i}.png';view.render(tri,col,p)
        pr=model.inset(f,p,view,x+16,325,1010)
        f.text(x+530,82,'分離式控制' if i==0 else '整合式全身 QP',40,INK,500,'middle')
        words(f,x+530,145,['先求任務速度 → 再做安全修正'] if i==0 else ['在安全限制內，同時求底盤與手臂動作'],28,GRAY,anchor='middle')
        # Explicit target and measured TCP symbols use identical geometry and scale.
        target=pr([.3,0,.55]);tcp=pr(r['tcp'])
        f.circle(*target,13,'white',BLUE,2.5)
        f.line([(target[0]-19,target[1]),(target[0]+19,target[1])],BLUE,2)
        f.line([(target[0],target[1]-19),(target[0],target[1]+19)],BLUE,2)
        f.circle(*tcp,8,ORANGE,ORANGE,1)
        if i==0:
            f.line([tcp,target],ORANGE,3,dash=True)
            f.leader(x+666,297,'TCP 停在目標外',tcp,(x+646,314),ORANGE,size=32)
            f.leader(x+735,1189,'固定目標',target,(x+715,1166),BLUE,size=30)
            bp=np.array([r['base'][0],r['base'][1],.12])
            arr=[pr(bp),pr(bp+[0,-.20,0])]
            f.line(arr,ORANGE,5,True)
            end=arr[-1]
            for d in [-1,1]:f.line([(end[0]-12,end[1]-12*d),(end[0]+12,end[1]+12*d)],ORANGE,3)
            f.text(x+520,1269,'底盤想往目標走，但被安全層擋住',32,INK,400,'middle')
            f.text(x+520,1321,'手臂未接手 → 末端仍有位置誤差',32,ORANGE,400,'middle')
        else:
            f.leader(x+666,297,'TCP 與目標重合',tcp,(x+646,314),BLUE,size=32)
            f.text(x+520,1269,'底盤移位／旋轉，手臂調整構形',32,INK,400,'middle')
            f.text(x+520,1321,'重新分配動作 → 到達預抓取位姿',32,GREEN,400,'middle')
    f.text(1150,1408,'藍色十字：固定目標　橙色圓點：實際 TCP　｜　兩組實驗的架構與構形權重均不同',25,GRAY,400,'middle')
    return f.save()


def sequence():
    a,b=model.load_runs();logs=b['log'];idx=[0,len(logs)//2,len(logs)-1];rows=[logs[j] for j in idx]
    bundles=[model.scene(model.qrow(r)) for r in rows]
    view=model.View(np.concatenate([s[0] for s in bundles]),eye=(.35,-3.0,1.35))
    f=art.Fig('05_固定目標下的全身動作',2800,1310)
    for i,(bundle,r,j) in enumerate(zip(bundles,rows,idx)):
        x=40+i*925
        tri,col,names=bundle;p=ASSETS/f'world_pose_{i}.png';view.render(tri,col,p)
        pr=model.inset(f,p,view,x,147,900)
        f.text(x+440,77,['① 起始','② 接近','③ 到達'][i],40,INK,500,'middle')
        f.text(x+440,130,f't = {r["t"]:.2f} s',26,GRAY,400,'middle')
        target=pr([.3,0,.55])
        f.circle(*target,12,'white',BLUE,2)
        f.line([(target[0]-19,target[1]),(target[0]+19,target[1])],BLUE,2)
        f.line([(target[0],target[1]-19),(target[0],target[1]+19)],BLUE,2)
        tcp=pr(r['tcp']);f.circle(*tcp,8,ORANGE,ORANGE,1)
        f.text(target[0],target[1]-34,'固定目標',27,BLUE,400,'middle')
        if j:
            hist=logs[:j+1]
            bt=[pr([s['base'][0],s['base'][1],.015]) for s in hist]
            tt=[pr(s['tcp']) for s in hist]
            f.line(bt,BLUE,3.2)
            f.line(tt,ORANGE,2.5,dash=True)
        # World origin footprint remains fixed across all panels.
        initial=rows[0]['base']
        ring=[pr([initial[0]+.3*np.cos(a),initial[1]+.3*np.sin(a),.012]) for a in np.linspace(0,2*np.pi,60)]
        f.line(ring,'#9EA8AF',1.5,dash=True)
        words(f,x+440,1106,[['底盤離目標較遠','TCP 尚未到達'][i*0]] if False else [
            '底盤與 TCP 均在起始位置',
            '底盤前移，手臂同步伸展',
            '底盤停在新位置，TCP 到達目標'][i:i+1],32,INK,anchor='middle')
    f.line([(100,1175),(2700,1175)],'#D2D7DC',1)
    f.text(1400,1230,'三格使用相同世界座標與比例　｜　藍線：底盤軌跡　橙虛線：TCP 軌跡　灰虛線：底盤起始輪廓',29,INK,400,'middle')
    f.text(2700,1282,'依同次實驗紀錄重建，模型未各自置中',23,GRAY,anchor='end')
    return f.save()


def package(paths):
    first='01_全身機器人_URDF';src=ROOT/'第五次進度報告素材/專業示意圖'
    for ext in ['.png','_白底.png','.svg','.pdf']:shutil.copyfile(src/(first+ext),OUT/(first+ext))
    sheet=Image.new('RGB',(1900,1430),'#EDF0F2');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',25)
    for i,p in enumerate(paths):
        x,y=20+(i%2)*940,20+(i//2)*705
        sheet.paste('white',(x,y,x+920,y+640))
        with Image.open(p.with_suffix('.png')) as im:
            im=im.copy();im.thumbnail((900,620),Image.Resampling.LANCZOS)
            sheet.paste(im,(x+(920-im.width)//2,y+(640-im.height)//2),im)
        draw.text((x+7,y+652),p.stem.replace('_','  '),font=font,fill=INK)
    sheet.save(OUT/'00_機制解說圖預覽.jpg',quality=95)
    (OUT/'圖檔說明.md').write_text('''# 機制解說圖

第一張原樣保留。本輪以完整解說機制為目標，保留必要文字、箭頭、圖例、訊號意義。
各圖有透明 PNG、白底 PNG、SVG、PDF；模型為內嵌位圖，文字與箭頭為向量。

02：靜态障礙、取樣誤差與停止距離的幾何示意，不代表一般性安全證明。
03：連桿剖面由 URDF collision mesh 與指定掃描平面相交取得，雷射線與牆面為合成示意。
姿態來自前輪構形求解；不是實測 scan，亦未進行該姿態自碰撞驗收。
04：A 終態來自 wholebody_pregrasp.json；B 終態來自 baseline_repeat/mu0p03_run1.json。
兩者控制架構與構形權重均不同，不能當成單一變因消融。
05：三個時刻、底盤與 TCP 軌跡均來自 baseline_repeat/mu0p03_run1.json，同世界座標與比例。
首末列 t 取紀錄值，沒有宣稱是 Isaac 模擬結果。
本轮只重製圖像，未重跑控制實驗。
''')
    zpath=OUT.parent/'第五次進度報告_機制解說圖.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,arcname='機制解說圖/'+str(p.relative_to(OUT)))
        for n in ['build_explanatory_mechanism_figures.py','build_model_explanatory_figures.py','build_fifth_technical_figures.py','render_urdf_diagram_asset.py','build_fifth_progress_visuals.py']:
            z.write(ROOT/'tools'/n,arcname='產生程式/'+n)
    print(zpath)


if __name__=='__main__':
    ASSETS.mkdir(parents=True,exist_ok=True)
    package([distance(),lidar(),comparison(),sequence()])
