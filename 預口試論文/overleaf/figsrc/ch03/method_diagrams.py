"""Redesigned chapter-3 illustrations: one visual question per figure."""
from pathlib import Path
import os, tempfile, json
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"thesis_ch03_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Polygon, Arc
import numpy as np
from scipy.spatial.transform import Rotation
from method_scene_geometry import Scene, solve_pose, IK_RECORDS
from figure_style import CAPTIONS, prepare

OUT=Path(__file__).resolve().parent
FONT_PATH=OUT.parents[1]/"fonts/kaiu.ttf"
if not FONT_PATH.is_file(): raise FileNotFoundError(FONT_PATH)
FONT=FontProperties(fname=str(FONT_PATH)); FONT.set_math_fontfamily("stix")
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"path",
                     "mathtext.fontset":"stix","savefig.facecolor":"white"})
INK="#151515"; GRAY="#777777"; LIGHT="#eeeeee"
MANIFEST=[]; BOX_CHECKS=[]


def text(ax,x,y,value,size=10.5,**kwargs):
    options=dict(fontproperties=FONT,fontsize=size,ha="center",va="center",
                 color=INK,linespacing=1.35,zorder=20)
    options.update(kwargs)
    return ax.text(x,y,value,**options)


def arrow(ax,a,b,rad=0,ls="-",lw=1.2,scale=10,**kwargs):
    p=FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=scale,
                      connectionstyle=f"arc3,rad={rad}",linewidth=lw,
                      color=INK,linestyle=ls,shrinkA=0,shrinkB=0,zorder=12,**kwargs)
    ax.add_patch(p); return p


def line(ax,points,**kwargs):
    p=np.asarray(points); opts=dict(color=INK,lw=.85,zorder=10);opts.update(kwargs)
    ax.plot(p[:,0],p[:,1],**opts)


def plain(fig,rect,xlim,ylim,equal=False):
    ax=fig.add_axes(rect);ax.set(xlim=xlim,ylim=ylim);ax.set_axis_off()
    if equal: ax.set_aspect("equal")
    return ax


def node(ax,x,y,w,h,title,detail=None,fill="white",size=10.8):
    r=Rectangle((x-w/2,y-h/2),w,h,ec=INK,fc=fill,lw=1.05,zorder=5)
    ax.add_patch(r)
    if detail:
        artists=[text(ax,x,y+.20,title,size),text(ax,x,y-.23,detail,9.4)]
    else: artists=[text(ax,x,y,title,size)]
    BOX_CHECKS.extend((r,t) for t in artists)
    return r


def robot_plan(ax,p,r=.30,yaw=0,ghost=False):
    p=np.asarray(p);color=GRAY if ghost else INK
    ax.add_patch(Circle(p,r,ec=color,fc="white",lw=1.25,ls="--" if ghost else "-",zorder=7))
    if ghost:return
    for a in np.arange(4)*np.pi/2+yaw:
        normal=np.array([np.cos(a),np.sin(a)]);tangent=np.array([-normal[1],normal[0]])
        c=p+r*.245/.300*normal
        vertices=[c+i*r*.07*normal+j*r*.17*tangent for i,j in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        ax.add_patch(Polygon(vertices,ec=INK,fc=LIGHT,lw=.65,zorder=8))
        for a_ in (-.5,0,.5):
            mid=c+a_*r*.17*tangent
            line(ax,[mid-r*.07*normal,mid+r*.07*normal],lw=.5)
    direction=np.array([np.cos(yaw),np.sin(yaw)])
    arrow(ax,p-r*.35*direction,p+r*.25*direction,lw=.9,scale=6)


def person(ax,p):
    x,y=p
    ax.add_patch(Circle((x,y),.25,ec=GRAY,fc="#f3f3f3",lw=.7,zorder=6))
    ax.add_patch(Circle((x,y+.105),.065,ec=INK,fc=INK,lw=.6,zorder=9))
    line(ax,[(x,y+.03),(x,y-.08)],lw=2.8,zorder=9)
    line(ax,[(x-.10,y-.01),(x+.10,y-.01)],lw=2,zorder=9)
    line(ax,[(x,y-.075),(x-.075,y-.18)],lw=1.7,zorder=9)
    line(ax,[(x,y-.075),(x+.075,y-.18)],lw=1.7,zorder=9)


def callout(ax,scene,p,label,position,ha="center"):
    ax.annotate(label,xy=scene.p(p),xycoords="data",xytext=position,textcoords="axes fraction",
                ha=ha,va="center",fontproperties=FONT,fontsize=10.5,zorder=25,
                arrowprops=dict(arrowstyle="-",color=INK,lw=.75,shrinkA=3,shrinkB=3),
                bbox=dict(fc="white",ec="none",pad=1.5))


def export(fig,name,caption,placement,purpose):
    style=prepare(fig)
    caption=CAPTIONS.get(name,caption)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text(): continue
        b=artist.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > w+1 or b.y1 > h+1:
            raise RuntimeError(f"{name}: text outside page: {artist.get_text()}")
    for box,artist in BOX_CHECKS:
        if artist.figure is not fig:continue
        b=artist.get_window_extent(renderer);r=box.get_window_extent(renderer)
        if not (r.x0+1 < b.x0 and b.x1 < r.x1-1 and r.y0+1 < b.y0 and b.y1 < r.y1-1):
            raise RuntimeError(f"{name}: text outside box: {artist.get_text()}")
    for ext in ("pdf","svg","png"):
        fig.savefig(OUT/f"{name}.{ext}",dpi=260,metadata={"Creator":"Original thesis diagrams; Matplotlib"})
    plt.close(fig)
    MANIFEST.append(dict(file=name,caption=caption,placement=placement,purpose=purpose,
                         source="本研究自行繪製之方法圖，未使用實驗結果。",
                         chinese_font="DFKai-SB",math_font="STIX",bounds_check="passed",
                         **style))
    print(name,flush=True)


def setpoint_tracking():
    fig=plt.figure(figsize=(7.2,4.35))
    top=plain(fig,[.025,.68,.95,.27],(0,12),(0,2))
    for x,title,detail in [(1.8,"命令 $u_a$","設定點的變化率"),
                            (6.,"位置設定點 $s_a$","控制器要求的位置"),
                            (10.2,"實際位置 $q_a$","關節量測的位置")]:
        node(top,x,1,3.35,1.22,title,detail)
    arrow(top,(3.475,1),(4.325,1));text(top,3.9,1.52,"積分",9.8)
    arrow(top,(7.675,1),(8.525,1));text(top,8.1,1.52,"追蹤",9.8)
    target=np.deg2rad(70)
    for i,(j,title) in enumerate([(0,"(a) 設定點已固定"),(5,"(b) 關節持續追蹤"),(25,"(c) 逐漸接近設定點")]):
        ax=plain(fig,[.035+i*.323,.18,.285,.42],(-.45,1.30),(-.25,1.45),True)
        text(ax,.5,1.07,title,10.8,transform=ax.transAxes)
        pivot=np.array([0.,0.]);q=target+(np.deg2rad(10)-target)*(.88**j)
        s_end=1.20*np.array([np.cos(target),np.sin(target)])
        q_end=1.20*np.array([np.cos(q),np.sin(q)])
        ax.add_patch(Rectangle((-.24,-.14),.48,.27,ec=INK,fc=LIGHT,lw=.9,zorder=2))
        line(ax,[pivot,s_end],color=GRAY,ls="--",lw=1.7,zorder=5)
        line(ax,[pivot,q_end],lw=4.2,zorder=6)
        ax.add_patch(Circle(pivot,.075,ec=INK,fc="white",lw=1.2,zorder=8))
        ax.add_patch(Circle(q_end,.035,ec=INK,fc="white",lw=.9,zorder=8))
        if i<2:
            a1=q+.06;a2=min(target-.08,q+.50);r=.47
            ax.add_patch(Arc(pivot,2*r,2*r,theta1=np.rad2deg(a1),theta2=np.rad2deg(a2),lw=1))
            arrow(ax,r*np.array([np.cos(a2-.10),np.sin(a2-.10)]),
                  r*np.array([np.cos(a2),np.sin(a2)]),lw=1,scale=7)
    export(fig,"arm_setpoint_tracking","手臂設定點與實際關節位置之追蹤關係",
           "3.2.2 增廣狀態與末端位姿說明之後","設定點固定後，關節仍可能持續追蹤。")


def dynamic_prediction():
    fig=plt.figure(figsize=(7.2,6.0))
    start=np.array([.4,0.]);old_obs=np.array([1.6,1.40]);future_obs=np.array([1.6,0.])
    candidate=[None,np.array([1.6,0.]),np.array([.78,0.]),np.array([1.6,-.72])]
    titles=["(a) 現在：觀測位置與速度","(b) 未來：維持原速度",
            "(c) 同一未來時刻：減速","(d) 同一未來時刻：側移"]
    notes=["依位置與速度預測行人運動","預測位置重疊","改變抵達交會區的時間","改變通行位置"]
    for i in range(4):
        col=i%2;row=i//2
        ax=plain(fig,[.025+col*.50,.56-row*.47,.45,.33],(-.15,2.55),(-1.05,1.85),True)
        fig.text(.25+col*.50,.95-row*.47,titles[i],fontproperties=FONT,fontsize=10.8,ha="center")
        if i==0:
            robot_plan(ax,start);person(ax,old_obs)
            arrow(ax,(.80,0),(1.30,0));arrow(ax,(1.6,1.04),(1.6,.55))
            text(ax,.42,-.47,"機器人",10)
            text(ax,2.16,1.40,"行人",10)
        else:
            ax.plot(*start,marker="o",mfc="white",mec=INK,ms=3.2,zorder=15)
            text(ax,start[0],-.40,"目前位置",9.6)
            p=candidate[i]
            line(ax,[old_obs,future_obs],ls=":",lw=.9,color=GRAY,zorder=3)
            arrow(ax,old_obs+[0,-.25],future_obs+[0,.30],ls=":",lw=.8,scale=7)
            if i==3:
                path=np.array([start,[1.0,-.36],p]);line(ax,path,lw=1.1)
            else: line(ax,[start,p],lw=1.1)
            robot_plan(ax,p)
            person(ax,future_obs)
            if i==1:
                text(ax,2.23,.36,"重疊",10.5)
                line(ax,[(2.03,.25),(1.83,.10)],lw=.8)
            else:
                # Bracket the surface-to-surface gap at the same future time.
                d=future_obs-p;d/=np.linalg.norm(d)
                a=p+.30*d;b=future_obs-.25*d
                ax.add_patch(FancyArrowPatch(a,b,arrowstyle="<->",mutation_scale=4,
                                             color=INK,lw=.9,zorder=15))
                if i==2:text(ax,1.20,.40,"間距",9.8)
                else:text(ax,2.17,-.39,"間距",9.8)
    export(fig,"dynamic_obstacle_prediction","同一未來時刻下原速度、減速與側移候選之位置比較",
           "3.4.2 最小預測間距說明之後","區分改變抵達時間與改變通行位置；間距比較須時間對齊。")


def visual_reference():
    fig=plt.figure(figsize=(7.2,4.3))
    goal=np.array([.47,0,.62]);chosen=(0.,0.,0.)
    q=solve_pose(chosen,goal,Rotation.from_euler("x",np.pi).as_matrix(),"visual_selected_base")
    folded=np.array([0.,-.55,1.7,0.,.5,0.])
    scenes=[]
    for base,qa in [((-.65,-.15,0),folded),(chosen,q)]:
        s=Scene(azimuth=-65,elevation=24);s.table();s.robot(base,qa);scenes.append(s)
    points=np.concatenate([s.p(p) for s in scenes for p in s.polygons])
    lo,hi=points.min(axis=0)-.13,points.max(axis=0)+.13
    for i,(base,qa,title) in enumerate([((-.65,-.15,0),folded,"(a) 選位前：目標距離較遠"),
                                        (chosen,q,"(b) 選位後：手臂接近目標")]):
        ax=fig.add_axes([.025+i*.50,.19,.45,.65]);s=scenes[i];s.draw(ax,margin=.11)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
        fig.text(.25+i*.50,.90,title,fontproperties=FONT,fontsize=10.8,ha="center")
        if i==0:
            callout(ax,s,goal,"操作目標",(.84,.84))
            callout(ax,s,[base[0],base[1],.28],"目前底盤位置",(.25,.08))
        else:
            callout(ax,s,goal,"夾爪接近目標",(.80,.86))
            callout(ax,s,[0,0,.28],"選定底盤位姿",(.25,.08))
    export(fig,"visual_operation_reference","操作目標導引之底盤選位與手臂接近",
           "3.5.3 候選集合與底盤目標說明之後","選擇底盤操作位姿，讓手臂能接近已定位目標。")


def articulated_constraints():
    fig=plt.figure(figsize=(7.2,4.8))
    ax=fig.add_axes([.025,.17,.45,.70]);s=Scene(azimuth=-25,elevation=25)
    phi=np.pi/4;pose=np.eye(4);pose[:3,:3]=Rotation.from_euler("z",phi).as_matrix()
    s.box((0,-.014,0),(.80,.014,1.05),gray=.89,pose=pose)
    s.cylinder((0,0,.525),.016,1.05,gray=.4)
    h=np.array([.68*np.cos(phi),.68*np.sin(phi),.66])
    normal=np.array([np.sin(phi),-np.cos(phi),0.]);base=h[:2]+.40*normal[:2]
    z=-normal;x=np.array([0,0,1.]);y=np.cross(z,x);r=np.stack([x,y,z],axis=1)
    q=solve_pose([*base,phi+np.pi/2],h+.015*normal,r,"revolute_door_grasp")
    s.cylinder(h,.013,.11,axis=(np.cos(phi),np.sin(phi),0),gray=.28)
    s.robot([*base,phi+np.pi/2],q);s.draw(ax,margin=.09)
    text(ax,.5,1.08,"(a) 開門：門扇繞鉸鏈轉動",10.8,transform=ax.transAxes)
    callout(ax,s,[0,0,.63],"鉸鏈",(.12,.69))
    callout(ax,s,h,"抓持把手",(.82,.81))
    angles=np.linspace(.03,phi-.02,30)
    arc=np.c_[.80*np.cos(angles),.80*np.sin(angles),np.ones(len(angles))*1.05]
    pp=s.p(arc);line(ax,pp,color=GRAY,ls=":",lw=1)
    arrow(ax,pp[-4],pp[-1],scale=8,lw=1)
    callout(ax,s,arc[10],"轉動方向",(.26,.88))

    ax=fig.add_axes([.525,.17,.45,.70]);s=Scene(azimuth=-60,elevation=23)
    # An open cabinet shell makes the translating drawer visibly distinct.
    s.box((-.40,0,0),(-.36,.48,.84),gray=.88)
    s.box((.36,0,0),(.40,.48,.84),gray=.88)
    s.box((-.40,.44,0),(.40,.48,.84),gray=.88)
    s.box((-.40,0,.81),(.40,.48,.85),gray=.88)
    s.box((-.36,0,0),(.36,.44,.08),gray=.85)
    # Drawer front and side walls displaced 0.28 m out of the cabinet.
    s.box((-.35,-.28,.37),(.35,-.25,.68),gray=.88)
    s.box((-.35,-.25,.37),(-.32,.27,.65),gray=.94)
    s.box((.32,-.25,.37),(.35,.27,.65),gray=.94)
    s.box((-.35,-.25,.37),(.35,.27,.39),gray=.90)
    h=np.array([0,-.32,.55]);s.cylinder(h,.013,.15,axis=(1,0,0),gray=.28)
    base=(0,-.73,np.pi/2)
    q=solve_pose(base,h+[0,-.015,0],Rotation.from_euler("x",-np.pi/2).as_matrix(),"prismatic_drawer_grasp")
    s.robot(base,q);s.draw(ax,margin=.09)
    text(ax,.5,1.08,"(b) 拉抽屜：沿滑軌平移",10.8,transform=ax.transAxes)
    callout(ax,s,h,"抓持把手",(.14,.67))
    a,b=s.p([[.48,.0,.55],[.48,-.28,.55]])
    arrow(ax,a,b,lw=1.25,scale=9)
    text(ax,.88,.56,"拉出方向",10,transform=ax.transAxes)
    export(fig,"articulated_operation_constraints","移動機械臂開門與抽屜操作之機構運動",
           "3.6.2 門與抽屜機構模型之後","門沿圓弧轉動，抽屜沿滑軌平移，夾爪在協同運動中維持抓持。")


def execution_timeline():
    fig=plt.figure(figsize=(7.2,4.25))
    ax=plain(fig,[.025,.04,.95,.92],(0,12),(0,6.4))
    a,b,end=2.65,7.10,11.60
    arrow(ax,(a,5.80),(end,5.80),lw=1)
    text(ax,a,6.16,"取得目前狀態",10.5)
    text(ax,b,6.16,"新命令預定生效",10.5)
    text(ax,11.3,6.15,"時間",10)
    for x in (a,b):line(ax,[(x,.55),(x,5.72)],ls="--",lw=.75,color=GRAY,zorder=1)
    text(ax,.85,4.71,"計算端",11)
    node(ax,(a+b-.25)/2,4.71,b-a-.25,.88,"預測、求解與核對",size=10.5,fill=LIGHT)
    text(ax,9.45,4.71,"結果須於期限內到達",10.3)
    text(ax,.85,3.32,"接受時",11)
    text(ax,.85,1.92,"拒絕或逾時",10.6)
    for y in (3.32,1.92):
        node(ax,(a+b)/2,y,b-a,.94,"既定命令持續執行",size=10.6,fill=LIGHT)
    node(ax,(b+end)/2,3.32,end-b,.94,"接續新控制段",size=10.8)
    r=node(ax,(b+end)/2,1.92,end-b,.94,"沿已核對停止尾段減速",size=10.6)
    r.set_hatch("///")
    for artist in list(ax.texts):
        if artist.get_text()=="沿已核對停止尾段減速":artist.set_bbox(dict(fc="white",ec="none",pad=2))
    export(fig,"command_execution_timeline","控制計算期間之命令接續與停止備援",
           "3.7.1 在途命令前推與延遲補償初始狀態之後","計算期間機器人持續執行舊命令，只有接受的新結果才接續執行。")


def overall_workflow():
    fig=plt.figure(figsize=(7.2,6.15))
    ax=plain(fig,[.025,.025,.95,.95],(0,11),(0,12.2))
    x=3.85;w=5.55;h=1.05
    node(ax,x,11.25,w,h,"1　更新狀態與感知","位姿、關節、設定點與有效觀測")
    node(ax,x,9.35,w,h,"2　建立預測狀態與參考","補償命令延遲；更新導航及操作目標")
    node(ax,x,7.45,w,h,"3　全身 GMPC 求解","共同計算底盤與手臂命令")
    for y1,y2 in [(10.725,9.875),(8.825,7.975),(6.925,6.18)]:arrow(ax,(x,y1),(x,y2))
    diamond=Polygon([(x,6.18),(x+2.15,5.35),(x,4.52),(x-2.15,5.35)],
                     ec=INK,fc="white",lw=1.05,zorder=5)
    ax.add_patch(diamond);text(ax,x,5.35,"命令通過限制檢查？",10.7)
    arrow(ax,(x,4.52),(x,3.63));text(ax,x-.28,4.08,"是",10,ha="right")
    node(ax,x,3.10,w,h,"4　執行控制命令","底盤命令防護；手臂設定點更新")
    stopx=8.80
    arrow(ax,(x+2.15,5.35),(stopx,5.35));text(ax,7.18,5.70,"否",10.1)
    line(ax,[(stopx,5.35),(stopx,3.95)],lw=1.15)
    arrow(ax,(stopx,3.95),(stopx,3.63))
    node(ax,stopx,3.10,3.15,h,"減速與停止","依實際速度判定",fill=LIGHT,size=10.7)
    node(ax,x,1.15,w,h,"5　執行回授","回報最終命令、設定點與新狀態")
    arrow(ax,(x,2.575),(x,1.675))
    line(ax,[(stopx,2.575),(stopx,1.15),(x+w/2,1.15)],lw=1.15)
    arrow(ax,(x+w/2+.4,1.15),(x+w/2,1.15))
    line(ax,[(x-w/2,1.15),(.40,1.15),(.40,11.25)],ls="--",lw=1)
    arrow(ax,(.40,11.25),(x-w/2,11.25),ls="--",lw=1)
    text(ax,.66,6.10,"下一控制週期",10,rotation=90)
    export(fig,"wholebody_control_workflow","全身預測控制之計算、命令核對與執行回授流程",
           "3.7.4 第一段之後、控制步驟列表之前","從有效觀測到共同求解，再經限制檢查與實際執行回授形成閉迴路。")


def contact_sheet(items=None,name="chapter3_figures_overview.png",columns=2):
    items=MANIFEST if items is None else items
    rows=int(np.ceil(len(items)/columns))
    fig,axes=plt.subplots(rows,columns,figsize=(6*columns,4.1*rows),squeeze=False)
    fig.subplots_adjust(left=.02,right=.98,top=.97,bottom=.015,wspace=.065,hspace=.13)
    for ax in axes.flat:ax.set_axis_off()
    for ax,item in zip(axes.flat,items):
        ax.imshow(plt.imread(OUT/f"{item['file']}.png"))
        ax.set_title(item["placement"].split()[0],fontproperties=FONT,fontsize=15,pad=6)
    fig.savefig(OUT/name,dpi=130);plt.close(fig)


def main():
    import sys
    from supplementary_method_diagrams import generate
    setpoint_tracking();dynamic_prediction();visual_reference()
    articulated_constraints();execution_timeline();overall_workflow()
    generate(sys.modules[__name__])
    order=["base_pose_error","wholebody_motion_coupling","arm_setpoint_tracking",
           "joint_optimization","dynamic_obstacle_prediction","navigation_command_layers",
           "visual_time_alignment","visual_operation_reference","grasp_state_transition",
           "door_operation_sequence","drawer_operation_sequence","base_relative_carrying",
           "command_execution_timeline","model_bias_estimation","tcp_target_compensation",
           "wholebody_control_workflow"]
    for item in MANIFEST:item["recommended"]=item["file"] in order
    MANIFEST.sort(key=lambda item:order.index(item["file"]) if item["file"] in order else len(order))
    selected=[item for item in MANIFEST if item["recommended"]]
    contact_sheet(selected,columns=3)
    for part,start in enumerate(range(0,len(selected),6),1):
        contact_sheet(selected[start:start+6],name=f"chapter3_figures_overview_{part}.png")
    (OUT/"figure_manifest.json").write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2)+"\n")
    (OUT/"scene_geometry_checks.json").write_text(json.dumps(dict(
        source="evaluation/models/omni_bot_wholebody_expanded.urdf and its STL visuals",
        mesh_visualisation="Vertex-clustered for vector illustration only.",
        ik=IK_RECORDS,
        limits="Pose IK only; no complete collision, grasp force, or trajectory validation."),ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":main()
