"""Additional, single-purpose method illustrations for chapter 3."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from scipy.spatial.transform import Rotation
from method_scene_geometry import Scene, solve_pose, frames


def generate(d):
    """Use the existing diagram module's font, export and primitive helpers."""
    _base_error(d);_wholebody_coupling(d);_joint_qp(d);_avoidance_layers(d)
    _visual_time(d);_grasp_sequence(d);_door_sequence(d);_drawer_sequence(d)
    _carrying(d);_model_bias(d);_tcp_offset(d)


def _paired_scenes(d, scenes, titles, labels, footer, name, caption, placement, purpose):
    fig=plt.figure(figsize=(7.2,4.1))
    points=np.concatenate([s.p(p) for s in scenes for p in s.polygons])
    lo,hi=points.min(axis=0)-.13,points.max(axis=0)+.13
    for i,s in enumerate(scenes):
        ax=fig.add_axes([.025+i*.50,.08,.45,.79]);s.draw(ax)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
        fig.text(.25+i*.50,.94,titles[i],fontproperties=d.FONT,fontsize=10.8,ha="center")
        for point,label,position in labels[i]:d.callout(ax,s,point,label,position)
    d.export(fig,name,caption,placement,purpose)


def _base_error(d):
    fig=plt.figure(figsize=(7.2,3.65))
    pr=np.array([.75,.65]);pb=np.array([2.60,1.75]);tr=np.deg2rad(35);tb=np.deg2rad(65)
    rr=Rotation.from_euler("z",tr).as_matrix()[:2,:2]
    relative=rr.T@(pb-pr)
    def axes(ax,p,theta,suffix,length=.60):
        ex=np.array([np.cos(theta),np.sin(theta)]);ey=np.array([-ex[1],ex[0]])
        d.arrow(ax,p,p+length*ex,scale=7,lw=.9);d.arrow(ax,p,p+length*ey,scale=7,lw=.9)
        d.text(ax,*(p+(length+.14)*ex),"$x_"+suffix+"$",10.3)
        d.text(ax,*(p+(length+.14)*ey),"$y_"+suffix+"$",10.3)
    a=d.plain(fig,[.025,.14,.45,.70],(-.15,3.65),(-.10,2.90),True)
    b=d.plain(fig,[.525,.14,.45,.70],(-.50,2.95),(-1.00,1.65),True)
    for ax,title in [(a,"(a) 世界座標中的兩個底盤位姿"),(b,"(b) 以參考底盤為座標基準")]:
        d.text(ax,.5,1.12,title,10.6,transform=ax.transAxes)
    d.robot_plan(a,pr,yaw=tr,ghost=True);d.robot_plan(a,pb,yaw=tb)
    axes(a,pr,tr,"r");axes(a,pb,tb,"B")
    d.line(a,[pr,pb],ls=":",color=d.GRAY)
    d.text(a,.60,.08,"參考位姿 $X_r$",10.1)
    d.text(a,2.75,2.65,"實際位姿 $X_B$",10.1)
    d.robot_plan(b,[0,0],yaw=0,ghost=True);d.robot_plan(b,relative,yaw=tb-tr)
    axes(b,np.array([0.,0.]),0,"r");axes(b,relative,tb-tr,"B")
    d.line(b,[relative,relative+[.67,0]],ls=":",lw=.75,color=d.GRAY)
    b.add_patch(Arc(relative,.92,.92,theta1=0,theta2=np.rad2deg(tb-tr),lw=.9))
    d.arrow(b,[.10,-.12],relative+[0,-.12],ls=":",lw=.9)
    d.text(b,1.15,-.60,"相對平移 $t_E$",10.1)
    d.text(b,2.08,1.18,"相對航向 $\\varphi$",10.1)
    d.export(fig,"base_pose_error","底盤參考位姿、實際位姿與相對幾何關係",
             "3.1.2 幾何追蹤誤差與局部模型說明之後","區分世界座標位姿、相對平移與李代數誤差，圖中平移箭頭不當作 rho。")


def _wholebody_coupling(d):
    scenes=[];labels=[];goal=np.array([.47,0,.62]);r=Rotation.from_euler("x",np.pi).as_matrix()
    endpoints=[]
    for i,y in enumerate([.10,-.10]):
        base=(0.,y,0.);q=solve_pose(base,goal,r,f"wholebody_compensation_{i}")
        s=Scene(-65,24);s.table();end=s.robot(base,q);scenes.append(s);endpoints.append(end)
        labels.append([(goal,"同一末端目標",(.81,.84)),([0,y,.28],"底盤位置",(.23,.08))])
    np.testing.assert_allclose(endpoints[0],endpoints[1],atol=1e-4)
    _paired_scenes(d,scenes,["(a) 底盤移動前","(b) 底盤側移，手臂配合調整"],labels,
        "底盤位姿改變時，可由手臂構型調整維持同一世界座標目標。",
        "wholebody_motion_coupling","底盤側移時之手臂構型調整與末端位置保持",
        "3.1.3 正向運動學公式及其說明之後","同一末端目標可由不同底盤與手臂構型共同達成。")


def _joint_qp(d):
    fig=plt.figure(figsize=(7.2,4.35));ax=d.plain(fig,[.025,.04,.95,.92],(0,12),(0,6.5))
    d.node(ax,1.95,4.65,3.35,1.05,"導航參考","底盤位姿與速度")
    d.node(ax,1.95,2.65,3.35,1.05,"操作目標","末端位置與姿態")
    d.node(ax,6.05,3.65,3.60,1.40,"同一全身 QP","共同求解控制序列")
    d.node(ax,6.05,5.65,3.90,.95,"共用增廣預測模型",size=10.6)
    d.node(ax,6.05,1.35,3.90,1.05,"共同限制條件","命令、關節與輪級限制")
    d.node(ax,10.20,4.65,3.35,1.05,"底盤速度命令","$\\xi_r+\\delta\\xi$")
    d.node(ax,10.20,2.65,3.35,1.05,"手臂設定點變化率","$u_a$")
    for y1,y2 in [(4.65,3.98),(2.65,3.32)]:
        d.line(ax,[(3.625,y1),(3.95,y1),(3.95,y2)])
        d.arrow(ax,(3.95,y2),(4.25,y2),scale=8)
        d.line(ax,[(7.85,y2),(8.10,y2),(8.10,y1)])
        d.arrow(ax,(8.10,y1),(8.525,y1),scale=8)
    d.arrow(ax,(6.05,5.175),(6.05,4.35))
    d.arrow(ax,(6.05,1.875),(6.05,2.95))
    d.export(fig,"joint_optimization","底盤與手臂於同一全身控制問題中共同求解",
             "3.3.3 最佳化問題及數值求解說明之後","底盤與手臂共享預測、末端目標與限制，在一個 QP 內共同決策。")


def _avoidance_layers(d):
    fig=plt.figure(figsize=(7.2,5.65));ax=d.plain(fig,[.025,.025,.95,.95],(0,11),(0,11.5))
    ys=[10.45,8.55,6.85,5.15,3.45,1.75];x=3.55;w=4.75
    entries=[("調整導航參考","比較速度與側移候選"),("全身 GMPC","CBF 避障條件納入同一 QP"),
             ("底盤速度平滑",None),("原始雷射防護",None),("最終輪級限制",None),("底盤執行",None)]
    for y,(title,detail) in zip(ys,entries):d.node(ax,x,y,w,1.05,title,detail)
    for first,second in zip(ys,ys[1:]):d.arrow(ax,(x,first-.525),(x,second+.525))
    d.node(ax,8.65,5.15,3.15,1.05,"最新原始掃描",size=10.7)
    d.arrow(ax,(7.075,5.15),(5.925,5.15))
    d.line(ax,[(x,1.225),(x,.55),(6.55,.55),(6.55,4.85)],ls="--",lw=.9)
    d.arrow(ax,(6.55,4.85),(5.925,4.85),ls="--",lw=.9,scale=8)
    d.text(ax,8.45,1.95,"最終放行命令回報",10.0)
    d.text(ax,8.45,1.32,"供後續限制及預測使用",9.6)
    d.export(fig,"navigation_command_layers","導航參考調整、預測避障與底盤命令處理之分層關係",
             "3.4.5 原始掃描防護與命令處理說明之後","參考調整、QP 內 CBF 與執行端雷射防護分別作用於不同位置。")


def _visual_time(d):
    fig=plt.figure(figsize=(7.2,4.1));ax=d.plain(fig,[.025,.04,.95,.92],(0,12),(0,6.0))
    d.arrow(ax,(1.0,5.20),(11.5,5.20),lw=1)
    for x,label in [(2.25,"影像取得 $t_o$"),(6.8,"處理完成 $t_k$"),(10.2,"後續控制")]:
        d.line(ax,[(x,5.08),(x,5.33)],lw=.9);d.text(ax,x,5.68,label,10.4)
    d.node(ax,2.25,3.70,3.85,1.05,"影像中的目標量測","保留原觀測時間 $t_o$")
    d.node(ax,2.25,1.65,3.85,1.05,"觀測時刻的相機位姿","$ {}^WT_C(t_o)$")
    d.node(ax,6.80,2.70,3.40,1.20,"轉換至世界座標","使用同一觀測時間")
    d.node(ax,10.30,2.70,2.85,1.20,"操作參考生成",size=10.5)
    for y in [3.70,1.65]:
        d.line(ax,[(4.175,y),(4.65,y),(4.65,2.70)])
    d.arrow(ax,(4.65,2.70),(5.10,2.70));d.arrow(ax,(8.50,2.70),(8.875,2.70))
    d.export(fig,"visual_time_alignment","視覺量測與相機位姿依觀測時間配對之流程",
             "3.5.2 影像時間對齊說明之後","影像時間與處理完成時間不同，座標轉換以觀測時刻相機位姿為準。")


def _grasp_sequence(d):
    fig=plt.figure(figsize=(7.2,3.2))
    for i,(gap,title,note) in enumerate([(.48,"(a) 接近目標","夾爪保持張開"),
                                          (.27,"(b) 閉爪確認","檢查手指位置與接觸"),
                                          (.27,"(c) 抓持成立","持續達標且確認連接")]):
        ax=d.plain(fig,[.025+i*.333,.10,.285,.76],(-.75,.75),(-.70,1.10),True)
        d.text(ax,.5,1.08,title,10.8,transform=ax.transAxes)
        ax.add_patch(Rectangle((-.55,.58),1.10,.18,ec=d.INK,fc=d.LIGHT,lw=1))
        ax.add_patch(Rectangle((-gap-.08,-.08),.08,.66,ec=d.INK,fc=d.LIGHT,lw=1))
        ax.add_patch(Rectangle((gap,-.08),.08,.66,ec=d.INK,fc=d.LIGHT,lw=1))
        ax.add_patch(Circle((0,0),.27,ec=d.INK,fc="#d3d3d3",lw=1.1,zorder=4))
        if i>0:
            for x in (-.27,.27):ax.plot(x,0,"o",color=d.INK,ms=3.2,zorder=8)
        if i==2:
            d.line(ax,[(-gap,.42),(gap,.42)],lw=1.3)
    ax=d.plain(fig,[.025,.10,.95,.76],(0,12),(0,2))
    d.arrow(ax,(3.45,1.08),(4.05,1.08),scale=9)
    d.arrow(ax,(7.65,1.08),(8.25,1.08),scale=9)
    d.export(fig,"grasp_state_transition","接近、閉爪確認與抓持成立之狀態轉移",
             "3.6.1 抓持證據與連接確認說明之後","閉爪命令、接觸證據與抓持成立是不同階段。")


def _door_scene(angle,index):
    s=Scene(-25,25);rdoor=Rotation.from_euler("z",angle).as_matrix();pose=np.eye(4);pose[:3,:3]=rdoor
    s.box((0,-.014,0),(.80,.014,1.05),gray=.89,pose=pose);s.cylinder((0,0,.525),.016,1.05,gray=.4)
    h=rdoor@np.array([.68,0,.66]);normal=rdoor@np.array([0.,-1,0]);base=h[:2]+.40*normal[:2]
    z=-normal;x=np.array([0.,0,1]);y=np.cross(z,x);rot=np.stack([x,y,z],axis=1)
    q=solve_pose([*base,angle+np.pi/2],h+.015*normal,rot,f"door_sequence_{index}")
    s.cylinder(h,.013,.11,axis=rdoor[:,0],gray=.28);s.robot([*base,angle+np.pi/2],q)
    return s,h,base


def _door_sequence(d):
    a,ha,ba=_door_scene(np.deg2rad(10),0);b,hb,bb=_door_scene(np.deg2rad(50),1)
    _paired_scenes(d,[a,b],["(a) 抓持初始門角","(b) 門扇開啟後"],
        [[([0,0,.68],"鉸鏈",(.13,.72)),(ha,"抓持把手",(.81,.82))],
         [(hb,"維持抓持",(.82,.82)),([*bb,.28],"底盤配合移動",(.26,.08))]],
        "把手繞鉸鏈運動；底盤與手臂共同維持抓持後的相對位姿。",
        "door_operation_sequence","開門前後之門扇、底盤與手臂構型",
        "3.6.2 旋轉門模型與門角更新說明之後","把手軌跡、底盤位置及手臂構型隨門角共同改變。")


def _drawer_scene(displacement,index):
    s=Scene(-60,23)
    for lo,hi in [((-.40,0,0),(-.36,.48,.84)),((.36,0,0),(.40,.48,.84)),
                  ((-.40,.44,0),(.40,.48,.84)),((-.40,0,.81),(.40,.48,.85)),
                  ((-.36,0,0),(.36,.44,.08))]:s.box(lo,hi,gray=.88)
    y=-displacement
    for lo,hi in [((-.35,y,.37),(.35,y+.03,.68)),((-.35,y+.03,.37),(-.32,y+.55,.65)),
                  ((.32,y+.03,.37),(.35,y+.55,.65)),((-.35,y+.03,.37),(.35,y+.55,.39))]:s.box(lo,hi,gray=.90)
    h=np.array([0,y-.04,.55]);base=(0,y-.45,np.pi/2)
    q=solve_pose(base,h+[0,-.015,0],Rotation.from_euler("x",-np.pi/2).as_matrix(),f"drawer_sequence_{index}")
    s.cylinder(h,.013,.15,axis=(1,0,0),gray=.28);s.robot(base,q)
    return s,h,base


def _drawer_sequence(d):
    a,ha,ba=_drawer_scene(0.,0);b,hb,bb=_drawer_scene(.28,1)
    _paired_scenes(d,[a,b],["(a) 抽屜初始位置","(b) 沿滑軌拉出後"],
        [[(ha,"抓持把手",(.13,.70))],[(hb,"把手沿軸向移動",(.14,.73)),([*bb[:2],.28],"底盤配合移動",(.24,.08))]],
        "抽屜保持姿態並沿滑軌平移，底盤與手臂配合操作進度。",
        "drawer_operation_sequence","抽屜拉出前後之滑軌位移與全身構型",
        "3.6.2 平移抽屜模型與有效行程說明之後","抽屜平移與門扇旋轉以獨立圖片說明，避免混用幾何關係。")


def _carrying(d):
    goal=np.array([.42,0,.64]);q=solve_pose((0,0,0),goal,Rotation.from_euler("x",np.pi).as_matrix(),"carry_relative_pose")
    scenes=[];labels=[];rel=[]
    for x in [-.28,.28]:
        base=(x,0.,0.);s=Scene(-65,24);p=s.robot(base,q)
        s.cylinder(p,.035,.14,gray=.42,n=24)
        s.stroke([[0,-.40,0],[0,-.28,0]],color=d.GRAY,lw=.8)
        s.stroke([[-.06,-.34,0],[.06,-.34,0]],color=d.GRAY,lw=.8)
        scenes.append(s);rel.append(p-np.array([x,0,0]))
        labels.append([(p,"隨底盤運送的物件",(.79,.82)),([x,0,.28],"底盤位置",(.23,.08))])
    np.testing.assert_allclose(rel[0],rel[1],atol=1e-10)
    _paired_scenes(d,scenes,["(a) 持物運送前","(b) 底盤平移後"],labels,
        "相對底盤的末端目標保持固定；物件在世界座標中的位置隨底盤改變。",
        "base_relative_carrying","相對底盤目標下之持物運送與構型保持",
        "3.6.3 持物相對目標與運送流程說明之後","區分世界固定目標與相對底盤固定目標。")


def _model_bias(d):
    fig=plt.figure(figsize=(7.2,4.45));ax=d.plain(fig,[.025,.04,.95,.92],(0,12),(0,6.3))
    d.node(ax,2.10,5.05,3.65,1.05,"對齊後的區間資料","起點位置、生效設定點")
    d.node(ax,7.75,5.05,4.30,1.05,"使用舊偏置進行預測","得到區間末預測位置")
    d.node(ax,2.10,3.10,3.65,1.05,"區間末實際關節位置",size=10.3)
    d.node(ax,7.75,3.10,4.30,1.05,"計算區間終點殘差","實際位置減去預測位置")
    d.node(ax,7.75,1.15,4.30,1.05,"平滑與限幅更新偏置","用於後續預測")
    d.node(ax,2.10,1.15,3.65,1.05,"更新共用預測模型","控制與執行共用")
    d.arrow(ax,(3.925,5.05),(5.60,5.05));d.arrow(ax,(3.925,3.10),(5.60,3.10))
    d.arrow(ax,(7.75,4.525),(7.75,3.625));d.arrow(ax,(7.75,2.575),(7.75,1.675))
    d.arrow(ax,(5.60,1.15),(3.925,1.15))
    d.line(ax,[(9.90,1.15),(11.10,1.15),(11.10,5.05)],ls="--",lw=.9)
    d.arrow(ax,(11.10,5.05),(9.90,5.05),ls="--",lw=.9)
    d.text(ax,11.43,3.10,"後續區間",9.8,rotation=90)
    d.export(fig,"model_bias_estimation","模型偏置估測之區間殘差與共用模型更新",
             "3.7.2 模型偏置估測說明之後、TCP 補償介紹之前","殘差先以舊模型計算，偏置更新作用於後續預測。")


def _tcp_offset(d):
    fig=plt.figure(figsize=(7.2,3.25));ax=d.plain(fig,[.025,.06,.95,.90],(0,12),(0,4.4))
    d.line(ax,[(.6,2.2),(11.4,2.2)],color=d.GRAY,lw=.9)
    points=[(1.6,"實際末端位置","$p_E$","o"),(6.0,"原始幾何目標","$p_E^d$","+"),(9.8,"命令目標","$p_E^{d,\\mathrm{cmd}}$","^")]
    for x,title,symbol,marker in points:
        ax.plot(x,2.2,marker=marker,ms=9,mfc="white",mec=d.INK,mew=1.4,zorder=10)
        d.text(ax,x,1.55,title,10.3);d.text(ax,x,1.05,symbol,11.6)
        d.line(ax,[(x,2.40),(x,3.25)],ls=":",color=d.GRAY,lw=.7)
    d.arrow(ax,(1.6,3.35),(6.0,3.35));d.text(ax,3.80,3.85,"追蹤誤差 $e_p$",10.5)
    d.arrow(ax,(6.0,3.35),(9.8,3.35));d.text(ax,7.90,3.85,"有界修正 $\\iota$",10.5)
    d.export(fig,"tcp_target_compensation","末端原始幾何目標與有界命令目標補償",
             "3.7.2 有界 TCP 目標補償公式與說明之後","以單一位置分量呈現 p_cmd=p_d+iota；補償改目標，不改原始完成判準。")
