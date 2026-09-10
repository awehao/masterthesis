#!/usr/bin/env python3
"""CPU orthographic illustration of actual URDF visual meshes (no simulator)."""
from pathlib import Path
import sys, json, math
import numpy as np
import xml.etree.ElementTree as ET
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics, rpy_to_rot


def primitive(geom):
    mesh=geom.find('mesh')
    if mesh is not None:
        path=mesh.get('filename').replace('file://','')
        if path.startswith('package://'):
            pkg,rest=path[10:].split('/',1)
            path=str(ROOT/'install'/pkg/'share'/pkg/rest)
        reader=vtk.vtkSTLReader();reader.SetFileName(path);reader.Update()
        poly=reader.GetOutput()
        if poly.GetNumberOfCells()>18000:
            dec=vtk.vtkQuadricDecimation();dec.SetInputData(poly)
            dec.SetTargetReduction(1-18000/poly.GetNumberOfCells());dec.Update()
            poly=dec.GetOutput()
        v=vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
        v*=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        faces=vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1,4)[:,1:]
        return v[faces]
    b=geom.find('box')
    if b is not None:
        d=np.fromstring(b.get('size'),sep=' ')/2
        v=np.array([[x,y,z] for x in [-d[0],d[0]] for y in [-d[1],d[1]] for z in [-d[2],d[2]]])
        faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
        return v[faces]
    c=geom.find('cylinder')
    if c is not None:
        r,h=float(c.get('radius')),float(c.get('length'))/2
        tri=[]
        for i in range(48):
            a,b=2*np.pi*i/48,2*np.pi*(i+1)/48
            p=np.array([r*np.cos(a),r*np.sin(a),-h]);q=np.array([r*np.cos(b),r*np.sin(b),-h])
            p1=p+[0,0,2*h];q1=q+[0,0,2*h]
            tri += [[p,q,p1],[q,q1,p1],[[0,0,-h],q,p],[[0,0,h],p1,q1]]
        return np.asarray(tri)
    return np.zeros((0,3,3))


def render(urdf,out,q=None,obstacles=False):
    tree=ET.parse(urdf).getroot()
    K=WholeBodyKinematics.from_urdf_file(str(urdf))
    if q is None:q=np.array([0,0,0,0,.3641,.5263,0,-1.4086,0])
    tris=[];cols=[]
    for link in tree.findall('link'):
        name=link.get('name')
        T=K.fk(q,name)
        for visual in link.findall('visual'):
            p=primitive(visual.find('geometry'))
            if not len(p):continue
            o=visual.find('origin')
            xyz=np.fromstring(o.get('xyz','0 0 0') if o is not None else '0 0 0',sep=' ')
            rpy=np.fromstring(o.get('rpy','0 0 0') if o is not None else '0 0 0',sep=' ')
            p=p@rpy_to_rot(*rpy).T+xyz
            p=p@T[:3,:3].T+T[:3,3]
            if name=='base_link':color=[.27,.39,.48]
            elif name.startswith(('roller','rim')):color=[.22,.25,.28]
            elif name=='link_eef' or 'gripper' in name or 'finger' in name:color=[.37,.4,.42]
            else:color=[.79,.82,.84]
            tris.append(p);cols.append(np.tile(color,(len(p),1)))
    if obstacles:
        for ctr,sz in [((.74,0,.60),(.4,1.,1.2)),((-.225,-.32,.30),(.16,.20,.6))]:
            g=ET.fromstring('<geometry><box size="'+' '.join(map(str,sz))+'"/></geometry>')
            p=primitive(g)+np.array(ctr)
            tris.append(p);cols.append(np.tile([.72,.59,.45],(len(p),1)))
    triangles=np.concatenate(tris);colors=np.concatenate(cols)
    eye=np.array([-1.7,-2.6,1.55] if obstacles else [1.7,-2.6,1.55]);view=eye/np.linalg.norm(eye)
    right=np.cross([0,0,1],view);right/=np.linalg.norm(right)
    up=np.cross(view,right)
    basis=np.array([right,-up,view])
    projected=triangles@basis.T
    lo=projected[:,:,:2].reshape(-1,2).min(0);hi=projected[:,:,:2].reshape(-1,2).max(0)
    scale=min(1900/(hi-lo)[0],2000/(hi-lo)[1])
    offset=np.array([1100,1100])-(lo+hi)/2*scale
    pixels=projected[:,:,:2]*scale+offset
    n=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
    # Two soft studio lights; metallic-like tonal range without invented parts.
    light=np.array([-1,-2,4.]);light/=np.linalg.norm(light)
    shade=.46+.42*np.abs(n@light)+.12*np.abs(n@view)
    c=np.clip(colors*shade[:,None]*255,0,255).astype(np.uint8)
    image=Image.new('RGBA',(2200,2200),(255,255,255,0));draw=ImageDraw.Draw(image)
    for i in np.argsort(projected[:,:,2].mean(1)):
        draw.polygon([tuple(v) for v in pixels[i]],fill=tuple(c[i])+(255,))
    image.save(out)
    points={}
    for label,link,off in [('arm','link2',[0,0,0]),('camera','link_eef',[0,0,0]),('tcp','link_tcp',[0,0,0]),('base','base_link',[0,-.24,.16]),('wheel','rim_d_link',[0,0,0])]:
        T=K.fk(q,link);v=(T[:3,3]+T[:3,:3]@off)@basis.T
        points[label]=(v[:2]*scale+offset).tolist()
    if obstacles:
        for label,v in [('target_surface',[.54,0,.55]),('target_box',[.74,-.5,.9]),('obstacle',[-.225,-.32,.6])]:
            pp=np.array(v)@basis.T
            points[label]=(pp[:2]*scale+offset).tolist()
    Path(out).with_suffix('.json').write_text(json.dumps({'projected_labels':points,'q':q.tolist(),'triangles':len(triangles),'source':str(urdf)},indent=2))
    print(out,len(triangles))


if __name__=='__main__':
    render(Path(sys.argv[1]),Path(sys.argv[2]))
