"""Orthographic vector scenes from the local robot's URDF and STL geometry.

No simulator is started. IK checks pose consistency, not trajectory safety.
Mesh vertex clustering is for illustration only and is never collision data.
"""
from pathlib import Path
from functools import lru_cache
import struct
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "evaluation/models/omni_bot_wholebody_expanded.urdf"
URDF = ET.parse(MODEL).getroot()
IK_RECORDS = []


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    out[:3, 3] = xyz
    return out


def origin(node):
    e = node.find("origin")
    return transform() if e is None else transform(
        np.fromstring(e.get("xyz", "0 0 0"), sep=" "),
        np.fromstring(e.get("rpy", "0 0 0"), sep=" "))


JOINTS = [URDF.find(f"joint[@name='joint{i}']") for i in range(1, 7)]
ORIGINS = [origin(j) for j in JOINTS]
LOW = np.array([float(j.find("limit").get("lower")) for j in JOINTS])
HIGH = np.array([float(j.find("limit").get("upper")) for j in JOINTS])


def frames(base, q):
    x, y, yaw = base
    ts = {"base_link": transform((x, y, .05), (0, 0, yaw))}
    ts["link_base"] = ts["base_link"] @ origin(URDF.find("joint[@name='arm_base_joint']"))
    last = ts["link_base"]
    for i, (o, a) in enumerate(zip(ORIGINS, q), 1):
        last = last @ o @ transform(rpy=(0, 0, a))
        ts[f"link{i}"] = last
    ts["uflite_gripper_link"] = last
    for name in ("uflite_finger1", "uflite_finger2"):
        joint = next(j for j in URDF.findall("joint") if j.find("child").get("link") == name)
        # Closed fingers; contact/grasp forces are not simulated.
        ts[name] = last @ origin(joint)
    ts["link_tcp"] = last @ origin(URDF.find("joint[@name='joint_tcp']"))
    return ts


def solve_pose(base, point, rotation, name):
    target = np.asarray(point)
    def residual(q):
        te = frames(base, q)["link_tcp"]
        return np.r_[te[:3, 3]-target,
                     .25*Rotation.from_matrix(rotation.T @ te[:3, :3]).as_rotvec()]
    rng = np.random.default_rng(14)
    seeds = [np.array([0., -.6, 1.3, 0., .6, 0.])]
    seeds.extend(rng.uniform(LOW+.05, HIGH-.05, size=(24, 6)))
    best = None
    for seed in seeds:
        fit = least_squares(residual, seed, bounds=(LOW, HIGH), max_nfev=220,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun):
            best = fit
        if np.linalg.norm(fit.fun) < 1e-7:
            break
    te = frames(base, best.x)["link_tcp"]
    pos = float(np.linalg.norm(te[:3, 3]-target))
    rot = float(np.linalg.norm(Rotation.from_matrix(rotation.T@te[:3, :3]).as_rotvec()))
    if pos > 1e-4 or rot > 1e-3:
        raise RuntimeError(f"{name}: illustration IK failed: {pos=}, {rot=}")
    IK_RECORDS.append(dict(scene=name, position_error_m=pos, orientation_error_rad=rot,
                           joint_limits_satisfied=bool(np.all(best.x >= LOW) and np.all(best.x <= HIGH))))
    return best.x


@lru_cache(maxsize=32)
def mesh_for(link_name):
    visual = URDF.find(f"link[@name='{link_name}']/visual")
    m = visual.find("geometry/mesh")
    path = Path(m.get("filename").replace("file://", ""))
    if not path.exists():
        rel = str(path).split("/meshes/")[-1]
        package = "my_omnibot_description" if link_name == "base_link" else "xarm_ros2/xarm_description"
        path = ROOT/"src"/package/"meshes"/rel
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    tri = np.frombuffer(data, dtype=dtype, count=count, offset=84)["vertices"].astype(float)
    verts = tri.reshape(-1, 3)
    # Preserve recognisable link surfaces while keeping the vector PDFs small.
    cell = .005 if "finger" in link_name or "gripper" in link_name else .009
    _, inv = np.unique(np.round(verts/cell).astype(int), axis=0, return_inverse=True)
    centres = np.zeros((inv.max()+1, 3)); counts = np.bincount(inv)
    np.add.at(centres, inv, verts)
    centres /= counts[:, None]
    faces = inv.reshape(-1, 3)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])
    faces = faces[keep]
    _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    tri = centres[faces[np.sort(unique)]]
    scale = np.fromstring(m.get("scale", "1 1 1"), sep=" ")
    vo = origin(visual)
    return (tri*scale) @ vo[:3, :3].T + vo[:3, 3]


class Scene:
    def __init__(self, azimuth=-65, elevation=25):
        az, el = np.deg2rad([azimuth, elevation])
        self.view = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
        self.right = np.array([-np.sin(az), np.cos(az), 0])
        self.up = np.cross(self.view, self.right)
        self.projection = np.stack([self.right, self.up])
        self.polygons, self.gray, self.edges, self.widths = [], [], [], []
        self.strokes = []

    def p(self, xyz):
        return np.asarray(xyz) @ self.projection.T

    def faces(self, polygons, gray=.85, shade=True):
        light = np.array([-.2, -.5, 1.]); light /= np.linalg.norm(light)
        for points in polygons:
            points = np.asarray(points)
            normal = np.cross(points[1]-points[0], points[2]-points[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            g = gray*(.82+.18*abs(float(normal@light/norm))) if shade else gray
            self.polygons.append(points)
            self.gray.append(g)
            self.edges.append(str(round(g,4)))
            self.widths.append(.05)

    def stroke(self, points, **kw):
        self.strokes.append((np.asarray(points), kw))

    def box(self, lo, hi, gray=.88, pose=None, outline=True):
        lo, hi = np.asarray(lo), np.asarray(hi)
        v = np.array([[x, y, z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        if pose is not None:
            v = v @ pose[:3, :3].T+pose[:3, 3]
        for inds in [[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]:
            points = v[inds]
            normal = np.cross(points[1]-points[0], points[2]-points[0])
            # Tessellate broad furniture faces so a nearby arm is not hidden
            # by a whole face's single centroid in the vector painter order.
            u=points[1]-points[0];v_=points[3]-points[0]
            nu=max(1,int(np.ceil(np.linalg.norm(u)/.04)))
            nv=max(1,int(np.ceil(np.linalg.norm(v_)/.04)))
            for i in range(nu):
                for j in range(nv):
                    p=points[0]+i/nu*u+j/nv*v_
                    self.faces([[p,p+u/nu,p+u/nu+v_/nv,p+v_/nv]],gray)
            if outline and normal @ self.view > 0:
                for a,b in zip(points,np.roll(points,-1,axis=0)):
                    n=max(1,int(np.ceil(np.linalg.norm(b-a)/.025)))
                    for k in range(n):
                        segment=np.array([a+(b-a)*k/n,a+(b-a)*(k+1)/n])
                        # Give the boundary a tiny outward offset to avoid
                        # z-fighting with the tessellated face itself.
                        segment+=self.view*.012
                        self.polygons.append(segment);self.gray.append(None)
                        self.edges.append("#333333");self.widths.append(.6)

    def cylinder(self, center, radius, length, axis=(0,0,1), gray=.7, n=20):
        axis = np.asarray(axis, dtype=float); axis /= np.linalg.norm(axis)
        u = np.cross(axis, [0,0,1] if abs(axis[2]) < .8 else [1,0,0]); u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        a = np.linspace(0,2*np.pi,n,endpoint=False)
        ring = radius*(np.cos(a)[:,None]*u + np.sin(a)[:,None]*v)
        bottom = np.asarray(center)-length/2*axis+ring
        top = np.asarray(center)+length/2*axis+ring
        self.faces([bottom[::-1],top],gray)
        self.faces([np.array([bottom[i],bottom[(i+1)%n],top[(i+1)%n],top[i]]) for i in range(n)],gray)

    def robot(self, base, q):
        ts = frames(base,q)
        for name in ["base_link", "link_base", *[f"link{i}" for i in range(1,7)],
                     "uflite_gripper_link", "uflite_finger1", "uflite_finger2"]:
            t = ts[name]
            gray = .70 if name == "base_link" else (.60 if "gripper" in name or "finger" in name else .90)
            self.faces(mesh_for(name) @ t[:3,:3].T+t[:3,3],gray)
        x,y,yaw=base
        for angle in (0,np.pi/2,np.pi,3*np.pi/2):
            a=angle+yaw; radial=np.array([np.cos(a),np.sin(a),0.])
            centre=np.array([x,y,.05])+.245*radial
            tangent=np.array([-radial[1],radial[0],0.]);up=np.array([0.,0.,1.])
            self.cylinder(centre,.038,.025,radial,gray=.35,n=18)
            # Two staggered rows of peripheral rollers identify an omni wheel.
            # Their ring radius/row offset follow rim.xacro's approximately
            # 0.0395 m and +/-0.01 m visual geometry.
            for row in (-1,1):
                for k in range(9):
                    t=2*np.pi*k/9+(0 if row<0 else np.pi/9)
                    roller=centre+.0395*(np.cos(t)*tangent+np.sin(t)*up)+row*.010*radial
                    axis=-np.sin(t)*tangent+np.cos(t)*up
                    self.cylinder(roller,.010,.024,axis,gray=.52,n=8)
        return ts["link_tcp"][:3,3]

    def table(self):
        self.box((.33,-.33,.50),(.95,.33,.55),gray=.88)
        for x in (.37,.91):
            for y in (-.28,.28):
                self.box((x-.018,y-.018,0),(x+.018,y+.018,.50),gray=.8)
        self.cylinder((.47,0,.62),.035,.14,gray=.50,n=24)

    def draw(self, ax, margin=.10):
        depths = np.array([p.mean(axis=0)@self.view for p in self.polygons])
        order = np.argsort(depths)
        polys = [self.p(self.polygons[i]) for i in order]
        colors = ["none" if self.gray[i] is None else str(round(self.gray[i],4)) for i in order]
        collection = PolyCollection(polys, facecolors=colors,
                                    edgecolors=[self.edges[i] for i in order],
                                    linewidths=[self.widths[i] for i in order],
                                    antialiaseds=True, zorder=2)
        ax.add_collection(collection)
        points = np.concatenate(polys)
        lo,hi=points.min(axis=0),points.max(axis=0)
        ax.set(xlim=(lo[0]-margin,hi[0]+margin),ylim=(lo[1]-margin,hi[1]+margin),aspect="equal")
        ax.set_axis_off()
        for p,kw in self.strokes:
            xy=self.p(p); ax.plot(xy[:,0],xy[:,1],zorder=3,**kw)

