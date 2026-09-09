"""Shared Isaac Sim import fixes for the omni_bot / ammr_base ports.

Every function here exists because the URDF importer produces a stage that
looks fine and is silently wrong in a specific way. They were found and
verified during Gate 0 / Gate 1 (evaluation/results/isaac_gate0_20260909.md,
isaac_gate1_20260909.md). They live in one module so the fix cannot drift
between the scripts that need it.

Import only AFTER SimulationApp has been constructed.
"""
import os


def import_urdf(path: str, prim_path: str, *, fix_base: bool = False,
                merge_fixed_joints: bool = False, out_dir: str = None,
                verbose: bool = True) -> str:
    """URDF -> USD -> referenced onto the stage, with physics actually loaded.

    The importer puts all physics behind a variant set and authors the
    selection "physx" unconditionally -- but it only EMITS a "physx" variant
    when the URDF has at least one movable joint (that variant sublayers
    "physics" and adds PhysxJointAPI to each movable joint). For an all-fixed
    URDF such as the omni_bot chassis, only "none" and "physics" are written,
    the authored selection dangles, and a dangling selection composes nothing:
    the stage carries geometry and no physics at all, surfacing later as
    "did not match any articulations" -- which reads like the model has no
    physics when it simply was never composed.

    So: keep whatever selection the importer authored if it actually exists
    (on ammr_base that is "physx", the richer variant), and only substitute
    when it dangles.
    """
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from isaacsim.core.utils.stage import add_reference_to_stage
    import omni.usd

    out_dir = out_dir or os.path.join(os.path.dirname(path) or '.', 'isaac_usd')
    os.makedirs(out_dir, exist_ok=True)
    cfg = URDFImporterConfig(urdf_path=path, usd_path=out_dir,
                             merge_fixed_joints=merge_fixed_joints,
                             fix_base=fix_base, allow_self_collision=False,
                             collision_from_visuals=False)
    usd = URDFImporter(cfg).import_urdf()
    add_reference_to_stage(usd_path=usd, prim_path=prim_path)

    stage = omni.usd.get_context().get_stage()
    vset = stage.GetPrimAtPath(prim_path).GetVariantSets().GetVariantSet('Physics')
    names = vset.GetVariantNames()
    authored = vset.GetVariantSelection()
    if authored in names:
        pick, why = authored, '沿用匯入器的選擇'
    else:
        pick = next((c for c in ('physx', 'physics') if c in names),
                    names[0] if names else '')
        why = f'匯入器選擇 {authored!r} 不存在（懸空），改選'
    if pick:
        vset.SetVariantSelection(pick)
    stage.Load(prim_path)
    if verbose:
        print(f'  URDF → USD：{usd}\n'
              f'  Physics 變體：可選 {names}，{why} {pick!r}')
    return prim_path


def walk(stage, under: str = None):
    """Traverse INCLUDING instance proxies.

    The importer marks link subtrees instanceable; a plain stage.Traverse()
    stops at an instance without descending, so most links are invisible to it.
    """
    from pxr import Usd
    for pr in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(
            Usd.PrimDefaultPredicate)):
        if under is None or str(pr.GetPath()).startswith(under):
            yield pr


def bbox_caches():
    """(visual, all-purposes) caches.

    Collision geometry is authored with purpose="guide", which a default bbox
    cache excludes -- it returns an empty range, so collision prims vanish from
    any measurement that uses only the default cache.
    """
    from pxr import Usd, UsdGeom
    return (UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']),
            UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              ['default', 'render', 'proxy', 'guide']))


def physics_parts(stage, prim_path: str):
    """(rigid body paths, articulation root paths) under prim_path."""
    from pxr import UsdPhysics
    bodies, arts = [], []
    for pr in walk(stage, prim_path):
        p = str(pr.GetPath())
        if pr.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(p)
        if pr.HasAPI(UsdPhysics.ArticulationRootAPI):
            arts.append(p)
    return bodies, arts


def bind_frictionless(stage, targets, mat_path='/World/PhysicsMaterials/frictionless'):
    """Bind mu=0 to the given collision prims and return how many took it.

    Both sides of a contact must be bound: the default combine mode averages
    the two materials, so zeroing only the robot still leaves half the
    ground's friction. The URDF states zero friction inside <gazebo><mu1>
    tags, which the URDF importer does not read at all -- the generated USD
    carries no PhysicsMaterialAPI and no binding, and PhysX falls back to its
    default (~0.5). Measured effect on the omni_bot chassis: 37% velocity
    tracking instead of 100%.
    """
    from pxr import UsdPhysics, UsdShade, Sdf
    stage.DefinePrim(Sdf.Path(mat_path).GetParentPath(), 'Scope')
    mp = stage.DefinePrim(mat_path, 'Material')
    m = UsdPhysics.MaterialAPI.Apply(mp)
    m.CreateStaticFrictionAttr().Set(0.0)
    m.CreateDynamicFrictionAttr().Set(0.0)
    m.CreateRestitutionAttr().Set(0.0)
    n = 0
    for pr in targets:
        if pr.IsInstanceProxy():      # an instance proxy cannot be edited
            continue
        UsdShade.MaterialBindingAPI.Apply(pr).Bind(
            UsdShade.Material(mp), UsdShade.Tokens.weakerThanDescendants,
            'physics')
        n += 1
    return n


def collision_prims(stage, under: str, name_filter=None):
    from pxr import UsdPhysics
    out = []
    for pr in walk(stage, under):
        if not pr.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if name_filter and not name_filter(str(pr.GetPath())):
            continue
        out.append(pr)
    return out
