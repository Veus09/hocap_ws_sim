"""Build one Isaac Sim USD scene per HO-Cap sequence, the PALM hand replacing MANO.

Kinematic replay, everything time-sampled at 30 fps (open in Isaac Sim, press Play or scrub the timeline):
  /World/Objects/<id>        annotated object pose (invalid frames hold the last valid pose), mesh from the dataset
  /World/PalmHand_<side>/... PALM link meshes posed by FK of the retargeted 22 joints (outputs/retarget/<NNNN>.npz)
  /World/Reference/MANO_<side>  dataset MANO mesh, half transparent, hidden by default (toggle the eye in Stage)
  /World/Cameras/cam_<serial>   8 calibrated RealSense views; hololens_pv follows the head-mounted camera
  /World/Table                  box under the objects (HO-Cap has no table mesh; top = lowest object point)
Objects and hand links carry collision (convex hull) and kinematic rigid bodies, so contacts can be queried later.
No mass/friction tuning: the annotations do not give them (see collection README).

  python scripts/02_build_usd.py [--seq 0000 ...] [--no-mano]
-> outputs/usd/<NNNN>.usd + outputs/usd/index.csv
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hocap_palm import data, palm  # noqa: E402

OUT = data.WS / 'outputs/usd'
SKIN = Gf.Vec3f(0.72, 0.49, 0.35)
OBJECT_COLORS = [(0.85, 0.33, 0.25), (0.25, 0.55, 0.85), (0.35, 0.75, 0.35), (0.9, 0.75, 0.2), (0.6, 0.4, 0.8)]
CV_TO_USD = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera (z forward, y down) -> USD camera (-z forward, y up)


def gf(m: np.ndarray) -> Gf.Matrix4d:
    return Gf.Matrix4d(*np.asarray(m, dtype=float).T.reshape(-1).tolist())  # Gf is row-vector


def mesh(stage, path: str, vertices: np.ndarray, faces: list[list[int]], color, collision: bool) -> UsdGeom.Mesh:
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    m.CreateSubdivisionSchemeAttr('none')
    m.CreateDoubleSidedAttr(True)
    m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    lo, hi = vertices.min(0), vertices.max(0)
    m.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
    return m


def animated_xform(stage, path: str, poses: np.ndarray, kinematic: bool) -> UsdGeom.Xform:
    x = UsdGeom.Xform.Define(stage, path)
    op = x.AddTransformOp()
    for t, pose in enumerate(poses):
        op.Set(gf(pose), Usd.TimeCode(t))
    if kinematic:
        body = UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        body.CreateKinematicEnabledAttr(True)
    return x


def hold_invalid(poses: np.ndarray, valid: np.ndarray) -> np.ndarray:
    poses = poses.copy()
    good = np.flatnonzero(valid)
    if not len(good):
        return poses
    poses[:good[0]] = poses[good[0]]
    for i in range(good[0] + 1, len(poses)):
        if not valid[i]:
            poses[i] = poses[i - 1]
    return poses


def camera(stage, path: str, world_from_cv: np.ndarray, fx: float, width: int, height: int, times=None):
    cam = UsdGeom.Camera.Define(stage, path)
    aperture = 20.955
    cam.CreateHorizontalApertureAttr(aperture)
    cam.CreateVerticalApertureAttr(aperture * height / width)
    cam.CreateFocalLengthAttr(fx * aperture / width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    op = cam.AddTransformOp()
    if times is None:
        op.Set(gf(world_from_cv @ CV_TO_USD))
    else:
        for t, pose in zip(times, world_from_cv):
            op.Set(gf(pose @ CV_TO_USD), Usd.TimeCode(int(t)))
    return cam


def palm_commit() -> str:
    try:
        return subprocess.run(['git', '-C', str(palm.PALM_WS), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def build(index: str, with_mano: bool) -> dict:
    seq = data.scene(index)
    man = seq['manifest']
    fit_file = data.WS / f'outputs/retarget/{index}.npz'
    if not fit_file.exists():
        raise SystemExit(f'{fit_file} missing: run scripts/01_retarget.py first')
    fit = np.load(fit_file, allow_pickle=False)
    n = seq['n']
    path = OUT / f'{index}.usd'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(data.FPS)
    stage.SetFramesPerSecond(data.FPS)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(n - 1)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    UsdLux.DomeLight.Define(stage, '/World/Lights/Dome').CreateIntensityAttr(600)
    key = UsdLux.DistantLight.Define(stage, '/World/Lights/Key')
    key.CreateIntensityAttr(1500)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-35, 15, 0))

    # Objects
    lowest, centers = [], []
    for k, oid in enumerate(man['object_ids']):
        vertices, faces = data.read_obj(seq['meshes'][oid])
        poses = hold_invalid(seq['object_poses'][:, k].astype(np.float64), seq['object_valid'][:, k])
        prim = f'/World/Objects/{oid}'
        animated_xform(stage, prim, poses, kinematic=True)
        mesh(stage, prim + '/mesh', vertices, faces, OBJECT_COLORS[k % len(OBJECT_COLORS)], collision=True)
        first = poses[0]
        world_v = vertices @ first[:3, :3].T + first[:3, 3]
        lowest.append(world_v[:, 2].min())
        centers.append(first[:3, 3])
    table_top = float(min(lowest))
    cx, cy = np.mean(centers, axis=0)[:2]
    table = UsdGeom.Cube.Define(stage, '/World/Table')
    table.CreateSizeAttr(1.0)
    table.AddTranslateOp().Set(Gf.Vec3d(float(cx), float(cy), table_top - 0.01))
    table.AddScaleOp().Set(Gf.Vec3f(1.4, 1.0, 0.02))
    table.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.5, 0.45)])
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())

    # PALM hands
    stats = {}
    for h, side in enumerate(man['hand_sides']):
        kin = palm.hand(side)
        q, pose = fit[f'{side}_q'], fit[f'{side}_palm']
        links = palm.link_poses(kin, q, pose)
        root = f'/World/PalmHand_{side}'
        UsdGeom.Xform.Define(stage, root)
        for link in palm.MESH_LINKS:
            animated_xform(stage, f'{root}/{link}', links[link], kinematic=True)
            vertices, faces = data.read_obj(palm.link_mesh(link, side))
            geom = mesh(stage, f'{root}/{link}/geom', vertices, faces, SKIN, collision=link != 'forearm')
            # Rigid link meshes are the collision proxies (as in isaac_scene.py); the continuous skin is the visual.
            UsdGeom.Imageable(geom).MakeInvisible()
        skin = palm.Skin(side)
        skinned = skin.frames(kin, q, pose)
        visual = mesh(stage, f'{root}/skin', skinned[0].astype(np.float64), skin.triangles.tolist(), SKIN,
                      collision=False)
        points = visual.GetPointsAttr()
        for t in range(n):
            points.Set(Vt.Vec3fArray.FromNumpy(skinned[t]), Usd.TimeCode(t))
        visual.GetExtentAttr().Set([Gf.Vec3f(*skinned.min((0, 1)).tolist()), Gf.Vec3f(*skinned.max((0, 1)).tolist())])
        joints = stage.GetPrimAtPath(root)
        joints.CreateAttribute('palm:joint_names', Sdf.ValueTypeNames.StringArray).Set(list(fit['joint_names']))
        attr = joints.CreateAttribute('palm:joint_positions', Sdf.ValueTypeNames.DoubleArray)
        for t in range(n):
            attr.Set(Vt.DoubleArray(q[t].tolist()), Usd.TimeCode(t))
        stats[side] = {'fit_ok': int(fit[f'{side}_ok'].sum()), 'rms_mm_mean': float(np.nanmean(fit[f'{side}_rms']) * 1000)}

        if with_mano and seq['hand_vertices'] is not None:
            ref = mesh(stage, f'/World/Reference/MANO_{side}', seq['hand_vertices'][0, h].astype(np.float64),
                       seq['hand_faces'][h].tolist(), (0.3, 0.6, 1.0), collision=False)
            ref.CreateDisplayOpacityAttr([0.35])
            points = ref.GetPointsAttr()
            for t in range(n):
                if seq['hand_valid'][t, h]:
                    points.Set(Vt.Vec3fArray.FromNumpy(seq['hand_vertices'][t, h].astype(np.float32)), Usd.TimeCode(t))
            UsdGeom.Imageable(ref).MakeInvisible()

    # Cameras (OpenCV world_from_camera, static or per frame)
    for cam in seq['cameras']:
        pose = cam['world_from_cv']
        if pose.ndim == 3:
            ok = np.isfinite(pose).all(axis=(1, 2)) & (np.abs(np.linalg.det(pose[:, :3, :3]) - 1) < 1e-3)
            if not ok.any():
                continue
            camera(stage, f'/World/Cameras/{cam["name"]}', hold_invalid(pose, ok), cam['fx'], cam['width'],
                   cam['height'], times=range(n))
        else:
            camera(stage, f'/World/Cameras/{cam["name"]}', pose, cam['fx'], cam['width'], cam['height'])

    info = {'dataset': man['dataset'], 'sequence_id': man['sequence_id'], 'task': man['task'],
            'object_ids': man['object_ids'], 'hand_sides': man['hand_sides'], 'frames': n, 'fps': data.FPS,
            'table_top_z': table_top, 'palm_commit': palm_commit(), 'retarget': stats,
            'world': seq['world_note'], 'source': f'sequences/{index}', 'mode': 'kinematic replay; hand = PALM 22-joint FK of retarget'}
    stage.GetRootLayer().customLayerData = {'hocap_palm': json.dumps(info)}
    stage.GetRootLayer().Save()
    print(index, path.name, f'{n} frames', man['hand_sides'], f'table_top={table_top:.3f}', flush=True)
    return {'index': index, 'dataset': man['dataset'], 'usd': str(path.relative_to(data.WS)), **{k: info[k] for k in
            ('sequence_id', 'task', 'frames')}, 'objects': ' '.join(man['object_ids']),
            'hands': ' '.join(man['hand_sides'])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', nargs='*')
    ap.add_argument('--no-mano', action='store_true', help='skip the MANO reference mesh (smaller files)')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [build(i, not args.no_mano) for i in (args.seq or [r['index'] for r in data.catalog()])]
    if not args.seq:
        with (OUT / 'index.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
