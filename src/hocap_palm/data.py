"""Read the PALM HO-Cap collection (schema palm-collection/1, palm-handoff/1). Pure numpy, no Isaac."""
import json
import os
from pathlib import Path

import numpy as np

WS = Path(os.environ.get('HOCAP_WS', Path(__file__).resolve().parents[2]))
FPS = 30.0


def collection_dir() -> Path:
    found = sorted((WS / 'data/hocap_tasks').glob('collection_*'))
    if not found:
        raise SystemExit(f'no collection_* under {WS / "data/hocap_tasks"}; unzip Hocap_tasks.zip there')
    return found[-1]


def catalog() -> list[dict]:
    root = collection_dir()
    rows = json.loads((root / 'catalog.json').read_text())
    for row in rows:
        row['index'] = Path(row['manifest']).parent.name  # "0000"
        row['dir'] = root / Path(row['manifest']).parent
    return rows


def load_sequence(index: str) -> dict:
    """Manifest + motion arrays of sequences/<index>. Hand arrays are indexed in manifest hand_sides order."""
    seq = collection_dir() / 'sequences' / index
    manifest = json.loads((seq / 'manifest.json').read_text())
    motion = np.load(seq / manifest['motion'], allow_pickle=False)
    data = {k: motion[k] for k in motion.files}
    return {'index': index, 'dir': seq, 'manifest': manifest, **data,
            'meshes': {oid: seq / rel for oid, rel in manifest['meshes'].items()}}


def _camera(name, world_from_cv, fx, width, height):
    return {'name': name, 'world_from_cv': np.asarray(world_from_cv, np.float64), 'fx': float(fx),
            'width': int(width), 'height': int(height)}


def scene(index: str) -> dict:
    """Sequence in a z-up world (metres, table top near z = 0), whatever the source dataset.

    Keys: object_ids, meshes, object_poses (N,K,4,4), object_valid (N,K), hand_sides, hand_joints (N,H,21,3),
    hand_valid (N,H), hand_vertices (N,H,778,3) | None, hand_faces (H,F,3) | None, cameras [dict], world_note.
    Camera world_from_cv is (4,4) static or (N,4,4) per frame, OpenCV axes.
    """
    seq = load_sequence(index)
    man = seq['manifest']
    n = len(seq['frame_ids'])
    out = {'object_ids': man['object_ids'], 'meshes': seq['meshes'], 'hand_sides': man['hand_sides'],
           'hand_joints': seq['hand_joints'].astype(np.float64), 'hand_valid': seq['hand_valid'],
           'hand_vertices': seq.get('hand_vertices'), 'hand_faces': seq.get('hand_faces'), 'manifest': man, 'n': n}
    poses = seq['object_poses'].astype(np.float64)
    if man['dataset'] == 'HO-Cap':
        # World = AprilTag tag_1 on the table: already z-up, table at z ~ 0.
        out['object_poses'], out['object_valid'] = poses, seq['object_valid']
        calib, md = man['metadata']['calibration'], man['metadata']['original']
        cams = [_camera(f'cam_{s}', seq['camera_world_from_camera'][i], calib[f'intrinsics.{s}.yaml']['color']['fx'],
                        640, 480) for i, s in enumerate(md['realsense']['serials'])]
        pv = md.get('hololens')
        if pv and f'intrinsics.{pv["serial"]}.yaml' in calib:
            c = calib[f'intrinsics.{pv["serial"]}.yaml']['color']
            cams.append(_camera('hololens_pv', seq['camera_pv_poses'], c['fx'], c['width'], c['height']))
        out['cameras'], out['world_note'] = cams, 'HO-Cap world (tag_1), unchanged'
        return out
    if man['dataset'] == 'HOI4D':
        # Poses are in the head camera of each frame; the camera log (Open3D trajectory) is world_from_camera with
        # world = first camera. Gravity is not given: up = object local +z at frame 0 (it rests upright on the table),
        # table top = lowest object vertex at frame 0, origin above the object.
        cam = seq['camera_extrinsics_native'].astype(np.float64)
        first = cam @ poses[:, 0]
        up = first[0, :3, 2]
        x = np.cross([0.0, 0.0, 1.0] if abs(up[2]) < 0.9 else [1.0, 0.0, 0.0], up)
        x /= np.linalg.norm(x)
        align = np.eye(4)
        align[:3, :3] = np.stack([x, np.cross(up, x), up])
        vertices, _ = read_obj(seq['meshes'][man['object_ids'][0]])
        rest = align @ first[0]
        low = (vertices @ rest[:3, :3].T + rest[:3, 3])[:, 2].min()
        align[:3, 3] = -np.array([rest[0, 3], rest[1, 3], low])
        world_cam = align @ cam
        out['object_poses'] = np.einsum('tij,tkjl->tkil', world_cam, poses)
        out['object_valid'] = np.isfinite(poses).all(axis=(2, 3))
        joints = out['hand_joints']
        out['hand_joints'] = np.einsum('tij,thkj->thki', world_cam[:, :3, :3], joints) + world_cam[:, None, None, :3, 3]
        if out['hand_vertices'] is not None:
            v = out['hand_vertices'].astype(np.float64)
            out['hand_vertices'] = np.einsum('tij,thkj->thki', world_cam[:, :3, :3], v) + world_cam[:, None, None, :3, 3]
        k = seq['camera_intrinsics']
        out['cameras'] = [_camera('hoi4d_cam', world_cam, k[0, 0], 1920, 1080)]  # HOI4D Kinect colour
        out['world_note'] = 'HOI4D: camera-0 world re-aligned, up = object +z at frame 0, table top z = 0 (estimated)'
        return out
    raise SystemExit(f'{index}: dataset {man["dataset"]} not supported')


def read_obj(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    """Vertices and polygon faces (0-based) of an OBJ; ignores normals/uv."""
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith('v '):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith('f '):
            faces.append([int(p.split('/')[0]) - 1 for p in line.split()[1:]])
    return np.array(vertices), faces
