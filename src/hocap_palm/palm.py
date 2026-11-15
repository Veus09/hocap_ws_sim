"""PALM hand: retarget 21 HO-Cap keypoints with the PALM repo's own fitter, and world link poses for USD.

Uses ~/palm_ws (PALM_WS) unchanged: palm_hand_tracking.kinematics.HandKinematics and palm_description URDF/meshes.
"""
import os
import sys
from pathlib import Path

import numpy as np

PALM_WS = Path(os.environ.get('PALM_WS', Path.home() / 'palm_ws'))
DESCRIPTION = PALM_WS / 'src/palm_description'
sys.path.insert(0, str(PALM_WS / 'src/palm_hand_tracking'))
from palm_hand_tracking.kinematics import HandKinematics  # noqa: E402
from palm_hand_tracking.landmark_protocol import MAX_FIT_RMS_METERS  # noqa: E402

# Links that carry a mesh in the URDF (upperarm left out: HO-Cap has no arm, so its pose would be invented).
MESH_LINKS = ('forearm', 'palm',
              'thumb_flex_link', 'thumb_mcp_link', 'thumb_ip_link',
              'index_flex_link', 'index_pip_link', 'index_dip_link',
              'middle_flex_link', 'middle_pip_link', 'middle_dip_link',
              'ring_flex_link', 'ring_pip_link', 'ring_dip_link',
              'little_flex_link', 'little_pip_link', 'little_dip_link')


def hand(side: str) -> HandKinematics:
    return HandKinematics(DESCRIPTION, side)


def joint_names(kin: HandKinematics) -> tuple[str, ...]:
    """22-joint PALM hand command order (model.HAND_NAMES): 21 fitted joints + wrist_pitch."""
    return (*kin.names, 'wrist_pitch')


def link_mesh(link: str, side: str) -> Path:
    return DESCRIPTION / 'meshes' / (f'{link}_left.obj' if side == 'left' else f'{link}.obj')


def retarget(kin: HandKinematics, joints: np.ndarray, valid: np.ndarray) -> dict:
    """Fit every frame. joints (N,21,3) world metres, MediaPipe order. Invalid/rejected frames hold the last good fit.

    Returns q (N,22) rad, palm (N,4,4) world_from_palm (wrist joints 0, so palm == human_hand_root frame),
    rms (N,) metres (NaN when not fitted), ok (N,) bool.
    """
    n = len(joints)
    q = np.zeros((n, 22))
    palm = np.tile(np.eye(4), (n, 1, 1))
    rms = np.full(n, np.nan)
    ok = np.zeros(n, bool)
    previous = None
    for i in range(n):
        points = joints[i].astype(np.float64)
        if valid[i] and np.isfinite(points).all():
            try:
                fit = kin.fit(points, previous=previous)
            except ValueError:
                fit = None
            if fit is not None:
                rms[i] = fit.rms_meters
                if fit.rms_meters <= MAX_FIT_RMS_METERS:
                    previous = fit.positions
                    q[i, :21], palm[i], ok[i] = fit.positions, fit.palm_pose, True
                    continue
        if i:
            q[i], palm[i] = q[i - 1], palm[i - 1]
    # Leading frames before the first good fit copy it, so the hand never jumps from the origin.
    good = np.flatnonzero(ok)
    if len(good) and good[0]:
        q[:good[0]], palm[:good[0]] = q[good[0]], palm[good[0]]
    return {'q': q, 'palm': palm, 'rms': rms, 'ok': ok}


def link_poses(kin: HandKinematics, q: np.ndarray, palm: np.ndarray) -> dict[str, np.ndarray]:
    """World pose (N,4,4) of every mesh link."""
    out = {link: np.empty((len(q), 4, 4)) for link in MESH_LINKS}
    for i in range(len(q)):
        local = kin.transforms(q[i, :21])
        for link in MESH_LINKS:
            out[link][i] = palm[i] @ local[link]
    return out


class Skin:
    """PALM continuous skin (meshes/skin.json), linear-blend skinned like isaac_scene.update_skin.

    The upper arm is cut off (HO-Cap has no arm): triangles touching a vertex weighted > 0.5 to `upperarm` are
    dropped, and the remaining upperarm weights follow the forearm in its rest relation.
    """

    def __init__(self, side: str):
        import json
        skin = json.loads((DESCRIPTION / 'meshes/skin.json').read_text())
        mirror = np.array([1.0, -1.0 if side == 'left' else 1.0, 1.0])
        self.vertices = np.array(skin['vertices']) * mirror
        self.centers = {name: np.array(c) * mirror for name, c in skin['centers'].items()}
        self.weights = {name: np.array([sum(w for link, w in row if link == name) for row in skin['weights']])
                        for name in self.centers}
        triangles = np.array(skin['triangles'])
        keep = (self.weights['upperarm'][triangles] <= 0.5).all(axis=1)
        triangles = triangles[keep]
        triangles = triangles[:, ::-1] if side == 'left' else triangles  # mirroring flips winding
        self.used = np.unique(triangles)
        remap = np.full(len(self.vertices), -1)
        remap[self.used] = np.arange(len(self.used))
        self.triangles = remap[triangles]  # indexes the vertices returned by frames()

    def frames(self, kin: HandKinematics, q: np.ndarray, palm_world: np.ndarray) -> np.ndarray:
        """Skinned vertices in world, (N,V,3) float32, only vertices used by self.triangles."""
        out = np.empty((len(q), len(self.vertices), 3), np.float32)
        offset = np.eye(4)
        offset[:3, 3] = self.centers['upperarm'] - self.centers['forearm']
        for i in range(len(q)):
            local = kin.transforms(q[i, :21])
            local['upperarm'] = local['forearm'] @ offset
            result = np.zeros_like(self.vertices)
            for name, weights in self.weights.items():
                if not weights.any():
                    continue
                transform = palm_world[i] @ local[name]
                result += weights[:, None] * ((self.vertices - self.centers[name]) @ transform[:3, :3].T + transform[:3, 3])
            out[i] = result
        return out[:, self.used]


def keypoints(kin: HandKinematics, q: np.ndarray, palm: np.ndarray) -> np.ndarray:
    """PALM 21 keypoints in world (N,21,3), for comparing with the dataset keypoints."""
    return np.stack([kin.landmarks(q[i, :21]) @ palm[i][:3, :3].T + palm[i][:3, 3] for i in range(len(q))])
