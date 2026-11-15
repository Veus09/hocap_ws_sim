"""HO-Cap task 1 (cầm - nhấc - đặt lại từng vật) -> input DemoDiffusion + file cho UR5e/Robotiq 2F-85.

Mỗi lần tay cầm một vật = 1 task (đoạn: 1.5 s trước lần chạm đầu .. 1 s sau lần nhả cuối). Với mỗi task ghi
  <human_data>/hocap_<NNNN>_<obj>/traj_0/processed_3d/
      righthand_3d_keypoints.npy  (T,21,3) 21 keypoint tay (thứ tự OpenPose/HaMeR), MÉT, FRAME ĐẾ ROBOT, 15 Hz
                                   = đúng input DemoDiffusion (README gốc: "T x 21 x 3, in robot frame")
      eef_pose.npy, retarget_gripper_action.npy   như deploy/preprocess.py gốc (Franka/DROID)
      tcp_pose.npy, gripper_touch.npy, time_warp_src.npy   cho UR5e + 2F-85 (giống ~/hot3d_ws_sim 08)
      demo_obj_pose.npy (K,T,4,4), demo_palm_skin.npz      để dựng lại video demo tay PALM trong sim
  <task>/meta.json  prompt, base robot, bàn, vật (frame robot), camera D435
Keypoint lấy thẳng từ annotation HO-Cap (= keypoint 3D đã triangulate từ 8 camera của dataset), không chạy HaMeR.
Chạm tay-vật: khoảng cách đỉnh MANO tới bề mặt vật < 1 cm (HO-Cap không có contact GT).
Các hàm retarget TCP (căn kẹp theo vật, pha tiếp cận, làm chậm, chọn chỗ đặt robot) gọi lại từ
~/hot3d_ws_sim/scripts/08_to_demodiffusion.py (không sửa file đó).

  ~/env_isaacsim6/bin/python -I scripts/05_to_demodiffusion.py --list            # liệt kê mọi đoạn cầm-đặt
  ~/env_isaacsim6/bin/python -I scripts/05_to_demodiffusion.py --tasks 0002:G10_4 0000:G18_3 ...
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

HOCAP = Path(__file__).resolve().parents[1]
HOT3D = Path(os.environ.get('HOT3D_WS', Path.home() / 'hot3d_ws_sim'))
sys.path.insert(0, str(HOCAP / 'src'))
sys.path.insert(0, str(HOT3D / 'src'))
from hocap_palm import data, palm  # noqa: E402
from hot3d_sim.usd_utils import mesh_to_usd  # noqa: E402

spec = importlib.util.spec_from_file_location('h8', HOT3D / 'scripts/08_to_demodiffusion.py')
h8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h8)

FPS = 30
STEP = 2                     # 30 -> 15 Hz
CONTACT = 0.01
PRE, POST = 45, 30           # khung 30 fps trước lần chạm đầu / sau lần nhả cuối
TABLE_H = 0.75               # độ cao bàn sim (HO-Cap không cho; chỉ để dựng chân bàn + bệ robot)
OUT_USD = HOCAP / 'assets_usd/hocap'
# Tên vật (nhìn ảnh render của scene); id HO-Cap không kèm tên.
NAMES = json.loads((HOCAP / 'configs/object_names.json').read_text()) if (HOCAP / 'configs/object_names.json').exists() else {}


def touch_mask(S, side_idx, k, vertices):
    """(N,) bool: tay chạm vật k (đỉnh MANO cách bề mặt vật < CONTACT)."""
    tree = cKDTree(vertices)
    P = S['object_poses'][:, k]
    hv = S['hand_vertices'][:, side_idx]
    out = np.zeros(S['n'], bool)
    for t in range(S['n']):
        local = (hv[t] - P[t, :3, 3]) @ P[t, :3, :3]          # tay trong frame vật
        out[t] = tree.query(local, distance_upper_bound=CONTACT)[0].min() < CONTACT
    return out


def runs(mask, gap=15):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - b > gap:
            out.append((a, b)); a = i
        b = i
    out.append((a, b))
    return out


def segments(index):
    """Đoạn cầm-đặt của từng vật trong 1 sequence task 1: (obj_k, side, start, end, touch_runs)."""
    S = data.scene(index)
    verts = {oid: data.read_obj(S['meshes'][oid])[0] for oid in S['object_ids']}
    found = []
    for h, side in enumerate(S['hand_sides']):
        for k, oid in enumerate(S['object_ids']):
            m = touch_mask(S, h, k, verts[oid][::max(1, len(verts[oid]) // 3000)])
            r = [x for x in runs(m) if x[1] - x[0] > 20]
            if r:
                found.append(dict(k=k, oid=oid, side=side, h=h, a=r[0][0], b=r[-1][1], mask=m))
    found.sort(key=lambda f: f['a'])
    for i, f in enumerate(found):      # không lấn sang lần cầm vật trước/sau
        lo = found[i - 1]['b'] + 5 if i else 0
        hi = found[i + 1]['a'] - 5 if i + 1 < len(found) else S['n'] - 1
        f['start'] = max(lo, f['a'] - PRE)
        f['end'] = min(hi, f['b'] + POST)
    return S, verts, found


def mesh_mass(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = abs(np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum() / 6)
    return float(np.clip(vol * 400.0, 0.05, 0.6)), vol      # 400 kg/m3: đồ gia dụng rỗng/nhựa (ước lượng)


def object_usd(oid, path):
    V, F = data.read_obj(path)
    F = np.array([f for f in F if len(f) == 3])
    mass, _ = mesh_mass(V, F)
    out = OUT_USD / f'{oid}.usd'
    if not out.exists():
        OUT_USD.mkdir(parents=True, exist_ok=True)
        mesh_to_usd(out, V, F, name='obj', mass=mass, friction=0.9, collision='convexDecomposition')
    return out, mass


def _blend(Ta, Tb, w):
    from scipy.spatial.transform import Slerp
    T = np.eye(4)
    T[:3, 3] = (1 - w) * Ta[:3, 3] + w * Tb[:3, 3]
    T[:3, :3] = Slerp([0, 1], Rot.from_matrix([Ta[:3, :3], Tb[:3, :3]]))(w).as_matrix()
    return T


def _gripper_zlow(origin, Rc, dx, ym, dz, w):
    """Điểm thấp nhất của 2F-85: 2 pad (±1.2 cm, từ 3.5 cm sau tới 3 cm trước TCP) và thân (đế + khớp ngón, ±4.5 cm
    quanh trục, từ 13 cm tới 3 cm sau TCP), TCP = origin + Rc @ (dx, ym, dz)."""
    pads = np.array([[dx + sx * 0.012, ym + sy * (w / 2 + 0.01), dz + sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-0.035, 0.03)])
    body = np.array([[dx + sx * 0.045, ym + sy * 0.05, dz + sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-0.13, -0.03)])
    return float((origin + np.r_[pads, body] @ Rc.T)[:, 2].min())


def plan_tcp(g, R, touch, Mo, V, n_app=8, n_blend=6, n_dwell=4, n_hold=8, n_ret=8, back=0.12):
    """TCP 2F-85 bám vật (thay grasp_correction + add_approach của 08).

    g, R: tâm nắm + frame tay người (z tiếp cận, y kẹp), frame robot, 15 Hz. Mo (T,4,4) pose GT vật chính, V đỉnh vật.
    1) Tư thế kẹp lúc chạm đầu t0: thử họ "theo tay" Rz(psi) R_tay Rz(phi) và họ "từ trên xuống" (z = -Z, xoay quanh
       trục đứng); chi phí = độ rộng vật giữa 2 pad (loại nếu > 8 cm hoặc < 1.5 cm) + khoảng cách tới tâm vật + lệch tâm pad + phạt pad dưới mặt bàn + phạt kẹp ngang
       sát bàn (thân gripper quệt vào vật/bàn) + lệch nhẹ khỏi tay. Tâm pad căn vào vật bằng h8._pad_fit.
    2) t0..t1 (đang cầm): T_tcp = Mo[t] @ G (G = pose kẹp trong frame vật) -> vật đi đúng quỹ đạo GT.
    3) Trước t0: theo tay, chuyển dần về điểm lùi `back` dọc trục tiếp cận, rồi đi thẳng vào.
    4) Sau t1: rút lùi `back` dọc trục tiếp cận trong n_ret bước rồi nối lại quỹ đạo tay.
    -> (tcp (T,7), grip (T,), info)"""
    n = len(g)
    idx = np.flatnonzero(touch)
    Th = np.tile(np.eye(4), (n, 1, 1)); Th[:, :3, :3] = R; Th[:, :3, 3] = g
    if not len(idx):
        return Th, np.zeros(n), dict(width=float('nan'), family='none')
    t0, t1 = idx[0], idx[-1]
    W = V @ Mo[t0, :3, :3].T + Mo[t0, :3, 3] - g[t0]
    centroid = W.mean(0)                                   # tâm vật (quanh tâm nắm): kẹp gần tâm cho vững
    cands = []
    for ps in np.radians(np.arange(-45, 46, 15)):
        for ph in np.radians(np.arange(-90, 90, 15)):
            cands.append(('hand', h8._rot_world_z(ps) @ R[t0] @ h8._rz(ph)))
    for a in np.radians(np.arange(0, 180, 10)):
        y = np.array([np.cos(a), np.sin(a), 0.0]); z = np.array([0, 0, -1.0])
        cands.append(('top', np.stack([np.cross(y, z), y, z], 1)))
    best = None
    for fam, Rc in cands:
        w, dx, ym, dz, k = h8._pad_fit(W @ Rc)
        if k < 20 or w > 0.08 or w < 0.015:      # mẩu mép quá mỏng / quá ít điểm dưới pad: không kẹp vững
            continue
        # không cho vật vào sâu quá 2 cm sau TCP (vùng khớp ngón trong của 2F-85 bóp đỉnh vật -> kẹt khi nhả)
        l = W @ Rc - np.array([dx, 0.0, 0.0])
        col = (np.abs(l[:, 0]) < 0.03) & (np.abs(l[:, 1] - ym) < 0.05)
        if col.any() and dz > l[col, 2].min() + 0.02:
            dz = float(l[col, 2].min() + 0.02)
            pad = (np.abs(l[:, 0]) < 0.012) & (l[:, 2] > dz - 0.035) & (l[:, 2] < dz + 0.03) & (np.abs(l[:, 1]) < 0.07)
            if pad.sum() < 20:
                continue
            w = float(np.ptp(l[pad, 1])); ym = float(0.5 * (l[pad, 1].min() + l[pad, 1].max()))
            if not 0.015 <= w <= 0.08:
                continue
        zlow = _gripper_zlow(g[t0], Rc, dx, ym, dz, w)
        if zlow < 0.003 and Rc[2, 2] < -0.3:      # kẹp chúc xuống vật thấp: lùi TCP lên để đầu ngón cách bàn 3 mm
            dz -= (0.003 - zlow) / -Rc[2, 2]
            l = W @ Rc - np.array([dx, 0.0, 0.0])
            pad = (np.abs(l[:, 0]) < 0.012) & (l[:, 2] > dz - 0.035) & (l[:, 2] < dz + 0.03) & (np.abs(l[:, 1]) < 0.07)
            if pad.sum() < 20:
                continue
            w = float(np.ptp(l[pad, 1])); ym = float(0.5 * (l[pad, 1].min() + l[pad, 1].max()))
            if not 0.015 <= w <= 0.08:
                continue
            zlow = _gripper_zlow(g[t0], Rc, dx, ym, dz, w)
        c = np.array([dx, ym, dz])
        p = g[t0] + Rc @ c
        side = 0.1 if (Rc[2, 2] > -0.6 and p[2] < 0.08) else 0.0    # kẹp ngang sát bàn: thân gripper quệt vật/bàn
        dev = np.arccos(np.clip((np.trace(R[t0].T @ Rc) - 1) / 2, -1, 1))
        off = np.linalg.norm(Rc @ c - centroid)
        if zlow < 0.003:                         # pad hoặc thân gripper xuyên mặt bàn: bỏ
            continue
        cost = 0.3 * w + 0.5 * abs(dx) + 1.0 * off + side + 0.02 * dev
        if best is None or cost < best[0]:
            best = (cost, fam, Rc, p, w)
    if best is None:
        return Th, touch.astype(float), dict(width=float('nan'), family='human_fallback')
    _, fam, Rc, p, width = best
    Tg = np.eye(4); Tg[:3, :3] = Rc; Tg[:3, 3] = p
    G = np.linalg.inv(Mo[t0]) @ Tg
    T = Th.copy()
    for t in range(t0, t1 + 1):
        T[t] = Mo[t] @ G
    pre = Tg.copy(); pre[:3, 3] -= Rc[:, 2] * back
    ta = max(0, t0 - n_dwell)                     # tới tư thế kẹp sớm n_dwell bước, đứng yên rồi mới đóng
    a0 = max(0, ta - n_app); b0 = max(0, a0 - n_blend)
    for t in range(b0, a0):
        T[t] = _blend(Th[t], pre, (t - b0 + 1) / (a0 - b0 + 1))
    for t in range(a0, ta):
        T[t] = _blend(pre, Tg, (t - a0 + 1) / (ta - a0 + 1))
    for t in range(ta, t0):
        T[t] = Tg
    # nhả: mở ở t1+1, đứng yên n_hold bước cho ngón mở hết, rút thẳng `back` dọc trục tiếp cận, rồi nối quỹ đạo tay
    Te = T[t1]; post = Te.copy(); post[:3, 3] -= Te[:3, 2] * back
    h1 = min(n, t1 + 1 + n_hold); e0 = min(n, h1 + n_ret)
    for t in range(t1 + 1, h1):
        T[t] = Te
    for t in range(h1, e0):
        T[t] = _blend(Te, post, (t - h1 + 1) / n_ret)
    for t in range(e0, min(n, e0 + n_blend)):
        T[t] = _blend(post, Th[t], (t - e0 + 1) / (n_blend + 1))
    for t in range(min(n, e0 + n_blend), n):
        T[t] = Th[t]
    grip = np.zeros(n); grip[t0:t1 + 1] = 1.0
    return T, grip, dict(width=float(width), family=fam, t0=int(t0), t1=int(t1))


def build(index, oid, out_root, args):
    S, verts, segs = segments(index)
    seg = next(s for s in segs if s['oid'] == oid)
    side, h, k = seg['side'], seg['h'], seg['k']
    sl = np.arange(seg['start'], seg['end'] + 1)
    s15 = sl[::STEP]
    n15 = len(s15)
    # ---- dữ liệu dạng HOT3D task cho các hàm của 08
    kp = S['hand_joints'][s15, h]                           # (T,21,3), thứ tự HO-Cap = OpenPose
    touch_full = np.stack([touch_mask(S, h, j, verts[o][::max(1, len(verts[o]) // 3000)])[sl]
                           for j, o in enumerate(S['object_ids'])])
    touch15 = np.stack([touch_full[:, i:i + STEP].any(1) for i in range(0, len(sl), STEP)], 1)
    P = dict(kp=kp, side=side, vis=np.ones(n15, bool), touch=touch15, fps=float(FPS))
    obj_pose = S['object_poses'][sl].transpose(1, 0, 2, 3)  # (K,N,4,4) world
    h8.mesh_vertices = lambda uid: verts[uid]
    T = dict(objects=[dict(uid=o, name=NAMES.get(o, o)) for o in S['object_ids']], obj_pose=obj_pose)
    # ---- bàn (HO-Cap: mặt bàn z = 0, không biết kích thước -> phủ vật + vùng tay chạm, ±25 cm)
    starts = obj_pose[:, 0, :3, 3]
    contact = kp[touch15.any(0)][:, 0] if touch15.any() else kp[:, 0]
    pts = np.r_[starts, contact]
    lo, hi = pts[:, :2].min(0) - 0.25, pts[:, :2].max(0) + 0.25
    pv = S['cameras'][-1]['world_from_cv'] if S['cameras'][-1]['name'] == 'hololens_pv' else None
    head = np.median(pv[sl, :3, 3], 0) if pv is not None else np.r_[pts[:, :2].mean(0) - [0.5, 0], 0.5]
    Sx = dict(tables=[(0.0, lo, hi)], head_median=head, rest_points=starts)
    T_wb, tab, h_tab = h8.choose_base(Sx, [P])
    to_r = np.linalg.inv(T_wb)
    tables_r = [dict(center=(to_r @ np.r_[(lo + hi) / 2, 0.0, 1])[:3].tolist(), size_xy_world=(hi - lo).tolist(),
                     yaw=float(-Rot.from_matrix(T_wb[:3, :3]).as_euler('xyz')[2]))]
    kp_r = h8.to_frame(T_wb, kp.reshape(-1, 3)).reshape(kp.shape)
    # ---- TCP (giống main() của 08)
    kps = np.stack([h8.ema_zero_phase(kp_r[:, i]) for i in range(21)], 1)
    R, A, B, C = h8.hand_frame(kps, mirror=side == 'left')
    g = (B + C) / 2
    Mo15 = to_r[None] @ obj_pose[k, ::STEP]
    # chạm liên tục: lấp khe ngắn trong mặt nạ chạm vật chính (tay nhích ngón khi cầm)
    tk = touch15[k].copy()
    if tk.any():
        i0, i1 = np.flatnonzero(tk)[[0, -1]]
        tk[i0:i1 + 1] = True
    Tt, grip, ginfo = plan_tcp(g, R, tk, Mo15, verts[oid])
    width = ginfo['width']
    q = Rot.from_matrix(Tt[:, :3, :3]).as_quat()
    for i in range(1, n15):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    g = Tt[:, :3, 3]
    zc = h8.soft_cap(g[:, 2].copy(), args.z_cap)
    g = np.c_[g[:, :2], np.maximum(zc, 0.012)]
    tcp = np.c_[g, q]
    corr = Tt[:, :3, 3] - (B + C) / 2
    eef_dd, grip_dd = h8.dd_original(kp_r)
    src = h8.time_warp(tcp[:, :3], tcp[:, 3:], args.vmax, args.wmax)
    tcp_w = np.concatenate(h8.resample(src, pos=tcp[:, :3], quat=tcp[:, 3:]), 1)
    grip_w = h8.resample(src, binary=grip)[0]
    # ---- kiểm tra
    dist = np.linalg.norm(g - [0, 0, h8.UR5E_SHOULDER_Z], axis=1)
    reach = float(np.mean((dist > 0.30) & (dist < h8.UR5E_REACH) & (np.linalg.norm(g[:, :2], axis=1) > 0.20)))
    name = NAMES.get(oid, oid)
    prompt = args.prompt_tpl.format(o=name)
    tname = f'hocap_{index}_{oid}'
    tdir = Path(out_root) / tname
    pdir = tdir / 'traj_0/processed_3d'
    pdir.mkdir(parents=True, exist_ok=True)
    np.save(pdir / 'righthand_3d_keypoints.npy', kp_r.astype(np.float64))
    np.save(pdir / 'eef_pose.npy', eef_dd)
    np.save(pdir / 'retarget_gripper_action.npy', grip_dd)
    np.save(pdir / 'tcp_pose.npy', tcp_w)
    np.save(pdir / 'gripper_touch.npy', grip_w)
    np.save(pdir / 'time_warp_src.npy', src)
    # ---- demo (để dựng video tay PALM làm task, frame robot, 15 Hz)
    np.save(pdir / 'demo_obj_pose.npy', np.einsum('ij,ktjl->ktil', to_r, obj_pose[:, ::STEP]))
    fit = np.load(HOCAP / f'outputs/retarget/{index}.npz')
    kin = palm.hand(side)
    skin = palm.Skin(side)
    sk = skin.frames(kin, fit[f'{side}_q'][s15], to_r[None] @ fit[f'{side}_palm'][s15])
    np.savez_compressed(pdir / 'demo_palm_skin.npz', vertices=sk, triangles=skin.triangles)
    objs = []
    for j, o in enumerate(S['object_ids']):
        usd, mass = object_usd(o, S['meshes'][o])
        Mo = to_r[None] @ obj_pose[j]
        traj = h8.resample(src, pos=Mo[::STEP, :3, 3])[0]
        objs.append(dict(uid=o, name=NAMES.get(o, o), usd=str(usd), mass=mass, T_robot_obj_start=Mo[0].tolist(),
                         T_robot_obj_end=Mo[-1].tolist(), pos_traj_15hz=traj.round(4).tolist(),
                         touched=bool(touch15[j].any()), main=j == k))
    lift = float(obj_pose[k, :, 2, 3].max() - obj_pose[k, 0, 2, 3])
    meta = dict(dataset='HO-Cap', index=index, sequence_id=S['manifest']['sequence_id'], hocap_task=S['manifest']['task'],
                frames_30fps=[int(seg['start']), int(seg['end'])], name=f'pick_place_{name}', prompt=prompt, hand=side,
                ctrl_hz=h8.CTRL_HZ, n_steps=len(src), n_steps_demo=n15, slowdown=round(len(src) / n15, 2),
                T_world_robot=T_wb.tolist(), table_height_world=0.0, floor_z_robot=-TABLE_H,
                tables_robot=tables_r, objects=objs, main_object=name, z_cap=args.z_cap,
                checks=dict(reach=reach, obj_min_width_m=width, gt_lift_m=lift, grasp=ginfo,
                            grasp_correction_cm=float(np.linalg.norm(corr, axis=1).max() * 100)))
    (tdir / 'meta.json').write_text(json.dumps(meta, indent=1))
    print(f'{tname}: {side} {n15} bước demo -> {len(src)} bước robot | rộng chỗ kẹp {width * 100:.1f} cm | '
          f'kẹp {ginfo["family"]} | tầm với {reach:.0%} | nhấc {lift * 100:.0f} cm | "{prompt}"', flush=True)
    return tname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--tasks', nargs='*', default=[], help='NNNN:OBJ_ID')
    ap.add_argument('--out', default=str(Path(os.environ.get('DEMODIFFUSION_DIR', Path.home() / 'demodiffusion')) / 'demodiffusion/human_data'))
    ap.add_argument('--prompt_tpl', default='pick up the {o} and put it back down')
    ap.add_argument('--z_cap', type=float, default=0.55)
    ap.add_argument('--vmax', type=float, default=0.5)
    ap.add_argument('--wmax', type=float, default=1.5)
    args = ap.parse_args()
    if args.list:
        for row in data.catalog():
            if row['dataset'] != 'HO-Cap' or row['task'] != '1':
                continue
            S, _, segs = segments(row['index'])
            for s in segs:
                P = S['object_poses'][s['a']:s['b'] + 1, s['k'], 2, 3]
                print(f"{row['index']}:{s['oid']} {s['side']:5s} frames {s['start']}-{s['end']} "
                      f"({(s['end'] - s['start']) / FPS:.1f}s) lift {(P.max() - P[0]) * 100:.0f} cm", flush=True)
    for t in args.tasks:
        index, oid = t.split(':')
        build(index, oid, args.out, args)


if __name__ == '__main__':
    main()
