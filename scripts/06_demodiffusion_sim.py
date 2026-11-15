"""Chạy 1 task HO-Cap (đã chuyển bằng 05) trên UR5e + Robotiq 2F-85 trong Isaac Sim có vật lý, theo DemoDiffusion.
Dựa trên ~/hot3d_ws_sim/scripts/09_demodiffusion_sim.py (dùng lại hot3d_sim.droid_franka / hot3d_sim.ur5e), khác:
  * Camera = asset D435 của người dùng (~/d435_import/d435_1/completed435ros.usd, camera màu Camera_rgb 69°x42°):
    3 camera ngoài bố trí như triangulation_scene (cam1, cam2 cùng cạnh dài, cam3 cạnh đối diện, cao 0.5 m trên
    mặt bàn, nhìn vào vùng làm việc) + 1 D435 gắn cổ tay. Asset không bị sửa: ActionGraph ROS tắt, collision tắt,
    clipping gần 2 cm — đều ghi ở layer của stage.
  * Pha DEMO trước (robot ẩn): tay PALM (retarget từ HO-Cap) làm task, vật đi theo pose GT -> demo_video.mp4
    quay từ 3 camera ngoài (đây là "video demo" của người; keypoint 3D đưa cho DemoDiffusion lấy từ annotation).
  * Pha ROBOT: vật đặt lại pose đầu, có vật lý. Mỗi 8 bước gửi Pi-0: ảnh cam ngoài (--pi0_cam, mặc định cam1) +
    ảnh D435 cổ tay (thời gian thực), 7 khớp Franka bóng, gripper, prompt, 10 hành động retarget, time.
    video.mp4 = 4 view: cam1 | cam2 / cam3 | cổ tay.

  bash ~/hot3d_ws_sim/scripts/serve_pi0.sh &        # server Pi-0 DROID (cổng 8000)
  source env_lab.sh
  python scripts/06_demodiffusion_sim.py --task hocap_0000_G18_3 --mode demodiffusion [--time_denoise 0.2] [--gui]
Mode: retarget (baseline động học) | demodiffusion | pi0 (Pi-0 thuần).
Ra: outputs/demodiffusion/runs/<task>/<mode>/{demo_video.mp4, video.mp4, metrics.json, rollout.npz}"""
import argparse
import json
import os
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--human_data", default=os.path.join(os.environ.get("DEMODIFFUSION_DIR", os.path.expanduser("~/demodiffusion")), "demodiffusion/human_data"))
ap.add_argument("--mode", choices=["retarget", "demodiffusion", "pi0"], default="demodiffusion")
ap.add_argument("--time_denoise", type=float, default=0.2)
ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--prompt", default="")
ap.add_argument("--pi0_cam", type=int, default=1, choices=[1, 2, 3], help="camera ngoài gửi cho Pi-0")
ap.add_argument("--cam_height", type=float, default=0.5, help="độ cao camera ngoài trên mặt bàn (m)")
ap.add_argument("--d435", default=os.environ.get("D435_USD", os.path.expanduser("~/d435_import/d435_1/completed435ros.usd")))
ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=360)
ap.add_argument("--no_demo", action="store_true", help="bỏ pha quay video demo tay PALM")
ap.add_argument("--no_video", action="store_true")
ap.add_argument("--gui", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gripper", choices=["mimic", "pd"], default="pd",
                help="pd: bản HOT3D, bỏ mimic, PD cứng từng khớp ngón (mặc định). mimic: khai lại mimic rotZ, chỉ drive finger_joint "
                     "-- THỬ 2026-11-14: mô phỏng nổ (SVD không hội tụ), chưa dùng được")
ap.add_argument("--grip_stiffness", type=float, default=1000.0,
                help="N·m/rad. Bản HOT3D 100: khi ngón chạm vật cách đích ~0.03 rad -> chỉ ~3 N·m, vật xoay/trượt trong tay")
ap.add_argument("--debug_park_wrist", action="store_true", help="(gỡ lỗi) không gắn D435 lên cổ tay")
ap.add_argument("--obj_depen", type=float, default=0.5, help="max_depenetration_velocity của vật (m/s)")
ap.add_argument("--grip_effort", type=float, default=20.0,
                help="giới hạn mô-men mỗi khớp ngón 2F-85 (N·m); bản HOT3D 40 ép vật lún vào cơ cấu ngón")
ap.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(ap)
if "--gui" in sys.argv and not any(a in sys.argv for a in ("--viz", "--visualizer")):
    sys.argv += ["--viz", "kit"]
args = ap.parse_args()
args.enable_cameras = True
if not args.gui:
    args.headless = True
app = AppLauncher(args).app

import numpy as np
import torch
import omni.usd
from PIL import Image, ImageDraw
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.spatial.transform import Rotation as Rot

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from hot3d_sim import droid_franka as F
from hot3d_sim import ur5e as U

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DD = os.environ.get("DEMODIFFUSION_DIR", os.path.expanduser("~/demodiffusion"))
sys.path.insert(0, os.path.join(DD, "demodiffusion/deploy/openpi/packages/openpi-client/src"))

CTRL_HZ = 15
PHYS_DT = 1 / 120
SUB = int(round(1 / (CTRL_HZ * PHYS_DT)))
OPEN_LOOP = 8
HORIZON = 10
FULL_DIM = 32
CAM_RGB = "Geometry/base_link/camera_bottom_screw_frame/camera_link/camera_color_frame/camera_color_optical_frame/Camera_rgb"

tdir = args.task if os.path.isdir(args.task) else os.path.join(args.human_data, args.task)
meta = json.load(open(os.path.join(tdir, "meta.json")))
pdir = os.path.join(tdir, "traj_0", "processed_3d")
tcp = np.load(os.path.join(pdir, "tcp_pose.npy"))
grip_demo = np.load(os.path.join(pdir, "gripper_touch.npy"))
N = len(tcp)
prompt = args.prompt or meta["prompt"]
tag = args.mode + (f"_t{args.time_denoise:g}" if args.mode == "demodiffusion" else "")
out = args.out or os.path.join(WS, "outputs", "demodiffusion", "runs", os.path.basename(os.path.normpath(tdir)), tag)
os.makedirs(out, exist_ok=True)
fdir = os.path.join(out, "frames"); os.makedirs(fdir, exist_ok=True)
for f in os.listdir(fdir): os.remove(os.path.join(fdir, f))
np.random.seed(args.seed)
print(f"[06] {os.path.basename(tdir)} | mode {tag} | {N} bước @ {CTRL_HZ} Hz | prompt: '{prompt}'", flush=True)

# ------------------------------------------------------------------ scene (world = frame đế robot, mặt bàn z = 0)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYS_DT, device="cuda:0"))
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
floor_z = float(meta["floor_z_robot"])
sim_utils.GroundPlaneCfg().func("/World/floor", sim_utils.GroundPlaneCfg(), translation=(0, 0, floor_z))
sim_utils.DomeLightCfg(intensity=700.0).func("/World/dome", sim_utils.DomeLightCfg(intensity=700.0))
sim_utils.DistantLightCfg(intensity=1500.0, angle=1.0).func("/World/sun", sim_utils.DistantLightCfg(intensity=1500.0, angle=1.0),
                                                            orientation=tuple(Rot.from_euler("xy", [0.4, 0.5]).as_quat()))


def quat_z(yaw):
    return tuple(Rot.from_euler("z", yaw).as_quat())


for i, tb in enumerate(meta["tables_robot"]):
    cx, cy, cz = tb["center"]; sx, sy = tb["size_xy_world"]
    top = sim_utils.CuboidCfg(size=(sx, sy, 0.04), collision_props=sim_utils.CollisionPropertiesCfg(),
                              physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.58, 0.42), roughness=0.7))
    top.func(f"/World/table{i}", top, translation=(cx, cy, cz - 0.02), orientation=quat_z(tb["yaw"]))
    h = cz - 0.04 - floor_z
    leg = sim_utils.CuboidCfg(size=(max(0.05, sx - 0.2), max(0.05, sy - 0.2), h),
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.3, 0.25)))
    leg.func(f"/World/table{i}_base", leg, translation=(cx, cy, floor_z + h / 2), orientation=quat_z(tb["yaw"]))
ped = sim_utils.CuboidCfg(size=(0.2, 0.2, -floor_z), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.22)))
ped.func("/World/pedestal", ped, translation=(0, 0, floor_z / 2 - 0.001))

# ------------------------------------------------------------------ robot (như 09: sửa asset gripper)
grip_act = dict(effort_limit_sim=args.grip_effort, velocity_limit_sim=3.0, stiffness=args.grip_stiffness, damping=args.grip_stiffness / 100)
robot = Articulation(ArticulationCfg(
    prim_path="/World/ur5e",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd", variants={"Gripper": "Robotiq_2f_85"},
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, solver_position_iteration_count=16,
                                                                   solver_velocity_iteration_count=1)),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={**dict(zip(U.JOINTS, U.HOME.tolist())), **{j: 0.0 for j in U.GRIPPER_JOINTS}}),
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=U.JOINTS, stiffness=4000.0, damping=200.0, effort_limit_sim=1000.0,
                                          velocity_limit_sim=3.14),
               **({"gripper": ImplicitActuatorCfg(joint_names_expr=U.GRIPPER_JOINTS, **grip_act)} if args.gripper == "pd" else
                  {"gripper_drive": ImplicitActuatorCfg(joint_names_expr=["finger_joint"], effort_limit_sim=args.grip_effort,
                                                        velocity_limit_sim=2.0, stiffness=100.0, damping=1.0),
                   "gripper_mimic": ImplicitActuatorCfg(joint_names_expr=U.GRIPPER_JOINTS[1:], effort_limit_sim=args.grip_effort,
                                                        velocity_limit_sim=2.0, stiffness=0.0, damping=0.0)})}))
fric = UsdShade.Material.Define(stage, "/World/Looks/pad_friction")
pm = UsdPhysics.MaterialAPI.Apply(fric.GetPrim()); pm.CreateStaticFrictionAttr(1.5); pm.CreateDynamicFrictionAttr(1.2)
for pr in Usd.PrimRange(stage.GetPrimAtPath("/World/ur5e")):
    path = str(pr.GetPath())
    if "/Gripper" not in path:
        continue
    if pr.HasAPI(UsdPhysics.ArticulationRootAPI):
        pr.RemoveAPI(UsdPhysics.ArticulationRootAPI); pr.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    if pr.IsA(UsdPhysics.RevoluteJoint):
        for a in list(pr.GetAppliedSchemas()):
            if not a.startswith("PhysxMimicJointAPI"):
                continue
            inst = a.split(":")[1]
            old_api = PhysxSchema.PhysxMimicJointAPI(pr, inst)
            gearing, offset = old_api.GetGearingAttr().Get(), old_api.GetOffsetAttr().Get()
            ref = old_api.GetReferenceJointRel().GetTargets()
            pr.RemoveAppliedSchema(a)
            if args.gripper == "mimic":      # asset khai rotX trên khớp quay quanh Z -> wrist_3 quay loạn; khai lại rotZ
                api = PhysxSchema.PhysxMimicJointAPI.Apply(pr, "rotZ")
                api.CreateGearingAttr(gearing); api.CreateOffsetAttr(offset or 0.0)
                api.CreateReferenceJointAxisAttr("rotZ"); api.CreateReferenceJointRel().SetTargets(ref)
        if args.gripper == "pd" or pr.GetName() == "finger_joint":
            UsdPhysics.DriveAPI.Apply(pr, "angular")
    if pr.HasAPI(UsdPhysics.RigidBodyAPI) and ("inner_finger" in pr.GetName()):
        UsdShade.MaterialBindingAPI.Apply(pr).Bind(fric, UsdShade.Tokens.strongerThanDescendants, "physics")

# ------------------------------------------------------------------ vật HO-Cap (động, có vật lý)
palette = [(0.9, 0.3, 0.3), (0.3, 0.7, 0.95), (0.4, 0.85, 0.45), (0.95, 0.8, 0.2)]
objs = []
for i, o in enumerate(meta["objects"]):
    M = np.array(o["T_robot_obj_start"])
    cfg = RigidObjectCfg(prim_path=f"/World/obj{i}",
                         spawn=sim_utils.UsdFileCfg(usd_path=o["usd"],
                                                    rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16,
                                                                                                 max_depenetration_velocity=args.obj_depen)),
                         init_state=RigidObjectCfg.InitialStateCfg(pos=tuple((M[:3, 3] + [0, 0, 0.003]).tolist()),
                                                                   rot=tuple(Rot.from_matrix(M[:3, :3]).as_quat().tolist())))
    ro = RigidObject(cfg)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/obj{i}/mesh"))
    if mesh:
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*palette[i % len(palette)])])
    objs.append((o, ro))


# ------------------------------------------------------------------ camera D435 (asset người dùng)
def gf(M):
    return Gf.Matrix4d(*np.asarray(M, float).T.reshape(-1).tolist())


def add_d435(name):
    """Reference D435; trả về (transform op của rig, T_rig_cam USD). Override chỉ ở layer stage."""
    rig = stage.DefinePrim(f"/World/{name}", "Xform")
    rig.GetReferences().AddReference(args.d435)
    for pr in Usd.PrimRange(rig):
        if pr.GetName() == "ActionGraph":
            pr.SetActive(False)                      # graph ROS 2 (crash khi headless, không cần ở đây)
        elif pr.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(pr).CreateCollisionEnabledAttr(False)
    cam = UsdGeom.Camera(stage.GetPrimAtPath(f"/World/{name}/{CAM_RGB}"))
    # gần 5 cm cho D435 cổ tay (thân Robotiq sát ống kính che nửa ảnh), 2 cm cho camera ngoài
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05 if name == "cam_wrist" else 0.02, 20.0))
    op = UsdGeom.Xformable(rig).AddTransformOp()
    op.Set(gf(np.eye(4)))
    T = np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T  # rig ở gốc
    return op, T


def place_d435(rig, eye, R_cv):
    op, T_rig_cam = rig
    T_cam = np.eye(4); T_cam[:3, :3] = R_cv @ np.diag([1.0, -1.0, -1.0]); T_cam[:3, 3] = eye
    op.Set(gf(T_cam @ np.linalg.inv(T_rig_cam)))


def look_R(eye, tgt, up=(0, 0, 1.0)):
    z = tgt - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x) + 1e-9
    return np.stack([x, np.cross(z, x), z], 1)          # R_world_cam (OpenCV)


rigs = {n: add_d435(n) for n in ("cam1", "cam2", "cam3", "cam_wrist")}
mk = lambda p: Camera(CameraCfg(prim_path=p, update_period=0, height=args.height, width=args.width, data_types=["rgb"], spawn=None))
cams = {n: mk(f"/World/{n}/{CAM_RGB}") for n in rigs}
cam_view = Camera(CameraCfg(prim_path="/World/cam_view", update_period=0, height=args.height, width=args.width, data_types=["rgb"],
                            spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955, clipping_range=(0.05, 30.0))))

# bố trí kiểu triangulation_scene: robot ở đầu bàn; cam1, cam2 cùng cạnh dài bên trái robot, cam3 cạnh đối diện
tc = np.array(meta["tables_robot"][0]["center"])
ws_c = np.median(tcp[:, :3], 0)
u = np.r_[tc[:2], 0]; u /= np.linalg.norm(u); v = np.cross([0, 0, 1.0], u)
tgt = np.r_[0.5 * (tc[:2] + ws_c[:2]), 0.0]
H = args.cam_height
ext_eye = {"cam1": tgt - 0.6 * u + 0.95 * v + [0, 0, H], "cam2": tgt + 0.6 * u + 0.95 * v + [0, 0, H],
           "cam3": tgt - 0.95 * v + [0, 0, H]}
for n, e in ext_eye.items():
    place_d435(rigs[n], e, look_R(e, tgt))
view_eye = np.array([ws_c[0] + 1.1, ws_c[1] - 0.9, 0.9]); view_tgt = np.array([ws_c[0] * 0.6, ws_c[1], 0.05])


def wrist_cam_pose(T_tcp):
    """D435 cổ tay (kiểu DROID): trên đế gripper, lệch 9 cm về -x TCP, nhìn dọc hướng tiếp cận về đầu ngón."""
    R = T_tcp[:3, :3]
    eye = T_tcp[:3, 3] + R @ np.array([-0.09, 0.0, -0.11])
    tg = T_tcp[:3, 3] + R @ np.array([0.0, 0.0, 0.04])
    return eye, look_R(eye, tg, up=-R[:, 0])


def set_cam(cam, eye, R_cv):
    q = Rot.from_matrix(R_cv @ np.diag([1.0, -1.0, -1.0])).as_quat()
    cam.set_world_poses(torch.tensor([eye], dtype=torch.float32, device=sim.device),
                        torch.tensor([q], dtype=torch.float32, device=sim.device), convention="opengl")


# tay PALM cho pha demo (mesh da, frame robot)
skin = np.load(os.path.join(pdir, "demo_palm_skin.npz"))
demo_obj = np.load(os.path.join(pdir, "demo_obj_pose.npy"))
hand = UsdGeom.Mesh.Define(stage, "/World/demo_palm_hand")
hand.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(skin["vertices"][0]))
hand.CreateFaceVertexCountsAttr([3] * len(skin["triangles"]))
hand.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(skin["triangles"].astype(np.int32).reshape(-1)))
hand.CreateSubdivisionSchemeAttr("none"); hand.CreateDoubleSidedAttr(True)
hand.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.49, 0.35)])

sim.reset()
dev = sim.device
jid = [robot.joint_names.index(n) for n in U.JOINTS]
gid = [robot.joint_names.index(n) for n in U.GRIPPER_JOINTS]
ib = robot.body_names.index("base_link_0")
set_cam(cam_view, view_eye, look_R(view_eye, view_tgt))


def robot_q():
    return robot.data.joint_pos.torch[0, jid].cpu().numpy().astype(float)


def set_targets(q_arm, grip):
    tg = robot.data.joint_pos_target.torch.clone() if hasattr(robot.data, "joint_pos_target") else robot.data.joint_pos.torch.clone()
    tg[0, jid] = torch.tensor(q_arm, dtype=tg.dtype, device=dev)
    tg[0, gid] = torch.tensor(U.GRIPPER_SIGNS * U.GRIPPER_CLOSED * float(grip), dtype=tg.dtype, device=dev)
    robot.set_joint_position_target_index(target=tg)


def physics(n, render_last=True):
    for k in range(n):
        robot.write_data_to_sim()
        sim.step(render=render_last and k == n - 1)
        robot.update(PHYS_DT)
        for _, ro in objs: ro.update(PHYS_DT)


def rgb(cam):
    cam.update(dt=PHYS_DT * SUB)
    return cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)


def put_objects(poses):
    for (o, ro), M in zip(objs, poses):
        p = np.r_[M[:3, 3], Rot.from_matrix(M[:3, :3]).as_quat()]
        ro.write_root_pose_to_sim_index(root_pose=torch.tensor([p], dtype=torch.float32, device=dev))
        ro.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=dev))


def label(img, text):
    im = Image.fromarray(img); ImageDraw.Draw(im).text((8, 6), text, fill=(255, 255, 0)); return np.asarray(im)


def grid(views):
    return np.concatenate([np.concatenate(views[:2], 1), np.concatenate(views[2:], 1)], 0)


def encode(frames_dir, path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(CTRL_HZ), "-i", os.path.join(frames_dir, "%05d.png"),
                    "-c:v", "libvpx-vp9", "-crf", "34", "-b:v", "0", "-row-mt", "1", "-cpu-used", "4", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", path], check=True)
    for f in os.listdir(frames_dir): os.remove(os.path.join(frames_dir, f))


q0 = torch.zeros((1, robot.num_joints), device=dev); q0[0, jid] = torch.tensor(U.HOME, dtype=torch.float32, device=dev)
robot.write_joint_position_to_sim_index(position=q0)
set_targets(U.HOME, 0.0)
physics(30)
P = robot.data.body_link_pose_w.torch[0, ib].cpu().numpy()
Tsim = np.eye(4); Tsim[:3, :3] = Rot.from_quat(P[3:]).as_matrix(); Tsim[:3, 3] = P[:3]
err = np.linalg.norm(Tsim[:3, 3] - U.fk_flange(robot_q())[:3, 3])
print(f"[06] FK numpy vs sim (đế gripper): {err * 1000:.2f} mm", flush=True)
assert err < 5e-3, "mô hình UR5e numpy không khớp asset"

# ------------------------------------------------------------------ pha DEMO: tay PALM làm task, quay 3 camera D435
if not args.no_demo and not args.no_video:
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/ur5e")).MakeInvisible()
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/cam_wrist")).MakeInvisible()
    nd = demo_obj.shape[1]
    for t in range(nd):
        put_objects(demo_obj[:, t])
        hand.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(skin["vertices"][t]))
        physics(1)
        views = [label(rgb(cams[n]), f"{n} (D435) - human demo (PALM hand)") for n in ("cam1", "cam2", "cam3")] + \
                [label(rgb(cam_view), f"overview  t={t / CTRL_HZ:4.1f}s")]
        Image.fromarray(grid(views)).save(os.path.join(fdir, f"{t:05d}.png"))
    encode(fdir, os.path.join(out, "demo_video.mp4"))
    print(f"[06] video demo ({nd} bước) -> {out}/demo_video.mp4", flush=True)
UsdGeom.Imageable(hand).MakeInvisible()
UsdGeom.Imageable(stage.GetPrimAtPath("/World/ur5e")).MakeVisible()
UsdGeom.Imageable(stage.GetPrimAtPath("/World/cam_wrist")).MakeVisible()
start = [np.array(o["T_robot_obj_start"]) for o, _ in objs]
for M in start:
    M[2, 3] += 0.003                                # như init_state: thả từ 3 mm trên pose GT
put_objects(start)
physics(int(1.0 / PHYS_DT), render_last=False)   # vật ổn định trên bàn


def obj_poses():
    return np.stack([ro.data.root_pose_w.torch[0].cpu().numpy() for _, ro in objs])


start_poses = obj_poses()

# ------------------------------------------------------------------ đưa robot tới pose đầu
q_init, info = U.choose_branch([F.pose_to_T(x) for x in tcp])
print(f"[06] nhánh IK UR5e: {info['n_branches']} nhánh, bám thử {info['mean_err'] * 100:.1f} cm", flush=True)
qa = U.HOME.copy()
n_move = int(2.0 * CTRL_HZ)
for k in range(1, n_move + 1):
    s = 0.5 - 0.5 * np.cos(np.pi * k / n_move)
    set_targets(qa + s * (q_init - qa), 0.0); physics(SUB, render_last=False)

client = None
if args.mode != "retarget":
    from openpi_client import image_tools, websocket_client_policy
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[06] đã kết nối Pi-0 {args.host}:{args.port}", flush=True)


def tcp_actual():
    return U.fk(robot_q())


def to7(T):
    return np.r_[T[:3, 3], Rot.from_matrix(T[:3, :3]).as_quat()]


if not args.debug_park_wrist:
    place_d435(rigs["cam_wrist"], *wrist_cam_pose(tcp_actual())); physics(1)
pi0_cam = f"cam{args.pi0_cam}"
qF = F.solve_ik(tcp_actual(), F.HOME, iters=300)
log = dict(tcp_target=[], tcp_actual=[], grip_cmd=[], obj=[], q=[], qF=[], infer_ms=[], chunks=[], q_grip=[])
chunk, k_chunk = None, OPEN_LOOP
q_cmd = robot_q()
grip_now = 0.0
t_wall = time.time()
for t in range(N):
    if args.mode == "retarget":
        T_tgt = F.pose_to_T(tcp[t]); grip_cmd = float(grip_demo[t])
    else:
        if k_chunk >= OPEN_LOOP:
            k_chunk = 0
            qF = F.solve_ik(tcp_actual(), qF, iters=30)
            img_ext, img_wrist = rgb(cams[pi0_cam]), rgb(cams["cam_wrist"])
            obs = {"observation/exterior_image_1_left": image_tools.convert_to_uint8(image_tools.resize_with_pad(img_ext, 224, 224)),
                   "observation/wrist_image_left": image_tools.convert_to_uint8(image_tools.resize_with_pad(img_wrist, 224, 224)),
                   "observation/joint_position": qF.astype(np.float64),
                   "observation/gripper_position": np.array([np.clip(robot.data.joint_pos.torch[0, gid[0]].item() / U.GRIPPER_CLOSED, 0, 1)]),
                   "prompt": prompt}
            if args.mode == "demodiffusion":
                acts = np.zeros((HORIZON, FULL_DIM)); qs = qF.copy()
                for i in range(HORIZON):
                    j = min(t + 1 + i, N - 1)
                    v = F.velocity_ik(qs, F.pose_to_T(tcp[j])); acts[i, :7] = v; qs = F.step(qs, v)
                    acts[i, 7] = grip_demo[min(t + i, N - 1)]
                obs["retargeted_actions"] = acts; obs["time"] = float(args.time_denoise)
            t0 = time.time()
            chunk = np.asarray(client.infer(obs)["actions"])[:, :8]
            log["infer_ms"].append((time.time() - t0) * 1000); log["chunks"].append(chunk)
        a = chunk[k_chunk]; k_chunk += 1
        grip_cmd = 1.0 if a[7] > 0.5 else 0.0
        qF = F.step(qF, a[:7])
        T_tgt = F.fk(qF)
    q_prev = q_cmd.copy(); q = robot_q()
    for _ in range(4):
        q, _ = U.ik_step(q, T_tgt, max_step=0.15)
    dq = q - q_prev; m = np.abs(dq).max() / 0.2
    q_cmd = q_prev + (dq / m if m > 1 else dq)
    for s in range(SUB):
        # gripper mở/đóng theo dốc 0.3 s (lệnh bước làm ngón PD giật, quệt vật)
        rate = PHYS_DT / 0.3
        grip_now += float(np.clip(grip_cmd - grip_now, -rate, rate))
        set_targets(q_prev + (s + 1) / SUB * (q_cmd - q_prev), grip_now)
        robot.write_data_to_sim(); sim.step(render=(s == SUB - 1)); robot.update(PHYS_DT)
        for _, ro in objs: ro.update(PHYS_DT)
    Ta = tcp_actual()
    if not args.debug_park_wrist:
        place_d435(rigs["cam_wrist"], *wrist_cam_pose(Ta))      # D435 cổ tay đi theo gripper
    log["tcp_target"].append(to7(T_tgt)); log["tcp_actual"].append(to7(Ta)); log["grip_cmd"].append(grip_cmd)
    log["obj"].append(obj_poses()); log["q"].append(robot_q()); log["qF"].append(qF.copy())
    log["q_grip"].append(robot.data.joint_pos.torch[0, gid].cpu().numpy().copy())
    if not args.no_video:
        txt = f"{tag} | {prompt} | t={t / CTRL_HZ:5.2f}s grip={'CLOSE' if grip_cmd > 0.5 else 'open'}"
        views = [label(rgb(cams[n]), (f"{n} (D435)" + (" -> Pi-0" if n == pi0_cam and client else "")) + (f"  {txt}" if n == "cam1" else ""))
                 for n in ("cam1", "cam2", "cam3")] + [label(rgb(cams["cam_wrist"]), "wrist D435" + (" -> Pi-0" if client else ""))]
        Image.fromarray(grid(views)).save(os.path.join(fdir, f"{t:05d}.png"))
    if t % 30 == 0:
        print(f"  t={t:4d}/{N} tcp err {np.linalg.norm(Ta[:3, 3] - T_tgt[:3, 3]) * 100:5.1f} cm  grip {grip_cmd:.0f}"
              + (f"  infer {np.mean(log['infer_ms'][-3:]):.0f} ms" if log["infer_ms"] else ""), flush=True)

for _ in range(CTRL_HZ):
    physics(SUB, render_last=False)
final = obj_poses()

# ------------------------------------------------------------------ đánh giá (như 09)
res = dict(task=os.path.basename(os.path.normpath(tdir)), mode=tag, prompt=prompt, n_steps=N, wall_s=round(time.time() - t_wall, 1),
           pi0_camera=pi0_cam if client else None)
tt = np.array(log["tcp_target"]); ta = np.array(log["tcp_actual"]); dem = tcp[:, :3]
res["tcp_track_err_cm"] = dict(mean=float(np.linalg.norm(ta[:, :3] - tt[:, :3], axis=1).mean() * 100),
                               p90=float(np.percentile(np.linalg.norm(ta[:, :3] - tt[:, :3], axis=1), 90) * 100))
res["deviation_from_demo_cm"] = float(np.linalg.norm(ta[:, :3] - dem, axis=1).mean() * 100)
res["objects"] = []
O = np.array(log["obj"])


def soft_cap(z, cap, knee=0.75):
    k = knee * cap; z = np.array(z, float); m = z > k
    z[m] = k + (cap - k) * np.tanh((z[m] - k) / (cap - k))
    return z


for i, (o, _) in enumerate(objs):
    gt_traj = np.array(o["pos_traj_15hz"]); gt_traj[:, 2] = soft_cap(gt_traj[:, 2], meta.get("z_cap", 0.55))
    gt_end = gt_traj[-1]
    p0 = start_poses[i, :3]; pe = final[i, :3]
    gt_lift = float(gt_traj[:, 2].max() - gt_traj[0, 2]); sim_lift = float(O[:, i, 2].max() - p0[2])
    gt_move = float(np.linalg.norm(gt_end - gt_traj[0])); sim_move = float(np.linalg.norm(pe - p0))
    r = dict(name=o["name"], main=o["main"], touched_by_human=o["touched"], end_err_cm=float(np.linalg.norm(pe - gt_end) * 100),
             gt_move_cm=gt_move * 100, sim_move_cm=sim_move * 100, gt_lift_cm=gt_lift * 100, sim_lift_cm=sim_lift * 100,
             fell_off_table=bool(pe[2] < -0.05),
             follow_err_cm=float(np.linalg.norm(O[:, i, :3] - gt_traj[:len(O)], axis=1).mean() * 100))
    if o["main"]:
        lift_ok = gt_lift < 0.05 or sim_lift > 0.5 * gt_lift
        move_ok = r["end_err_cm"] < max(5.0, 0.3 * gt_move * 100)
        r["success"] = bool(lift_ok and move_ok and not r["fell_off_table"])
        res["success"] = r["success"]
    res["objects"].append(r)
if log["infer_ms"]:
    res["infer_ms_mean"] = float(np.mean(log["infer_ms"][1:] or log["infer_ms"]))
json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=1)
np.savez_compressed(os.path.join(out, "rollout.npz"), **{k: np.array(v) for k, v in log.items() if k != "chunks"},
                    chunks=np.array(log["chunks"]) if log["chunks"] else np.zeros(0), demo_tcp=tcp, demo_grip=grip_demo)
main = [r for r in res["objects"] if r.get("main")]
print(f"[06] KẾT QUẢ {res['task']} [{tag}]: success={res.get('success')}  bám TCP {res['tcp_track_err_cm']['mean']:.1f} cm"
      + (f"  | {main[0]['name']}: lệch cuối {main[0]['end_err_cm']:.1f} cm, nâng {main[0]['sim_lift_cm']:.1f}/{main[0]['gt_lift_cm']:.1f} cm"
         if main else ""), flush=True)
if not args.no_video:
    encode(fdir, os.path.join(out, "video.mp4"))
    os.rmdir(fdir)
    print("[06] video ->", os.path.join(out, "video.mp4"), flush=True)
app.close()
