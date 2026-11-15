# hocap_ws_sim — task HO-Cap/HOI4D trong Isaac Sim với bàn tay PALM + DemoDiffusion trên UR5e

Repo gồm hai phần:

1. **Scene replay** (script 01–04). Đưa 51 sequence trong `Hocap_tasks.zip` (50 HO-Cap + 1 HOI4D) vào Isaac Sim 6.0.
   Bàn tay MANO của dataset được thay bằng **bàn tay PALM 22 khớp** (repo `binboong/palm`), retarget bằng chính
   `palm_hand_tracking.kinematics.HandKinematics.fit` của repo PALM.
2. **DemoDiffusion có vật lý** (script 05–07). Cắt các đoạn người cầm–đặt vật thành task DemoDiffusion, rồi cho
   **UR5e + Robotiq 2F-85** làm lại task trong Isaac Lab. Có 3 camera D435 ngoài + 1 D435 cổ tay, Pi-0 DROID làm policy.

Repo **chỉ có code và tài liệu**. Dữ liệu, scene USD, video và task đã sinh **không** nằm trong git
(xem [Dữ liệu](#2-dữ-liệu-không-có-trong-repo)); chạy các script bên dưới để sinh lại trên máy mình.

---

## 1. Yêu cầu

| Thành phần | Phiên bản đã dùng |
|---|---|
| Ubuntu | 24.04 (22.04 cũng được) |
| GPU NVIDIA | RTX, **≥ 16 GB VRAM** nếu chạy Pi-0 cùng Isaac Sim trên một GPU (Pi-0 ~9 GB). Riêng phần 1: ≥ 8 GB |
| Ổ đĩa | ~4 GB dữ liệu + 2,3 GB scene USD + 12 GB checkpoint Pi-0 |
| Isaac Sim | 6.0.0.1 (pip), Python 3.12 |
| Isaac Lab | 3.0.0b2 (chỉ cần cho script 06/07) |
| uv | để cài openpi (Pi-0), Python 3.11 |
| ffmpeg | có `libvpx-vp9` (`sudo apt install ffmpeg`) |

Các repo phụ thuộc. Đường dẫn mặc định là ở `~`; đặt chỗ khác thì đặt biến môi trường tương ứng.

| Repo | Dùng cho | Đường dẫn mặc định | Biến môi trường | Quyền truy cập |
|---|---|---|---|---|
| `git@github.com:binboong/palm.git` | Động học + mesh/skin tay PALM (01, 02, 05) | `~/palm_ws` | `PALM_WS` | **private**, xin quyền chủ repo |
| `git@github.com:Veus09/hot3d_ws_sim.git` | IK UR5e, Franka "bóng", tạo USD vật, server Pi-0 (05, 06) | `~/hot3d_ws_sim` | `HOT3D_WS` | public |
| `git@github.com:Veus09/realsense_d435.git` | Asset camera D435 (06) | `~/d435_import` | `D435_USD` (đường dẫn file `.usd`) | **private**, xin quyền chủ repo |
| `https://github.com/demodiffusion/demodiffusion.git` (+ submodule `openpi`) | Pi-0 DROID + client | `~/demodiffusion` | `DEMODIFFUSION_DIR` | public |

Commit đã kiểm (2026-11-14): palm `c3c6758`, hot3d_ws_sim `d6f6719`, realsense_d435 `2344b4b`,
demodiffusion `ad9cacc` / openpi `18006ae`. Không repo nào ở trên bị sửa; mọi thay đổi nằm trong repo này.

## 2. Dữ liệu (không có trong repo)

`Hocap_tasks.zip` (~725 MB) là gói task HO-Cap/HOI4D do nhóm chuẩn bị, định dạng `palm-handoff/1`.
Mỗi sequence có `motion.npz` (khớp tay MANO 21 điểm, mesh MANO, pose 6 bậc tự do của vật, pose camera),
`meshes/` (mesh vật), `sources/` (file gốc + thông số camera), `manifest.json`. Gói không có video RGB.
**Xin file này từ nhóm.** Nó chứa dữ liệu HO-Cap/HOI4D và mesh MANO, phải tuân theo license gốc của các dataset
đó nên không phân phối công khai.

Repo này cũng không đưa lên các file sinh ra có chứa hình học tay PALM (scene USD, `demo_palm_skin.npz`, video)
vì repo PALM là private.

## 3. Cài đặt từng bước

### 3.1 Clone

```bash
cd ~
git clone git@github.com:Veus09/hocap_ws_sim.git
git clone git@github.com:binboong/palm.git palm_ws               # cần quyền
git clone git@github.com:Veus09/hot3d_ws_sim.git
git clone git@github.com:Veus09/realsense_d435.git d435_import   # cần quyền; asset ở d435_import/d435_1/completed435ros.usd
git clone --recurse-submodules https://github.com/demodiffusion/demodiffusion.git   # chỉ cần cho phần 2
```

Repo PALM chỉ cần mã nguồn (`src/palm_hand_tracking`, `src/palm_description`), **không cần build ROS**.

### 3.2 Môi trường Isaac Sim 6 (script 01–05)

```bash
python3.12 -m venv ~/env_isaacsim6
source ~/env_isaacsim6/bin/activate
pip install --upgrade pip
pip install "isaacsim[all,extscache]==6.0.0.1" --extra-index-url https://pypi.nvidia.com
pip install trimesh scipy pillow pyyaml
deactivate
```

Lần đầu chạy Isaac Sim phải đồng ý EULA; `env.sh` đặt sẵn `OMNI_KIT_ACCEPT_EULA=YES`.
Nếu lệnh `pip install isaacsim` đổi theo bản mới, làm theo hướng dẫn cài pip của NVIDIA cho Isaac Sim 6.0.

### 3.3 Môi trường Isaac Lab 3.0 (script 06–07)

Cần một venv có Isaac Sim 6.0.0.1 + Isaac Lab 3.0.0b2 (giống `hot3d_ws_sim`). Máy gốc dùng `~/fall26_envtest/.venv`
và cài Isaac Lab từ wheel `isaaclab-3.0.0b2.post1` build từ mã nguồn Isaac Lab 3.0 beta 2 (bản mà
IsaacLab-Arena `release/0.3.0` dùng). Ví dụ:

```bash
python3.12 -m venv ~/isaaclab_venv && source ~/isaaclab_venv/bin/activate
pip install "isaacsim[all,extscache]==6.0.0.1" --extra-index-url https://pypi.nvidia.com
pip install /đường/dẫn/isaaclab-3.0.0b2*.whl          # hoặc pip install -e <IsaacLab>/source/isaaclab (bản 3.0.0b2)
pip install trimesh scipy pillow
pip install -e ~/demodiffusion/demodiffusion/deploy/openpi/packages/openpi-client   # client gọi server Pi-0
deactivate
```

Khi `source env_lab.sh`, đặt `HOT3D_VENV=~/isaaclab_venv` nếu venv không nằm ở `~/fall26_envtest/.venv`.

### 3.4 Pi-0 DROID (openpi, chỉ cho chế độ `demodiffusion`/`pi0`)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh               # nếu chưa có uv
cd ~/demodiffusion/demodiffusion/deploy/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync                                 # tạo .venv (Python 3.11)
```

Lần chạy server đầu tiên tự tải checkpoint `pi0_droid` (~12 GB) từ `s3://openpi-assets` về `~/.cache/openpi/`.
Phải dùng openpi trong submodule của DemoDiffusion (có sửa để nhận demo + `time`), không dùng openpi gốc.

### 3.5 Giải nén dữ liệu

```bash
cd ~/hocap_ws_sim
mkdir -p data/hocap_tasks
unzip -q /đường/dẫn/Hocap_tasks.zip -d data/hocap_tasks      # -> data/hocap_tasks/collection_<timestamp>/sequences/0000..0050
```

### 3.6 Đường dẫn khác mặc định

Mọi đường dẫn đều đọc từ biến môi trường, đặt **trước** khi `source`:

```bash
export PALM_WS=/opt/palm_ws HOT3D_WS=/opt/hot3d_ws_sim DEMODIFFUSION_DIR=/opt/demodiffusion
export D435_USD=/opt/d435/d435_1/completed435ros.usd ISAAC_PYTHON=/opt/env_isaacsim6/bin/python HOT3D_VENV=/opt/isaaclab_venv
```

## 4. Chạy

### 4.1 Scene replay với tay PALM

```bash
cd ~/hocap_ws_sim && source env.sh
$ISAAC_PYTHON -I scripts/01_retarget.py                # ~3 phút, 12 tiến trình (`--jobs`) -> outputs/retarget/<NNNN>.npz + summary.csv
$ISAAC_PYTHON -I scripts/02_build_usd.py               # ~2 phút -> outputs/usd/<NNNN>.usd + index.csv (2,3 GB)
$ISAAC_PYTHON scripts/03_render.py --seq 0009 [--mano] # video kiểm tra headless -> outputs/render/0009.mp4
$ISAAC_PYTHON scripts/04_view.py --seq 0009 [--mano] [--cam 043422252387|hololens_pv] [--paused]   # xem trực tiếp (GUI)
```

Có thể chạy một sequence: `01_retarget.py --seq 0009`, `02_build_usd.py --seq 0009`.
Mở thủ công: Isaac Sim > File > Open `outputs/usd/<NNNN>.usd`, bấm Play (30 fps).
`outputs/usd/index.csv` liệt kê sequence → task, vật, tay.

**Kiểm tra cài đúng:** `outputs/retarget/summary.csv` có 65 tay, RMS trung bình ~3,5 mm; `04_view.py --seq 0000`
hiện tay PALM trái cầm giày.

### 4.2 DemoDiffusion: UR5e + Robotiq 2F-85 làm lại task

```bash
cd ~/hocap_ws_sim && source env.sh
# 1. HO-Cap -> task DemoDiffusion (CPU, vài giây/task) -> $DEMODIFFUSION_DIR/demodiffusion/human_data/hocap_<NNNN>_<vật>/
$ISAAC_PYTHON -I scripts/05_to_demodiffusion.py --list           # liệt kê mọi đoạn cầm-đặt (task 1)
$ISAAC_PYTHON -I scripts/05_to_demodiffusion.py --tasks 0000:G18_3 0021:G20_4 0005:G19_3 0002:G10_2 0023:G21_2

# 2. server Pi-0 (terminal khác, giữ chạy; lần đầu tải checkpoint 12 GB)
bash $HOT3D_WS/scripts/serve_pi0.sh                               # cổng 8000, ~9 GB VRAM

# 3. mô phỏng
source env_lab.sh
python scripts/06_demodiffusion_sim.py --task hocap_0000_G18_3 --mode retarget        # không cần server Pi-0
python scripts/06_demodiffusion_sim.py --task hocap_0000_G18_3 --mode demodiffusion --time_denoise 0.2 [--gui]
python scripts/06_demodiffusion_sim.py --task hocap_0000_G18_3 --mode pi0
bash scripts/07_batch.sh                                          # 5 task x 3 mode (~15 phút, mỗi lần chạy ~1 phút)
```

Chế độ `retarget` không gọi Pi-0, nên dùng để kiểm tra cài đặt Isaac Lab trước.
Ra: `outputs/demodiffusion/runs/<task>/<mode>/{demo_video.mp4, video.mp4, metrics.json, rollout.npz}`.
Lần đầu `--gui` mất 1–2 phút nạp shader. Dừng server Pi-0 (Ctrl+C) sau khi chạy xong để trả VRAM.

## 5. Cấu trúc repo

```
env.sh, env_lab.sh          biến môi trường (Isaac Sim pip / Isaac Lab)
configs/object_names.json   tên vật theo mã HO-Cap (G18_3 = shoe, ...), tự đặt từ ảnh render
src/hocap_palm/data.py      đọc gói task, chuẩn hoá HO-Cap/HOI4D về world z lên, mặt bàn z = 0
src/hocap_palm/palm.py      retarget sang PALM, pose link, skin liền
scripts/01_retarget.py      keypoint MANO -> 22 góc khớp PALM
scripts/02_build_usd.py     scene USD mỗi sequence
scripts/03_render.py        render headless -> mp4
scripts/04_view.py          xem trong GUI
scripts/05_to_demodiffusion.py  cắt đoạn cầm-đặt -> task DemoDiffusion + USD vật có vật lý (assets_usd/hocap/)
scripts/06_demodiffusion_sim.py mô phỏng UR5e + 2F-85, D435, gọi Pi-0
scripts/07_batch.sh         chạy 5 task x 3 mode
outputs/demodiffusion/summary.md  kết quả lần chạy 2026-11-14
```

## 6. Trong mỗi scene USD (phần 1)

| Prim | Nội dung |
|---|---|
| `/World/Objects/<id>` | Mesh vật của dataset, pose theo annotation (frame không hợp lệ giữ pose trước). Kinematic rigid body + collision convex hull |
| `/World/PalmHand_<side>/skin` | Da liền của PALM (`skin.json`, skinning theo link), phần nhìn thấy |
| `/World/PalmHand_<side>/<link>` | Mesh từng link PALM (ẩn), kinematic rigid body + collision, như `isaac_scene.py` |
| `/World/PalmHand_<side>` attr `palm:joint_positions` | 22 góc khớp mỗi frame, thứ tự `HAND_NAMES` (lệnh `/palm/hand/joint_states`) |
| `/World/Reference/MANO_<side>` | Tay MANO gốc, xanh trong suốt, **ẩn sẵn** (bật con mắt trong Stage để so) |
| `/World/Cameras/*` | 8 RealSense đã hiệu chuẩn + `hololens_pv` (ego, theo frame); HOI4D: `hoi4d_cam` |
| `/World/Table` | Hộp xấp xỉ, mặt = điểm thấp nhất của vật ở frame 0 (dataset không có mesh bàn) |

Metadata (dataset, task, commit PALM, sai số retarget) nằm trong `customLayerData["hocap_palm"]` của layer.

### Quy ước

- World z lên, mét. HO-Cap: giữ nguyên world tag_1 (bàn z ≈ 0). HOI4D: pose theo camera, đã chuyển sang
  world camera frame 0 (log Open3D = world_from_camera), **hướng lên ước lượng** = trục +z của vật ở frame 0,
  mặt bàn z = 0.
- Retarget: 21 keypoint dataset (thứ tự MediaPipe) → `HandKinematics.fit` → 21 góc + pose lòng bàn tay 6 bậc.
  `wrist_yaw` = 0 và `wrist_pitch` = 0; toàn bộ hướng cổ tay nằm trong pose gốc. Cẳng tay gắn cứng theo bàn tay;
  cánh tay trên bị cắt (dataset không có tay/vai).
- Replay **kinematic**: tay và vật đi theo dữ liệu, không có lực. Đã kiểm: bấm Play thì vị trí PhysX trùng
  animation (sai 0,00 mm ở 4 mốc, sequence 0009).

### Kết quả đã kiểm (2026-11-14)

- Retarget 65 tay / 70 514 frame: mọi frame fit được (RMS ≤ 20 mm, ngưỡng của bridge). RMS trung bình 3,5 mm
  (2,4–4,9 mm theo sequence), p95 lớn nhất 8,8 mm; sai số đầu ngón trung bình 6,4 mm (lớn nhất 11,6 mm).
- Render headless: 0000 (1 tay trái), 0009 (2 tay, video đủ 485 frame), 0050 (HOI4D): tay PALM cầm đúng vật,
  trùng tay MANO khi bật tham chiếu.

### Chưa làm

- Chưa có kiểm tra xuyên vật/tiếp xúc: da PALM khác da MANO nên có thể xuyên hoặc hở vài mm ở chỗ cầm.
- Chưa replay vật lý tay PALM (gantry/drive + vật dynamic).

## 7. DemoDiffusion: chi tiết (phần 2)

Pipeline lấy từ `hot3d_ws_sim` (Franka "bóng" cho Pi-0 DROID, IK UR5e, sửa asset gripper). Repo đó không bị sửa.

- **Một task = một lần cầm–đặt một vật**, tên `hocap_<sequence>_<mã vật>`. Đoạn chạm = đỉnh MANO cách bề mặt vật
  < 1 cm, gộp khoảng hở < 15 frame, thêm 45 frame trước và 30 frame sau; 30 fps → 15 Hz.
- **Input DemoDiffusion** (`human_data/hocap_<NNNN>_<obj>/traj_0/processed_3d/`):
  `righthand_3d_keypoints.npy` (T,21,3) mét, frame đế robot, 15 Hz, đúng định dạng README gốc. Keypoint lấy từ
  annotation HO-Cap (đã triangulate từ 8 camera thật của dataset), không chạy HaMeR trên video sim.
  Kèm `eef_pose.npy`/`retarget_gripper_action.npy` (như `preprocess.py` gốc), `tcp_pose.npy`/`gripper_touch.npy`
  cho UR5e, `time_warp_src.npy` (bước robot ↔ bước demo), `demo_obj_pose.npy`, `demo_palm_skin.npz`, và `meta.json`
  (bàn, đế robot, vật + USD, prompt, đường đi GT của vật).
- **Mỗi 8 bước Pi-0 nhận**: ảnh D435 ngoài `cam1` + ảnh D435 cổ tay (thời gian thực, 224×224), 7 khớp Franka
  bóng, độ đóng gripper, prompt, 10 hành động retarget, `time` = 0.2.
- **Camera**: asset D435 `completed435ros.usd` (không sửa; ActionGraph ROS, collision tắt và clipping đổi ở layer
  stage). 3 camera ngoài (cam1, cam2 cùng cạnh dài, cam3 đối diện, cao 0,5 m trên bàn) + 1 D435 trên cổ tay.
- **TCP cho 2F-85** (`plan_tcp` trong 05): chọn tư thế kẹp trên mesh vật lúc chạm (họ theo tay + họ từ trên xuống;
  loại nếu pad/thân gripper chạm bàn, chỗ kẹp < 1,5 hoặc > 8 cm, vật vào sâu > 2 cm sau TCP); tới trước 4 bước rồi
  đóng; khi cầm TCP gắn theo pose GT của vật; nhả: giữ 8 bước rồi rút 12 cm dọc trục tiếp cận.
- **Chấm điểm** (`metrics.json`): so đường đi vị trí (x, y, z) của vật trong sim với GT: độ cao nhấc, lệch vị trí cuối,
  vật rơi khỏi bàn; thành công = nhấc ≥ 50% GT, lệch cuối < 5 cm, không rơi. Hướng vật chưa được chấm.
- Bàn: mặt z = 0, cao 0,75 m (HO-Cap không cho kích thước). Khối lượng vật = thể tích mesh × 400 kg/m³ (ước lượng).

### Kết quả 2026-11-14 ([summary.md](outputs/demodiffusion/summary.md))

0/5 task thành công ở cả 3 mode. `retarget` nhấc đúng độ cao 4/5 vật nhưng vật dính vào ngón 2F-85 khi nhả hoặc đổ;
DemoDiffusion chỉ nhấc được giày (Pi-0 lệch demo 2–3 cm, 15–20°); Pi-0 thuần lệch 21–35 cm.

Lỗi đã biết: vật dính ngón khi nhả; thân gripper che nửa trên ảnh camera cổ tay; bọt biển (2,5 cm) không kẹp được;
hộp carton đổ trong task giày.
