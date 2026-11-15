# hocap_ws_sim — task HO-Cap/HOI4D trong Isaac Sim với bàn tay PALM

Đưa 51 sequence trong `Hocap_tasks.zip` (50 HO-Cap + 1 HOI4D) vào Isaac Sim 6.0. Bàn tay MANO của dataset được thay bằng
**bàn tay PALM 22 khớp** (repo `binboong/palm`), retarget bằng chính `palm_hand_tracking.kinematics.HandKinematics.fit`
của repo PALM (giống bridge HandMoCap). Mỗi sequence thành một scene USD: vật và tay PALM chuyển động theo dữ liệu,
kèm các camera đã hiệu chuẩn của dataset.

Repo **chỉ có code và tài liệu**. Dữ liệu và scene USD sinh ra **không** nằm trong git
(xem [Dữ liệu](#2-dữ-liệu-không-có-trong-repo)); chạy các script bên dưới để sinh lại trên máy mình.

---

## 1. Yêu cầu

| Thành phần | Phiên bản đã dùng |
|---|---|
| Ubuntu | 24.04 (22.04 cũng được) |
| GPU NVIDIA | RTX, ≥ 8 GB VRAM |
| Ổ đĩa | ~1 GB dữ liệu giải nén + 2,3 GB scene USD |
| Isaac Sim | 6.0.0.1 (pip), Python 3.12 |
| ffmpeg | cho `03_render.py` (`sudo apt install ffmpeg`) |

Repo phụ thuộc duy nhất:

| Repo | Dùng cho | Đường dẫn mặc định | Biến môi trường | Quyền truy cập |
|---|---|---|---|---|
| `git@github.com:binboong/palm.git` | Động học `HandKinematics`, mesh link và skin tay PALM | `~/palm_ws` | `PALM_WS` | **private**, xin quyền chủ repo |

Chỉ cần mã nguồn PALM (`src/palm_hand_tracking`, `src/palm_description`), **không cần build ROS**. Repo PALM không bị sửa.
Commit đã kiểm: palm `c3c6758` (2026-11-14).

## 2. Dữ liệu (không có trong repo)

`Hocap_tasks.zip` (~725 MB) là gói task HO-Cap/HOI4D do nhóm chuẩn bị, định dạng `palm-handoff/1`.
**Xin file này từ nhóm.** Nó chứa dữ liệu HO-Cap/HOI4D và mesh MANO, phải tuân theo license gốc của các dataset đó nên
không phân phối công khai. Scene USD sinh ra cũng không đưa lên vì có hình học tay PALM (repo PALM là private).

Mỗi sequence (`sequences/<NNNN>/`) có:

| File | Nội dung |
|---|---|
| `motion.npz` → `hand_joints` (N, H, 21, 3) | 21 khớp tay 3D mỗi frame, mét |
| `hand_vertices`, `hand_faces`, `mano_params`, `mano_betas` | Mesh và tham số tay MANO |
| `object_poses` (N, K, 4, 4), `object_valid` | Pose 6 bậc tự do (vị trí + hướng) của K vật mỗi frame |
| `camera_world_from_camera`, `camera_pv_poses` | 8 RealSense đã hiệu chuẩn, kính HoloLens theo frame |
| `meshes/` | Mesh 3D các vật |
| `sources/` | File gốc của dataset, intrinsics/extrinsics camera |
| `manifest.json` | Dataset, sequence, task, danh sách vật, tay |

Gói không có video RGB, chỉ có quỹ đạo 3D. HO-Cap ghi ở 30 fps.

## 3. Cài đặt từng bước

### 3.1 Clone

```bash
cd ~
git clone git@github.com:Veus09/hocap_ws_sim.git
git clone git@github.com:binboong/palm.git palm_ws      # cần quyền
```

### 3.2 Môi trường Isaac Sim 6

```bash
python3.12 -m venv ~/env_isaacsim6
source ~/env_isaacsim6/bin/activate
pip install --upgrade pip
pip install "isaacsim[all,extscache]==6.0.0.1" --extra-index-url https://pypi.nvidia.com
pip install scipy pillow
deactivate
```

Lần đầu chạy Isaac Sim phải đồng ý EULA; `env.sh` đặt sẵn `OMNI_KIT_ACCEPT_EULA=YES`.
Nếu lệnh `pip install isaacsim` đổi theo bản mới, làm theo hướng dẫn cài pip của NVIDIA cho Isaac Sim 6.0.

### 3.3 Giải nén dữ liệu

```bash
cd ~/hocap_ws_sim
mkdir -p data/hocap_tasks
unzip -q /đường/dẫn/Hocap_tasks.zip -d data/hocap_tasks      # -> data/hocap_tasks/collection_<timestamp>/sequences/0000..0050
```

### 3.4 Đường dẫn khác mặc định

Đặt **trước** khi `source env.sh`:

```bash
export PALM_WS=/opt/palm_ws ISAAC_PYTHON=/opt/env_isaacsim6/bin/python
```

## 4. Chạy

```bash
cd ~/hocap_ws_sim && source env.sh
$ISAAC_PYTHON -I scripts/01_retarget.py                # ~3 phút, 12 tiến trình (--jobs) -> outputs/retarget/<NNNN>.npz + summary.csv
$ISAAC_PYTHON -I scripts/02_build_usd.py               # ~2 phút -> outputs/usd/<NNNN>.usd + index.csv (2,3 GB)
$ISAAC_PYTHON scripts/03_render.py --seq 0009 [--mano] # video kiểm tra headless -> outputs/render/0009.mp4
$ISAAC_PYTHON scripts/04_view.py --seq 0009 [--mano] [--cam 043422252387|hololens_pv] [--paused]   # xem trực tiếp (GUI)
```

- Một vài sequence: `01_retarget.py --seq 0009 0010`, `02_build_usd.py --seq 0009 0010`.
- `02_build_usd.py --no-mano`: bỏ mesh MANO tham chiếu cho file nhỏ hơn.
- Mở thủ công: Isaac Sim > File > Open `outputs/usd/<NNNN>.usd`, bấm Play (30 fps) hoặc kéo timeline.
- `outputs/usd/index.csv` liệt kê sequence → task, vật, tay.
- `--cam` của 03/04 nhận serial RealSense (xem `/World/Cameras/` trong scene) hoặc `hololens_pv`.

**Kiểm tra cài đúng:** `outputs/retarget/summary.csv` có 65 tay, RMS trung bình ~3,5 mm; `04_view.py --seq 0000`
hiện tay PALM trái cầm giày.

## 5. Cấu trúc repo

```
env.sh                      biến môi trường (ISAAC_PYTHON, PALM_WS, EULA)
configs/object_names.json   tên vật theo mã HO-Cap (G18_3 = shoe, ...), tự đặt từ ảnh render
src/hocap_palm/data.py      đọc gói task, chuẩn hoá HO-Cap/HOI4D về world z lên, mặt bàn z = 0
src/hocap_palm/palm.py      retarget sang PALM, pose link, skin liền
scripts/01_retarget.py      keypoint MANO -> 22 góc khớp PALM + pose lòng bàn tay
scripts/02_build_usd.py     scene USD mỗi sequence
scripts/03_render.py        render headless -> mp4
scripts/04_view.py          xem trong GUI
data/                       (git bỏ qua) giải nén Hocap_tasks.zip vào đây
outputs/                    (git bỏ qua) kết quả sinh ra
```

## 6. Trong mỗi scene USD

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

## 7. Quy ước

- World z lên, mét. HO-Cap: giữ nguyên world tag_1 (bàn z ≈ 0). HOI4D: pose theo camera, đã chuyển sang
  world camera frame 0 (log Open3D = world_from_camera), **hướng lên ước lượng** = trục +z của vật ở frame 0,
  mặt bàn z = 0.
- Retarget: 21 keypoint dataset (thứ tự MediaPipe) → `HandKinematics.fit` → 21 góc + pose lòng bàn tay 6 bậc.
  `wrist_yaw` = 0 và `wrist_pitch` = 0; toàn bộ hướng cổ tay nằm trong pose gốc. Cẳng tay gắn cứng theo bàn tay;
  cánh tay trên bị cắt (dataset không có tay/vai). Frame fit hỏng (RMS > 20 mm) giữ kết quả frame tốt trước đó.
- Replay **kinematic**: tay và vật đi theo dữ liệu, không có lực. Đã kiểm: bấm Play thì vị trí PhysX trùng
  animation (sai 0,00 mm ở 4 mốc, sequence 0009).

## 8. Kết quả đã kiểm (2026-11-14)

- Retarget 65 tay / 70 514 frame: mọi frame fit được (RMS ≤ 20 mm, ngưỡng của bridge). RMS trung bình 3,5 mm
  (2,4–4,9 mm theo sequence), p95 lớn nhất 8,8 mm; sai số đầu ngón trung bình 6,4 mm (lớn nhất 11,6 mm).
- Render headless: 0000 (1 tay trái), 0009 (2 tay, video đủ 485 frame), 0050 (HOI4D): tay PALM cầm đúng vật,
  trùng tay MANO khi bật tham chiếu.

## 9. Chưa làm

- Chưa có kiểm tra xuyên vật/tiếp xúc: da PALM khác da MANO nên có thể xuyên hoặc hở vài mm ở chỗ cầm.
- Chưa replay vật lý tay PALM (gantry/drive + vật dynamic); mass, ma sát chưa có trong dữ liệu.
- Chưa render video cho cả 51 sequence (mới 3).
