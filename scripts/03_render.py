"""Render a built HO-Cap/PALM USD in Isaac Sim (headless RTX) to check it: left = a calibrated RealSense view
(HOI4D: its head camera), right = HoloLens PV (ego, HO-Cap only). Scrubs the timeline; no physics stepping.

  ~/env_isaacsim6/bin/python scripts/03_render.py --seq 0000 [--cam 043422252387] [--step 2] [--mano] [--max 0]
-> outputs/render/<NNNN>[_mano]/*.png and outputs/render/<NNNN>[_mano].mp4
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--seq', default='0000')
ap.add_argument('--cam', default='043422252387', help='HO-Cap RealSense serial for the left view (else first camera)')
ap.add_argument('--step', type=int, default=2)
ap.add_argument('--max', type=int, default=0, help='max rendered frames (0 = all)')
ap.add_argument('--mano', action='store_true', help='show the dataset MANO hand (blue, transparent) as reference')
ap.add_argument('--height', type=int, default=480)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting'})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import UsdGeom  # noqa: E402

ws = Path(__file__).resolve().parents[1]
usd = ws / f'outputs/usd/{args.seq}.usd'
if not usd.exists():
    sys.exit(f'{usd} missing: run scripts/02_build_usd.py --seq {args.seq}')
tag = args.seq + ('_mano' if args.mano else '')
out = ws / 'outputs/render' / tag
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True)

ctx = omni.usd.get_context()
ctx.open_stage(str(usd))
for _ in range(20):
    app.update()
stage = ctx.get_stage()
if args.mano:
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith('/World/Reference/'):
            UsdGeom.Imageable(prim).MakeVisible()
cameras = [p for p in stage.GetPrimAtPath('/World/Cameras').GetChildren()]
names = [p.GetName() for p in cameras]
first = f'cam_{args.cam}' if f'cam_{args.cam}' in names else names[0]
views = [f'/World/Cameras/{first}'] + (['/World/Cameras/hololens_pv'] if 'hololens_pv' in names else [])
sizes = {}
for view in views:
    cam = UsdGeom.Camera(stage.GetPrimAtPath(view))
    aspect = cam.GetHorizontalApertureAttr().Get() / cam.GetVerticalApertureAttr().Get()
    sizes[view] = (int(round(args.height * aspect / 2)) * 2, args.height)
annotators = []
for view in views:
    rp = rep.create.render_product(view, sizes[view])
    annotator = rep.AnnotatorRegistry.get_annotator('rgb')
    annotator.attach(rp)
    annotators.append(annotator)

timeline = omni.timeline.get_timeline_interface()
fps = stage.GetTimeCodesPerSecond()
frames = list(range(int(stage.GetStartTimeCode()), int(stage.GetEndTimeCode()) + 1, args.step))
if args.max:
    frames = frames[:args.max]
for k, t in enumerate(frames):
    timeline.set_current_time(t / fps)
    rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
    images = [np.asarray(a.get_data())[..., :3] for a in annotators]
    Image.fromarray(np.concatenate(images, axis=1)).save(out / f'{k:05d}.png')
    if k % 50 == 0:
        print(f'frame {t} ({k + 1}/{len(frames)})', flush=True)

video = ws / 'outputs/render' / f'{tag}.mp4'
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', f'{fps / args.step:g}', '-i', str(out / '%05d.png'),
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], check=True)
print('HOCAP_RENDER_DONE', video, len(frames), 'frames', flush=True)
app.close()
