"""Open a HO-Cap/PALM scene in the Isaac Sim GUI and play it (loops). Close the window to quit.

  ~/env_isaacsim6/bin/python scripts/04_view.py --seq 0009 [--mano] [--cam 043422252387 | --cam hololens_pv] [--paused]
"""
import argparse
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--seq', default='0000')
ap.add_argument('--mano', action='store_true', help='also show the dataset MANO hand (blue, transparent)')
ap.add_argument('--cam', default='', help='view through a scene camera, e.g. 043422252387 or hololens_pv')
ap.add_argument('--paused', action='store_true', help='open without pressing Play')
args = ap.parse_args()

usd = Path(__file__).resolve().parents[1] / f'outputs/usd/{args.seq}.usd'
if not usd.exists():
    sys.exit(f'{usd} missing: run scripts/02_build_usd.py --seq {args.seq}')

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': False, 'width': 1600, 'height': 900})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402

ctx = omni.usd.get_context()
ctx.open_stage(str(usd))
for _ in range(30):
    app.update()
stage = ctx.get_stage()
if args.mano:
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith('/World/Reference/'):
            UsdGeom.Imageable(prim).MakeVisible()
if args.cam:
    from omni.kit.viewport.utility import get_active_viewport
    name = args.cam if args.cam.startswith(('hololens', 'hoi4d')) else f'cam_{args.cam}'
    get_active_viewport().camera_path = f'/World/Cameras/{name}'
else:
    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(eye=[0.9, -0.9, 0.8], target=[0.0, 0.0, 0.1])
timeline = omni.timeline.get_timeline_interface()
timeline.set_looping(True)
if not args.paused:
    timeline.play()
print(f'HOCAP_VIEW {args.seq}: {int(stage.GetEndTimeCode()) + 1} frames, 30 fps. Space = play/pause.', flush=True)
while app.is_running():
    app.update()
app.close()
