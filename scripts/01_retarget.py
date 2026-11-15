"""Retarget HO-Cap hand keypoints (21, MediaPipe order) to the PALM hand, every sequence and hand side.

-> outputs/retarget/<NNNN>.npz with, per side in manifest order: <side>_q (N,22, PALM HAND_NAMES order),
   <side>_palm (N,4,4 world_from_palm), <side>_rms (N,), <side>_ok (N,), <side>_err (N,21 per-keypoint error m)
   + outputs/retarget/summary.csv

  python scripts/01_retarget.py [--seq 0000 0001 ...] [--jobs 12]
"""
import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hocap_palm import data, palm  # noqa: E402

OUT = data.WS / 'outputs/retarget'


def run(index: str) -> list[dict]:
    seq = data.scene(index)
    sides = seq['hand_sides']
    result, rows = {'joint_names': np.array(palm.joint_names(palm.hand('right'))), 'sides': np.array(sides)}, []
    for h, side in enumerate(sides):
        kin = palm.hand(side)
        joints, valid = seq['hand_joints'][:, h], seq['hand_valid'][:, h]
        fit = palm.retarget(kin, joints, valid)
        err = np.linalg.norm(palm.keypoints(kin, fit['q'], fit['palm']) - joints, axis=-1)
        err[~fit['ok']] = np.nan
        for key, value in fit.items():
            result[f'{side}_{key}'] = value
        result[f'{side}_err'] = err
        rows.append({'index': index, 'dataset': seq['manifest']['dataset'], 'sequence_id': seq['manifest']['sequence_id'],
                     'task': seq['manifest']['task'],
                     'side': side, 'frames': len(joints), 'hand_valid': int(valid.sum()), 'fit_ok': int(fit['ok'].sum()),
                     'rms_mm_mean': round(float(np.nanmean(fit['rms'])) * 1000, 2),
                     'rms_mm_p95': round(float(np.nanpercentile(fit['rms'], 95)) * 1000, 2),
                     'tip_err_mm_mean': round(float(np.nanmean(err[:, [4, 8, 12, 16, 20]])) * 1000, 2)})
    np.savez_compressed(OUT / f'{index}.npz', **result)
    print(index, ' '.join(f"{r['side']}:ok={r['fit_ok']}/{r['frames']} rms={r['rms_mm_mean']}mm" for r in rows), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', nargs='*', help='sequence folders (default: all in catalog)')
    ap.add_argument('--jobs', type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    indices = args.seq or [row['index'] for row in data.catalog()]
    with Pool(args.jobs) as pool:
        rows = [r for rs in pool.map(run, indices) for r in rs]
    summary = OUT / ('summary.csv' if not args.seq else 'summary_partial.csv')
    with summary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('wrote', summary)


if __name__ == '__main__':
    main()
