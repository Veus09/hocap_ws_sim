# source env_lab.sh — venv Isaac Sim 6.0 + Isaac Lab 3.0b2 (chung với hot3d_ws_sim) cho script 06/07 (UR5e + DemoDiffusion)
# Ghi đè được: HOT3D_VENV, HOT3D_WS, DEMODIFFUSION_DIR, D435_USD (xem README, mục Cài đặt)
source "${HOT3D_VENV:-$HOME/fall26_envtest/.venv}/bin/activate"
export HOCAP_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HOT3D_WS="${HOT3D_WS:-$HOME/hot3d_ws_sim}"
export DEMODIFFUSION_DIR="${DEMODIFFUSION_DIR:-$HOME/demodiffusion}"
export D435_USD="${D435_USD:-$HOME/d435_import/d435_1/completed435ros.usd}"
export PYTHONPATH="$HOT3D_WS/src:$HOCAP_WS/src:${PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
