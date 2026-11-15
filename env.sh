# source env.sh — Isaac Sim 6.0 pip env (cùng env với palm_ws) cho script 01-05
# Ghi đè được: ISAAC_PYTHON, PALM_WS, HOT3D_WS, DEMODIFFUSION_DIR (xem README, mục Cài đặt)
export HOCAP_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/env_isaacsim6/bin/python}"
export OMNI_KIT_ACCEPT_EULA=YES
export PALM_WS="${PALM_WS:-$HOME/palm_ws}"
export HOT3D_WS="${HOT3D_WS:-$HOME/hot3d_ws_sim}"
export DEMODIFFUSION_DIR="${DEMODIFFUSION_DIR:-$HOME/demodiffusion}"
