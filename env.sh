# source env.sh — Isaac Sim 6.0 pip env (cùng env với palm_ws)
# Ghi đè được (đặt trước khi source): ISAAC_PYTHON, PALM_WS
export HOCAP_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/env_isaacsim6/bin/python}"
export OMNI_KIT_ACCEPT_EULA=YES
export PALM_WS="${PALM_WS:-$HOME/palm_ws}"
