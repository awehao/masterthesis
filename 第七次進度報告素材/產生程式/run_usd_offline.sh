#!/usr/bin/env bash
# 不啟動 Isaac Kit／模擬器，只用 Isaac 內建的 USD 函式庫（omni.usd.libs）＋ PhysX schema 在記憶體建／讀 USD 舞台。
#   bash evaluation/run_usd_offline.sh evaluation/door_k0_readback.py
set -eu
VENV="${ISAAC_VENV:-$HOME/venvs/isaacsim-6.0.1}"
X="$VENV/lib/python3.12/site-packages/isaacsim/extscache"
U=$(ls -d "$X"/omni.usd.libs-1.0.3* | head -1)
E=$(ls -d "$X"/omni.usd.schema.physx-110* | head -1)
export PYTHONPATH="$U:$E${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$U/bin:$E/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export USD_OFFLINE_PHYSX_PLUGINS="$E/plugins"
exec "$VENV/bin/python" "$@"
