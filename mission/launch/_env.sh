# 共用：路徑、ROS 環境、讀設定檔（被其他啟動腳本 source，不單獨執行）
MISSION=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WS=$(dirname "$MISSION")
D2OUT=$WS/evaluation/results/motm_direct/d2          # 替身結果
unset VIRTUAL_ENV                                     # 避免 .venv-dl0 等虛擬環境搶走 python3（無 osqp）
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
source /opt/ros/jazzy/setup.bash >/dev/null 2>&1
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash" >/dev/null 2>&1
load_env() { set -a; for f in "$@"; do source "$MISSION/config/$f"; done; set +a; }
apply_overrides() { for kv in "$@"; do case "$kv" in *=*) export "$kv";; *) echo "略過非 KEY=VAL 參數：$kv";; esac; done; }
