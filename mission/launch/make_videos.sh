#!/usr/bin/env bash
# Isaac 趟次的俯視＋跟車實錄重播影片（也會開 Isaac，所以不能和實跑同時）。約 15–20 分鐘。
# 用法：mission/launch/make_videos.sh <N> <成功|失敗> "<內容描述>"
# 輸出：第八次進度報告素材/影片/<日期>_<時間>_<成敗>_<描述>_{top,follow}_實錄重播.mp4
source "$(dirname "$0")/_env.sh"
N=${1:?趟號}; OK=${2:?成功或失敗}; DESC=${3:?描述}
pgrep -f "[i]saac_m8_house" >/dev/null && { echo "已有 Isaac 在跑"; exit 1; }
RUN=m8house_d2_unified_$N; RD=$WS/evaluation/runs/$RUN; OUT=$MISSION/logs/replay_$N
[ -d "$RD" ] || { echo "找不到 $RD"; exit 1; }
rm -rf "$OUT"
load_env task.env
cd "$WS"
REPLAY_VIEWS=top,follow ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_d2_replay.py --run $RUN --out "$OUT" > "$MISSION/logs/replay_$N.log" 2>&1 || { echo "重播失敗，看 mission/logs/replay_$N.log"; exit 1; }
STAMP=$(date -r "$RD/initpose.log" +%Y%m%d_%H%M%S 2>/dev/null || date +%Y%m%d_%H%M%S)
V=$WS/第八次進度報告素材/影片; mkdir -p "$V"
for v in top follow; do nice -n 5 ffmpeg -loglevel error -y -framerate 10 -i "$OUT/${v}_%05d.png" -c:v libx264 -pix_fmt yuv420p -crf 23 "$V/${STAMP}_${OK}_${DESC}_${v}_實錄重播.mp4"; done
ls -la "$V" | grep "$STAMP"
