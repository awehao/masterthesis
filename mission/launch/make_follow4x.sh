#!/usr/bin/env bash
# 只出跟車視角 4 倍速實錄重播（Howard 11/25「只要出四倍的跟車鏡頭不轉即可，不然太多影片」）。會開 Isaac，不能和實跑同時。約 8–12 分鐘。
# 用法：mission/launch/make_follow4x.sh <N> <成功|失敗> "<內容描述>"
# 輸出：第八次進度報告素材/影片/<日期>_<時間>_<成敗>_<描述>_follow_實錄重播_4倍速.mp4（原速影格只留在 mission/logs/replay_<N>/）
source "$(dirname "$0")/_env.sh"
N=${1:?趟號}; OK=${2:?成功或失敗}; DESC=${3:?描述}
pgrep -f "[i]saac_m8_house" >/dev/null && { echo "已有 Isaac 在跑"; exit 1; }
RUN=m8house_d2_unified_$N; RD=$WS/evaluation/runs/$RUN; OUT=$MISSION/logs/replay_$N
[ -d "$RD" ] || { echo "找不到 $RD"; exit 1; }
rm -rf "$OUT"
load_env task.env
cd "$WS"
bash "$MISSION/tools/clean_carb_shm.sh" > /dev/null 2>&1
REPLAY_VIEWS=follow ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_d2_replay.py --run $RUN --out "$OUT" > "$MISSION/logs/replay_$N.log" 2>&1 || { echo "重播失敗，看 mission/logs/replay_$N.log"; exit 1; }
STAMP=$(date -r "$RD/initpose.log" +%Y%m%d_%H%M%S 2>/dev/null || date +%Y%m%d_%H%M%S)
V=$WS/第八次進度報告素材/影片; mkdir -p "$V"
nice -n 5 ffmpeg -loglevel error -y -framerate 40 -i "$OUT/follow_%05d.png" -r 20 -c:v libx264 -pix_fmt yuv420p -crf 23 "$V/${STAMP}_${OK}_${DESC}_follow_實錄重播_4倍速.mp4"   # 影格 10 Hz 模擬時間 ⇒ 40 fps 播放＝4 倍速，輸出 20 fps
ls -la "$V" | grep "$STAMP"
