#!/usr/bin/env bash
# M0 手臂模型資料蒐集（11/27 規劃）：三個構型區域 × 四段獨立激勵（train／validation／test／test）＝12 段，逐段各開一次 Isaac。
# 用法：run_m0.sh [批次名=m0]   區域中心可用環境變數覆蓋：M0_FOLDED／M0_MIDDLE／M0_EXTENDED（六關節，逗號分隔）
# 規則：一次只能一個 Isaac（同 run_guarded_v3.sh 的互斥鎖）；導航批次（gz）在跑時不開；CPU ≤55 °C 才起跑；92 °C 熱中止。
cd "$(dirname "$0")/.."
WS=$PWD
B=${1:-m0}
R=$WS/evaluation/results/$B; mkdir -p "$R"
LOG=$R/$B.log
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
FOLDED=${M0_FOLDED:--0.003,-1.202,0.721,0.006,0.527,-0.006}       # 穿門收臂（task.env PASS_Q_STOW）
MIDDLE=${M0_MIDDLE:-0.0,0.0,0.035,0.0,-1.5708,0.0}                 # K2_READY
EXTENDED=${M0_EXTENDED:?要給伸展區中心 M0_EXTENDED（Howard 選定的既有抓取姿勢）}
K=$(grep -l k10temp /sys/class/hwmon/*/name | head -1 | xargs dirname)
LOCK="${XDG_RUNTIME_DIR:-/tmp}/masterthesis-isaac-$(id -u).lock"
echo "$(date +%F_%T) batch $B folded=$FOLDED middle=$MIDDLE extended=$EXTENDED git=$(git rev-parse --short HEAD)" >> "$LOG"
SPLITS=(train validation test test)
RI=0
for REG in folded middle extended; do
  case $REG in folded) C=$FOLDED;; middle) C=$MIDDLE;; extended) C=$EXTENDED;; esac
  for k in 0 1 2 3; do
    [ -f "$R/stop" ] && { echo "$(date +%T) stop file" >> "$LOG"; exit 0; }
    SEED=$((1000 * (RI + 1) + k)); SP=${SPLITS[$k]}; OUT=$R/${REG}_r${k}
    [ -f "$OUT/m0.npz" ] && { echo "$(date +%T) $REG r$k 已有，略過" >> "$LOG"; continue; }
    while pgrep -f "[r]un_n2_batch.sh|[r]un_n1_r2.sh|[r]un_n1_batch.sh|[r]un_omnibot_dynamic.sh" > /dev/null; do sleep 30; done
    until [ $(( $(cat $K/temp1_input) / 1000 )) -le 55 ]; do sleep 15; done
    mkdir -p "$OUT"
    (
      flock -n 9 || { echo "$(date +%T) Isaac 互斥鎖忙碌 ⇒ $REG r$k 未啟動" >> "$LOG"; exit 95; }
      bash "$WS/mission/tools/clean_carb_shm.sh" > /dev/null 2>&1
      setsid systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 env -u VIRTUAL_ENV timeout 900 "$ISAAC_PY" -u evaluation/isaac_m0_excite.py \
        --out "$OUT" --region $REG --center="$C" --seed $SEED --split $SP > "$OUT/sim.log" 2>&1 < /dev/null &
      P=$!
      while kill -0 $P 2>/dev/null; do                                # 92 °C 熱中止
        if [ $(( $(cat $K/temp1_input) / 1000 )) -ge 92 ]; then echo "$(date +%T) 92 °C 熱中止 $REG r$k" >> "$LOG"; pkill -KILL -s $P 2>/dev/null; kill -KILL $P 2>/dev/null; fi
        sleep 2
      done
      wait $P; echo "rc=$?" >> "$OUT/sim.log"
    ) 9>"$LOCK"
    echo "$(date +%T) $REG r$k $SP seed $SEED : $(grep -E '^\[m0\]' "$OUT/sim.log" | tail -1)" >> "$LOG"
    ST=fail; [ -f "$OUT/m0.npz" ] && ST=done                           # 登記（鍵同規劃清單 M0/M/<區域>/recording0N/nominal_excitation）
    echo "M0/M/$REG/recording0$((k + 1))/nominal_excitation,$ST,$B,$OUT/m0.npz,$(date +%F_%T)" >> "$WS/evaluation/results/plan1127/registry.csv"
    sleep 20
  done
  RI=$((RI + 1))
done
echo "$(date +%F_%T) BATCH DONE $B" >> "$LOG"
