#!/usr/bin/env bash
# run_plan_batch.sh 的假運行器反例測試（不開 Isaac）。程序查找只看 PID 與環境標記，
# 不用 pgrep -f 樣式（避免比對到測試本身的指令列）。
#   bash evaluation/test_run_plan_batch.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
T=$(mktemp -d)
RUNS="$WS/evaluation/runs"
DOM=199
PASS=0; FAIL=0
ok(){ if [ "$1" = 0 ]; then echo "  ok    $2"; PASS=$((PASS+1)); else echo "  FAIL  $2"; FAIL=$((FAIL+1)); fi; }

cat > "$T/fake_runner.sh" <<'EOF'
#!/usr/bin/env bash
DIR=$RUNS_DIR/$RUN_ID
PIDS=(); trap 'kill ${PIDS[@]} 2>/dev/null' EXIT
sleep 300 & PIDS+=($!); echo "[x]   起 sim PID=${PIDS[0]}" >> $DIR/run.log
case "${FAKE_MODE:-ok}" in
  scene) exit 83 ;;
  handover) for f in align_solver.json task.json room_run.json mission.json wholebody.json; do echo '{}' > $DIR/$f; done; exit 1 ;;
esac
setsid sleep 301 < /dev/null > /dev/null 2>&1 &
for f in align_solver.json task.json mission.json wholebody.json; do echo '{}' > $DIR/$f; done
[ "${FAKE_MODE:-ok}" = noarch ] || echo '{}' > $DIR/room_run.json
sleep "${FAKE_T:-2}"
echo "=== 結束 rc=0 ===" >> $DIR/run.log
EOF
printf 'idx\trid\tcase\topen_m\tmethod\tN\tpair\torder\n01\t_tpb_01_T100_H5\tT100\t0.100\tH5\t5\tA\tx\n02\t_tpb_02_T100_H1\tT100\t0.100\tH1\t1\tA\tx\n' > "$T/plan.tsv"
echo aaa > "$T/target"; sha256sum "$T/target" > "$T/freeze"

tagged_alive(){  # 本測試批次標記的程序數
  local n=0 q
  for q in $(pgrep -u "$USER"); do
    tr '\0' '\n' < "/proc/$q/environ" 2>/dev/null | grep -q '^C1_RUN_TAG=_tpb:' && n=$((n+1))
  done
  echo "$n"
}
run_batch(){
  RUNS_DIR="$RUNS" RUNNER="$T/fake_runner.sh" PLAN="$T/plan.tsv" FREEZE="$T/freeze" \
    BATCH=_tpb DOMAIN=$DOM bash evaluation/run_plan_batch.sh > "$T/out" 2>&1
}
reset(){ rm -rf "$RUNS"/_tpb*; }
cls(){ python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['class'])" "$RUNS/$1.status.json" 2>/dev/null; }

echo "T1 正常兩趟、帶標記殘留被清"
reset; run_batch; rc=$?
ok $([ $rc = 0 ] && echo 0 || echo 1) "rc=0（實得 $rc）"
ok $([ "$(cls _tpb_01_T100_H5)" = executed ] && echo 0 || echo 1) "第一趟 class=executed"
ok $([ "$(tagged_alive)" = 0 ] && echo 0 || echo 1) "無本批次殘留程序"

echo "T2 父環境帶 domain（繼承）不誤判、不殺自己"
reset; ( export ROS_DOMAIN_ID=$DOM; run_batch ); rc=$?
ok $([ $rc = 0 ] && echo 0 || echo 1) "rc=0（實得 $rc）"

echo "T3 domain 上有外來程序 ⇒ 停批、不殺"
reset; ROS_DOMAIN_ID=$DOM setsid sleep 302 < /dev/null > /dev/null 2>&1 & FP=$!
sleep 0.5; FPID=$(pgrep -P $FP 2>/dev/null || echo $FP)
for q in $(pgrep -u "$USER" -x sleep); do
  tr '\0' '\n' < /proc/$q/environ 2>/dev/null | grep -qx "ROS_DOMAIN_ID=$DOM" && \
    [ "$(tr '\0' ' ' < /proc/$q/cmdline)" = "sleep 302 " ] && FPID=$q
done
run_batch; rc=$?
ok $([ $rc = 98 ] && echo 0 || echo 1) "rc=98（實得 $rc）"
ok $(kill -0 "$FPID" 2>/dev/null && echo 0 || echo 1) "外來程序仍在（未被殺）"
kill "$FPID" 2>/dev/null

echo "T4 場景建不起來（83）⇒ 停批、下一趟不起跑"
reset; FAKE_MODE=scene run_batch; rc=$?
ok $([ $rc = 97 ] && echo 0 || echo 1) "rc=97（實得 $rc）"
ok $([ "$(cls _tpb_01_T100_H5)" = startup_failure ] && echo 0 || echo 1) "class=startup_failure"
ok $([ ! -e "$RUNS/_tpb_02_T100_H1" ] && echo 0 || echo 1) "第二趟未起跑"

echo "T5 續跑遇到非 executed 趟次 ⇒ 拒絕"
RESUME_FROM=02 run_batch; rc=$?
ok $([ $rc = 96 ] && echo 0 || echo 1) "rc=96（實得 $rc）"

echo "T6 導航交棒失敗（運行器 1）⇒ 停批、交人工分類"
reset; FAKE_MODE=handover run_batch; rc=$?
ok $([ $rc = 97 ] && [ "$(cls _tpb_01_T100_H5)" = handover_failure ] && echo 0 || echo 1) "rc=97、class=handover_failure（實得 $rc／$(cls _tpb_01_T100_H5)）"

echo "T7 封存檔不齊 ⇒ missing_artifacts、停批"
reset; FAKE_MODE=noarch run_batch; rc=$?
ok $([ $rc = 97 ] && [ "$(cls _tpb_01_T100_H5)" = missing_artifacts ] && echo 0 || echo 1) "rc=97、class=missing_artifacts"

echo "T8 中途中斷 ⇒ 頻率取樣器收掉"
reset
( RUNS_DIR="$RUNS" RUNNER="$T/fake_runner.sh" PLAN="$T/plan.tsv" FREEZE="$T/freeze" FAKE_T=30 \
    BATCH=_tpb DOMAIN=$DOM exec bash evaluation/run_plan_batch.sh > "$T/out8" 2>&1 ) & BP=$!
sleep 6; kill -TERM "$BP" 2>/dev/null; sleep 5
SAMP=0
for q in $(pgrep -u "$USER" -P "$BP" 2>/dev/null); do SAMP=$((SAMP+1)); done
ok $([ $SAMP = 0 ] && echo 0 || echo 1) "批次程序無存活子程序（取樣器已收）"
ok $([ "$(tagged_alive)" = 0 ] && echo 0 || echo 1) "本趟帶標記的程序已收掉"
ok $([ "$(cls _tpb_01_T100_H5)" = interrupted ] && echo 0 || echo 1) "class=interrupted（實得 $(cls _tpb_01_T100_H5)）"
# 收掉本測試留下的帶標記程序（run_guarded 那一支在 T8 被中斷，不在本測試範圍）
for q in $(pgrep -u "$USER"); do
  tr '\0' '\n' < /proc/$q/environ 2>/dev/null | grep -q '^C1_RUN_TAG=_tpb:' && kill "$q" 2>/dev/null
done
sleep 1
reset; rm -rf "$T"
echo "$PASS 通過、$FAIL 失敗"
[ $FAIL = 0 ]
