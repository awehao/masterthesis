#!/usr/bin/env bash
# 全身協調預抓取：底盤邊移動、手臂邊展開，最後夾爪停在箱面前方。
#
# 四個終端機分開跑，這樣任何一段掛掉都看得出來是哪一段。
# 每一步都等前一步的訊息出現再往下。
set -e
cd "$(dirname "$0")/.."
. /opt/ros/jazzy/setup.bash
. install/setup.bash

case "${1:-help}" in

sim)    # 終端 1：Gazebo（GUI 必須開，headless 下關節不動，原因未定位）
        exec ros2 launch my_omnibot_description arm_barrier_test.launch.py gui:=true
        ;;

stack)  # 終端 2：安全層（--free-base = 底盤進入解算、live TF、橋接 /cmd_vel）
        exec python3 evaluation/barrier_stack.py --free-base
        ;;

reset)  # 終端 3：每次執行前復位。中止時 sim 會保留當下姿態，不會自己回去。
        gz service -s /world/arm_barrier_test/set_pose \
          --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 5000 \
          --req 'name: "omni_bot", position: {x: -0.70, y: 0.0, z: 0.05},
                 orientation: {x:0, y:0, z:0, w:1}'
        sleep 5
        exec python3 evaluation/barrier_probe.py --home 0 0 0.30 0 0 0 --home-only \
             --out /tmp/wb_home.json
        ;;

run)    # 終端 3：跑。目標是世界座標；底盤停在哪裡是輸出。
        exec python3 evaluation/wholebody_pregrasp.py --target 0.30 0.0 0.55
        ;;

rec)    # 終端 4（選用）：錄影。來源是世界檔裡的相機感測器，不是螢幕。
        exec python3 evaluation/record_gz_camera.py \
             --topic /demo_cam --out evaluation/results/wholebody_pregrasp.mp4
        ;;

*)      cat <<'USAGE'
  用法：evaluation/run_wholebody_demo.sh <步驟>

    sim     終端 1  開 Gazebo，等控制器 active（約 60 s）
    stack   終端 2  開安全層，等 "certified sampling" 那行出現（約 40 s）
    reset   終端 3  底盤退到 x=-0.70、手臂歸位，等 "到位"
    run     終端 3  執行；要錄影就先在終端 4 開 rec，跑完 Ctrl-C 停錄
    rec     終端 4  錄影（可省略）

  要重跑：再做一次 reset 然後 run。
USAGE
        ;;
esac
