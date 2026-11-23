#!/usr/bin/env bash
# 用法：stub_end_guard.sh <tag>：替身寫出 _stub.json 後 30 s 本趟執行腳本仍在 ⇒ 只對本趟執行腳本送 INT（其收尾 trap 只殺本趟程序群組）；不做全域 pkill（11/20：曾誤殺下一趟 s114d／s118d）
TAG=$1; D=/home/howardchen/masterthesis/evaluation/results/motm_direct/d2
until [ -f $D/${TAG}_stub.json ]; do sleep 5; done
sleep 30
P=$(pgrep -f "[r]un_d2_stub.sh evaluation/results/motm_direct/d2/${TAG} " | head -1)
if [ -n "$P" ]; then echo "guard: stub ended, run still alive -> INT $P"; kill -INT $P; fi
echo "guard done: $(env -u VIRTUAL_ENV /usr/bin/python3 -c "import json;r=json.load(open('$D/${TAG}_stub.json'));print(r['abort'], r['end'], r['stage'])")"
