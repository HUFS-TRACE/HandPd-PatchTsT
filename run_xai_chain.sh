#!/bin/bash
# X1 -> X2 -> X3 자동 연결.
# 프로세스 감시가 아니라 산출물 감시로 대기한다 — 이 환경의 Git Bash에는
# pgrep이 없고, 있어도 네이티브 python.exe를 보지 못한다.
cd "C:/Users/USER/Desktop/학술제/patchtst_pd/HandPd-PatchTsT" || exit 1
CFG=${1:-d64L2}; SEED=${2:-42}
LOG=results/chain_${CFG}_s${SEED}.log
TRAINLOG=results/log_${CFG}_cpu.txt
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

say "학습 대기 시작 (체크포인트 5개 + 로그 정지 확인)"
STABLE=0
for i in $(seq 1 900); do        # 최대 15시간
  N=$(ls results/ckpt_${CFG}_f*_s${SEED}*.pt 2>/dev/null | wc -l)
  SZ=$(stat -c %s "$TRAINLOG" 2>/dev/null || echo 0)
  if [ "$N" -ge 5 ] && [ "$SZ" = "${PREV:-x}" ]; then
    STABLE=$((STABLE+1)); [ $STABLE -ge 2 ] && break
  else
    STABLE=0
  fi
  PREV=$SZ
  sleep 60
done
N=$(ls results/ckpt_${CFG}_f*_s${SEED}*.pt 2>/dev/null | wc -l)
say "체크포인트 ${N}개 확보"
[ "$N" -lt 5 ] && { say "부족 - 중단"; exit 1; }

say "X1 occlusion 시작"
python -u occlusion.py --config configs/protocol.yaml --configs $CFG --seed $SEED \
  --occ-batch-size 32 >> "$LOG" 2>&1 && say "X1 완료" || { say "X1 실패"; exit 1; }

say "X2 대역 파워 검정"
python -u bandpower_test.py --configs $CFG --seed $SEED --win-ms 500,1000 >> "$LOG" 2>&1 \
  && say "X2 완료" || say "X2 실패(계속)"

say "X3 그림"
for W in 200 500; do
  python -u plot_occlusion.py --configs $CFG --seed $SEED --win-ms $W >> "$LOG" 2>&1 \
    && say "X3 ${W}ms 완료" || say "X3 ${W}ms 실패(계속)"
done
say "=== 전체 종료 ==="
