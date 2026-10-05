#!/bin/bash
# =====================================================================
#  GPU 대여 실행 스크립트 — 이 파일 하나만 돌리면 됩니다
# =====================================================================
#
#  받는 분께
#    NVIDIA GPU가 있는 리눅스/WSL에서 아래 한 줄이면 끝납니다.
#
#        bash run_on_gpu.sh
#
#    끝나면 results_to_return/ 폴더가 생깁니다. 그 폴더만 압축해서
#    돌려주시면 됩니다 (전부 합쳐 100MB 이하).
#
#  필요한 것
#    - NVIDIA GPU (VRAM 8GB 이상 권장. T4/3060 이상이면 충분)
#    - python 3.9+ 와 pip
#    - data/windows_2s.npz (306MB) — 같이 받으신 파일을 data/ 아래 두세요
#
#  걸리는 시간
#    우선순위 1만  약 1시간   <- 이것만 해주셔도 큰 도움입니다
#    전체         약 3시간
#
#  중간에 끊겨도 괜찮습니다
#    이미 만들어진 체크포인트는 건너뜁니다. 다시 실행하면 이어서 합니다.
# =====================================================================

set -u
cd "$(dirname "$0")"
OUT=results_to_return
mkdir -p results "$OUT"
LOG="$OUT/run_log.txt"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ── 사전 점검 ────────────────────────────────────────────────────────
say "환경 점검"
python -c "
import sys, torch
print('  python', sys.version.split()[0])
print('  torch ', torch.__version__)
ok = torch.cuda.is_available()
print('  GPU   ', torch.cuda.get_device_name(0) if ok else '없음 (CPU로는 매우 오래 걸립니다)')
" 2>&1 | tee -a "$LOG"

if [ ! -f data/windows_2s.npz ]; then
  say "!! data/windows_2s.npz 가 없습니다. 같이 받으신 npz 파일을 data/ 아래 두고 다시 실행해 주세요."
  exit 1
fi
# 데이터가 우리 것과 '완전히 같은 파일'인지 확인한다.
#
#   원본 NewHandPD에서 윈도우를 다시 만들면 안 된다. 이 팀의 전처리 코드가
#   남아 있지 않아 윈도우 분할이 비트 단위로 재현되지 않는다(0.6% 차이).
#   다른 npz로 돌리면 fold 분할과 test_subjects 짝맞춤이 어긋나 결과가
#   조용히 달라진다. 그래서 md5가 다르면 여기서 멈춘다.
python - <<'PY' 2>&1 | tee -a "$LOG"
import sys, numpy as np, hashlib
EXPECT = 'e8c887ef62098cb7384fbf0e74ef4708'
h = hashlib.md5(open('data/windows_2s.npz','rb').read()).hexdigest()
print('  데이터 md5', h)
if h != EXPECT:
    print('  !! 기대값과 다릅니다:', EXPECT)
    print('  !! 원본에서 새로 만든 npz가 아니라, 팀이 보낸 그 파일이어야 합니다.')
    print('  !! 이 상태로 돌리면 결과를 우리 것과 나란히 놓을 수 없습니다.')
    sys.exit(1)
d = np.load('data/windows_2s.npz', allow_pickle=True)
print('  shape', {k: d[k].shape for k in d.files})
print('  데이터 확인 완료')
PY
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  say '!! 데이터 확인 실패 - 중단합니다. 위 메시지를 팀에 알려주세요.'
  exit 1
fi

pip install -q numpy scipy scikit-learn matplotlib pyyaml 2>&1 | tail -2

CFG="--config configs/protocol.yaml"

# 체크포인트가 이미 있으면 학습을 건너뛴다 (중단 후 재개용)
have(){ ls results/ckpt_$1_f4_s$2_* >/dev/null 2>&1; }

train(){  # train <suite> <config> <seed>
  if have "$2" "$3"; then say "  건너뜀 (이미 있음): $2 seed $3"; return; fi
  say "  학습: $2 seed $3"
  python -u evaluate.py $CFG --suite "$1" --configs "$2" --seed "$3" \
    --save-ckpt --save-probs >> "$LOG" 2>&1 \
    || say "  !! 실패: $2 seed $3"
}

xai(){   # xai <config> <seed>
  say "  occlusion: $1 seed $2"
  python -u occlusion.py $CFG --configs "$1" --seed "$2" \
    --win-ms 500,1000 --occ-batch-size 64 >> "$LOG" 2>&1 || return
  for B in 1,3 2,4 4,6 6,8 8,12 12,20; do
    python -u bandpower_test.py --configs "$1" --seed "$2" --win-ms 1000 \
      --n-perm 1000 --importance abs --band "$B" >> "$LOG" 2>&1
  done
}

# ── 우선순위 1 : 패치 길이 통제 실험 (약 1시간) ──────────────────────
#   폭·깊이를 고정하고 패치 길이만 바꾼 두 짝을 학습해 비교합니다.
#   이 팀 논문에서 가장 중요한 미해결 질문입니다.
say ""
say "=== [1/3] 패치 길이 통제 실험 — 가장 중요 ==="
for C in d64L2p16 p100d128L1; do
  train patchctl "$C" 42
  xai "$C" 42
done

# ── 우선순위 2 : p16d128L1 시드 1·7 (약 1시간) ───────────────────────
say ""
say "=== [2/3] p16d128L1 시드 1·7 ==="
for S in 1 7; do
  train depth p16d128L1 "$S"
  xai p16d128L1 "$S"
done

# ── 우선순위 3 : d64L2p16·p100d128L1 시드 1·7 (약 1시간) ─────────────
say ""
say "=== [3/3] 통제 실험 시드 확장 ==="
for C in d64L2p16 p100d128L1; do
  for S in 1 7; do
    train patchctl "$C" "$S"
    xai "$C" "$S"
  done
done

# ── 돌려받을 것만 모으기 ─────────────────────────────────────────────
say ""
say "결과 정리"
cp -f results/bandpower_*.csv          "$OUT/" 2>/dev/null
cp -f results/occlusion_*_summary*.csv "$OUT/" 2>/dev/null
cp -f results/pareto_*.csv results/folds_*.csv "$OUT/" 2>/dev/null
cp -f results/ckpt_*.pt                "$OUT/" 2>/dev/null   # 체크포인트는 작습니다 (개당 1MB 미만)
say "완료. $OUT 폴더를 압축해 돌려주시면 됩니다."
du -sh "$OUT" 2>/dev/null | tee -a "$LOG"
ls "$OUT" | wc -l | xargs -I{} say "파일 {}개"
