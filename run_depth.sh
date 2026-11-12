#!/bin/bash
# V1 깊이 축 재측정 — 팀 고정 프로토콜(08 §2)
for s in 42 1 7; do
  python -u evaluate.py --suite depth --seed $s --n-folds 5 \
    --save-probs > results/_depth_s$s.log 2>&1
  echo "seed $s 완료"
done
echo "깊이 축 전부 완료"
