"""여러 seed의 스윕 결과를 모아 평균±표준편차로 요약 + hard/soft voting 결론.

seed 하나의 결과는 fold 분할 운에 크게 흔들린다(피험자 61명, fold당 test 12명).
여러 seed를 평균내야 "설정이 좋은 것"과 "seed가 좋았던 것"을 구분할 수 있다.

실행: python summarize_seeds.py [glob패턴]
      기본 results/pareto_patch_2s*.csv
"""
import sys, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from evaluate import load_csv

pattern = sys.argv[1] if len(sys.argv) > 1 else "results/pareto_patch_2s*.csv"
files = sorted(glob.glob(pattern))
if not files:
    sys.exit(f"파일 없음: {pattern}")

# {설정명: {지표: [seed별 값]}}
data, seeds = {}, []
for f in files:
    # 파일명 끝의 _s<숫자>만 seed로 인정 (…_size_2s.csv 의 's'를 오인하지 않도록)
    m = re.search(r"_s(\d+)\.csv$", os.path.basename(f))
    seed = m.group(1) if m else "42"
    seeds.append(seed)
    for r in load_csv(f):
        d = data.setdefault(r["name"], {k: [] for k in
                                        ("hard", "soft", "win", "auc", "ep", "gap", "mflops")})
        d["hard"].append(r["subject_acc"] * 100)
        d["soft"].append(r["subject_acc_soft"] * 100)
        d["win"].append(r["window_acc"] * 100)
        d["auc"].append(r["roc_auc"])
        d["ep"].append(r["mean_epochs"])
        d["gap"].append(r["overfit_gap"])
        d["mflops"].append(r["mflops"])

names = list(data)
print(f"seed {len(files)}개: {', '.join(seeds)}\n")

print(f"{'설정':<7}{'MFLOPs':>9}{'윈도우':>9}"
      f"{'hard(평균±표준편차)':>22}{'soft(평균±표준편차)':>22}{'AUC':>8}{'ep':>6}")
for n in names:
    d = data[n]
    h, s = np.array(d["hard"]), np.array(d["soft"])
    print(f"{n:<7}{d['mflops'][0]:>9.1f}{np.mean(d['win']):>9.1f}"
          f"{np.mean(h):>13.1f} ± {np.std(h):<6.1f}"
          f"{np.mean(s):>13.1f} ± {np.std(s):<6.1f}"
          f"{np.mean(d['auc']):>8.3f}{np.mean(d['ep']):>6.1f}")

# ── 개별 seed 값 (튀는 정도 확인용) ──
print(f"\n[seed별 hard 정확도]  {'  '.join(f's{x}' for x in seeds)}")
for n in names:
    vals = "  ".join(f"{v:5.1f}" for v in data[n]["hard"])
    rng = max(data[n]["hard"]) - min(data[n]["hard"])
    print(f"  {n:<6} {vals}   (최대−최소 {rng:.1f}%p)")

# ── voting 결론 ──
all_h = np.concatenate([data[n]["hard"] for n in names])
all_s = np.concatenate([data[n]["soft"] for n in names])
diff = all_s - all_h
win_s, win_h, tie = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
print(f"\n[hard vs soft]  총 {len(diff)}개 비교 (설정 {len(names)} × seed {len(files)})")
print(f"  soft 우세 {win_s}  |  hard 우세 {win_h}  |  동률 {tie}")
print(f"  평균 차이 {diff.mean():+.2f}%p (표준편차 {diff.std():.2f})  "
      f"최대 {diff.max():+.1f} / 최소 {diff.min():+.1f}")

# 이항검정: soft가 hard보다 나을 확률이 50%라는 가정하에, 관측된 승수 이상이 나올 확률
from math import comb
n_eff = win_s + win_h                                   # 동률 제외
if n_eff:
    p = sum(comb(n_eff, k) for k in range(win_s, n_eff + 1)) / 2 ** n_eff
    print(f"  이항검정(동률 {tie}개 제외, n={n_eff}): p = {p:.4f}"
          f"  → {'유의미' if p < 0.05 else '판단 보류'}")

# ── 그림 ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8),
                               gridspec_kw={"width_ratios": [1.4, 1]})
x = np.arange(len(names))
hm = [np.mean(data[n]["hard"]) for n in names]
hs = [np.std(data[n]["hard"]) for n in names]
sm = [np.mean(data[n]["soft"]) for n in names]
ss = [np.std(data[n]["soft"]) for n in names]

ax1.bar(x - 0.2, hm, 0.4, yerr=hs, capsize=4, label="hard", color="#1e5aa8")
ax1.bar(x + 0.2, sm, 0.4, yerr=ss, capsize=4, label="soft", color="#c0392b")
for i, n in enumerate(names):                            # seed별 원본 점
    ax1.scatter([i - 0.2] * len(data[n]["hard"]), data[n]["hard"],
                s=18, color="#0d2f5c", zorder=5, alpha=0.8)
    ax1.scatter([i + 0.2] * len(data[n]["soft"]), data[n]["soft"],
                s=18, color="#7d1d13", zorder=5, alpha=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(names)
ax1.set_ylabel("피험자 정확도 (%)")
ax1.set_title(f"패치 크기별 정확도 (seed {len(files)}개 평균 ± 표준편차, 점=개별 seed)")
ax1.axhline(90, ls="--", color="#666", lw=1, alpha=0.7)
ax1.annotate("90%", (len(names) - 0.5, 90.3), fontsize=8, color="#666")
ax1.legend(); ax1.grid(alpha=0.3, axis="y")

ax2.hist(diff, bins=np.arange(diff.min() - 0.25, diff.max() + 0.5, 0.5),
         color="#1a7a3a", edgecolor="white")
ax2.axvline(0, color="#c0392b", lw=2)
ax2.axvline(diff.mean(), color="#0d2f5c", lw=2, ls="--",
            label=f"평균 {diff.mean():+.2f}%p")
ax2.set_xlabel("soft − hard (%p)"); ax2.set_ylabel("빈도")
ax2.set_title(f"soft voting 이득 분포 (n={len(diff)})")
ax2.legend(); ax2.grid(alpha=0.3, axis="y")

fig.tight_layout()
out = "results/seed_summary.png"
fig.savefig(out, dpi=140, facecolor="white")
print(f"\n그림 저장: {out}")
