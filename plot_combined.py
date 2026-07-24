"""고정깊이 설정 스윕 + Early Exit 곡선을 한 장에 겹쳐 그린다.

results/pareto.csv (설정별) 와 results/earlyexit.csv (임계값별)를 읽어
[연산량 vs 피험자 정확도] 평면에 함께 표시 → 두 접근을 직접 비교.

실행: python plot_combined.py
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import load_csv, pareto_mask

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

fixed = load_csv("results/pareto.csv")
ee    = load_csv("results/earlyexit.csv")

fx = np.array([r["mflops"] for r in fixed])
fa = np.array([r["subject_acc"] for r in fixed]) * 100
ex = np.array([r["mflops"] for r in ee])
ea = np.array([r["subject_acc"] for r in ee]) * 100

fig, ax = plt.subplots(figsize=(10, 6))

# Early Exit 곡선 (임계값 스윕 — 하나의 학습된 모델에서 나온 연속 곡선)
ax.plot(ex, ea, "-", color="#c0392b", lw=2.2, alpha=0.85, zorder=3,
        label="Early Exit 6층 (임계값 0.50→1.00)")
ax.scatter(ex, ea, s=28, color="#c0392b", zorder=4)

# 고정깊이 설정들
ax.scatter(fx, fa, s=150, color="#1e5aa8", zorder=5, marker="s",
           edgecolor="white", linewidth=1.5, label="고정깊이 설정")
for r, x, a in zip(fixed, fx, fa):
    ax.annotate(r["name"], (x, a), fontsize=10, fontweight="bold",
                color="#1e5aa8", textcoords="offset points", xytext=(8, 7))

# 두 접근을 합친 Pareto 프론티어
allx = np.concatenate([fx, ex])
alla = np.concatenate([fa, ea])
mask = pareto_mask(allx, alla)
order = np.argsort(allx[mask])
ax.plot(allx[mask][order], alla[mask][order], "--", color="#1a7a3a",
        lw=1.6, alpha=0.75, zorder=2, label="통합 Pareto 프론티어")

# 핵심 두 점 강조 (제목·범례와 겹치지 않게 배치)
i_ee = int(np.argmin(ex))                       # EE 최소 연산 (thr 0.5)
i_full = int(np.argmax(ex))                     # EE 조기종료 OFF (thr 1.0)
ax.annotate(f"thr 0.50\n{ea[i_ee]:.1f}% @ {ex[i_ee]:.0f}M",
            (ex[i_ee], ea[i_ee]), fontsize=9.5, color="#c0392b",
            fontweight="bold", ha="left", va="top",
            textcoords="offset points", xytext=(10, -6))
ax.annotate(f"thr 1.00 (조기종료 OFF)\n{ea[i_full]:.1f}% @ {ex[i_full]:.0f}M",
            (ex[i_full], ea[i_full]), fontsize=9.5, color="#c0392b",
            fontweight="bold", ha="right", va="bottom",
            textcoords="offset points", xytext=(-8, 10))

ax.set_xscale("log")
ax.set_xlabel("샘플당 연산량 (MFLOPs, 로그 스케일)")
ax.set_ylabel("피험자 정확도 (%)")
ax.set_title("고정깊이 설정 vs Early Exit — 정확도-연산량 트레이드오프 (windows_1s)")
ax.grid(alpha=0.3, which="both")
ax.set_ylim(min(alla.min(), 78) - 1.2, alla.max() + 1.2)
ax.legend(loc="lower right", framealpha=0.95)
fig.tight_layout()
fig.savefig("results/combined.png", dpi=150, facecolor="white")
print("저장: results/combined.png")

print(f"\nEarly Exit  thr0.50: {ea[i_ee]:.1f}% @ {ex[i_ee]:.1f} MFLOPs")
print(f"Early Exit  thr1.00: {ea[i_full]:.1f}% @ {ex[i_full]:.1f} MFLOPs")
print(f"→ 연산 {ex[i_full]/ex[i_ee]:.1f}배 절감, 정확도 {ea[i_ee]-ea[i_full]:+.1f}%p")
best_fixed = fixed[int(np.argmax(fa))]
print(f"\n고정깊이 최고: {best_fixed['name']} "
      f"{best_fixed['subject_acc']*100:.1f}% @ {best_fixed['mflops']:.1f} MFLOPs")
print(f"→ EE thr0.50이 같은 정확도를 "
      f"{best_fixed['mflops']/ex[i_ee]:.1f}배 적은 연산으로 달성")
