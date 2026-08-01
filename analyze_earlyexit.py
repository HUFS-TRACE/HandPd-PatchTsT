"""Early Exit 심화 분석 — 세 가지 질문에 답한다.

1. 왜 깊을수록 나빠지는가?
   a. 깊은 출구가 단독으로도 나쁜가?  → *_solo.csv (조기종료 없이 출구별 성능)
   b. 조기종료 조합에서만 나쁜가?
2. max softmax가 옳은 종료 기준인가?  → *_pabee.csv 와 비교
3. 출구 손실 가중을 어떻게 줄 것인가? → uniform / deep / shallow 비교

실행: python analyze_earlyexit.py
"""
import sys, glob, os, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
from evaluate import load_csv

SCHEMES = ["uniform", "deep", "shallow"]
COLORS = {"uniform": "#1e5aa8", "deep": "#c0392b", "shallow": "#1a7a3a"}


def scheme_of(f):
    b = os.path.basename(f)
    for s in ("deep", "shallow"):
        if f"_{s}" in b:
            return s
    return "uniform"


def collect(suffix):
    """{가중방식: [파일별 rows]}

    suffix=""는 임계값 스윕 원본을 뜻한다. glob이 _solo/_pabee까지 잡으므로
    명시적으로 걸러낸다(컬럼이 달라 섞이면 KeyError).
    """
    out = defaultdict(list)
    for f in sorted(glob.glob(f"results/earlyexit_L6*{suffix}.csv")):
        if not suffix and re.search(r"_(solo|pabee)\.csv$", f):
            continue
        out[scheme_of(f)].append(load_csv(f))
    return out


def mean_std(vals):
    v = np.array(vals, dtype=float)
    return v.mean(), v.std()


# ══════════════════════════════════════════════════════════════
print("=" * 66)
print("Q1. 깊은 출구가 단독으로도 나쁜가? (조기종료와 분리해서 측정)")
print("=" * 66)

solo = collect("_solo")
solo_stats = {}
for sch in SCHEMES:
    runs = solo.get(sch, [])
    if not runs:
        continue
    n_exit = len(runs[0])
    print(f"\n### 손실 가중: {sch}  (seed {len(runs)}개)")
    print(f"{'출구':>5}{'MFLOPs':>9}{'피험자acc 평균±표준편차':>24}{'AUC':>8}")
    accs, aucs = [], []
    for e in range(n_exit):
        a = [r[e]["subject_acc"] * 100 for r in runs]
        u = [r[e]["roc_auc"] for r in runs]
        m, s = mean_std(a)
        accs.append((m, s)); aucs.append(np.mean(u))
        print(f"{e+1:>5}{runs[0][e]['mflops']:>9.1f}{m:>16.1f} ± {s:<5.1f}{np.mean(u):>8.3f}")
    solo_stats[sch] = accs

    best_i = int(np.argmax([m for m, _ in accs]))
    drop = accs[best_i][0] - accs[-1][0]
    print(f"  → 최고 출구 {best_i+1}({accs[best_i][0]:.1f}%) vs "
          f"마지막 출구 {n_exit}({accs[-1][0]:.1f}%) = {drop:+.1f}%p")
    # 최고와 마지막이 유의미하게 다른가 (표준오차 기준)
    n = len(runs)
    se = np.sqrt(accs[best_i][1] ** 2 / n + accs[-1][1] ** 2 / n)
    t = drop / se if se > 0 else 0
    print(f"    t = {t:.2f}  ⇒ "
          f"{'깊은 출구가 유의미하게 나쁨' if t >= 2 else '판단 보류' if t >= 1 else '차이 없음'}")

print("\n[Q1 답] 위 표는 조기종료를 전혀 쓰지 않고 각 출구를 단독 분류기로 썼을 때다.")
print("        여기서 이미 깊은 출구가 나쁘면 원인은 조합이 아니라 출구 자체(과적합)다.")

# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 66)
print("Q2. max softmax vs PABEE (연속 일치)")
print("=" * 66)

pabee = collect("_pabee")
thr = collect("")          # 임계값 스윕 (원본)

for sch in SCHEMES:
    pr, tr = pabee.get(sch, []), thr.get(sch, [])
    if not pr or not tr:
        continue
    print(f"\n### 손실 가중: {sch}")
    print(f"{'방식':>16}{'연산%':>8}{'피험자acc 평균±표준편차':>24}")

    # PABEE: k별
    for k in range(len(pr[0])):
        a = [r[k]["subject_acc"] * 100 for r in pr]
        c = [r[k]["compute_pct"] for r in pr]
        m, s = mean_std(a)
        print(f"{'PABEE k='+str(pr[0][k]['patience']):>16}"
              f"{np.mean(c):>8.1f}{m:>16.1f} ± {s:<5.1f}")

    # max softmax: 각 run에서 "연산량이 PABEE k=1과 비슷한 지점"과 "최고 정확도 지점"
    target = np.mean([r[0]["compute_pct"] for r in pr])
    same_cost, best_acc = [], []
    for r in tr:
        i = int(np.argmin([abs(x["compute_pct"] - target) for x in r]))
        same_cost.append((r[i]["subject_acc"] * 100, r[i]["compute_pct"]))
        j = int(np.argmax([x["subject_acc"] for x in r]))
        best_acc.append((r[j]["subject_acc"] * 100, r[j]["compute_pct"]))
    m, s = mean_std([a for a, _ in same_cost])
    print(f"{'softmax(동일연산)':>16}{np.mean([c for _, c in same_cost]):>8.1f}"
          f"{m:>16.1f} ± {s:<5.1f}")
    m2, s2 = mean_std([a for a, _ in best_acc])
    print(f"{'softmax(최고)':>16}{np.mean([c for _, c in best_acc]):>8.1f}"
          f"{m2:>16.1f} ± {s2:<5.1f}")

    diff = m2 - mean_std([r[0]["subject_acc"] * 100 for r in pr])[0]
    print(f"  → 동일 연산에서 softmax {m:.1f}% vs PABEE k=1 "
          f"{mean_std([r[0]['subject_acc']*100 for r in pr])[0]:.1f}%")

# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 66)
print("Q3. 출구 손실 가중 — uniform vs deep vs shallow")
print("=" * 66)
print(f"\n{'가중':>9}{'전층(thr1.0)':>16}{'최적 운영점':>18}{'최소연산':>16}{'단독최고':>14}")

summary = {}
for sch in SCHEMES:
    tr, so = thr.get(sch, []), solo.get(sch, [])
    if not tr:
        continue
    full = mean_std([r[-1]["subject_acc"] * 100 for r in tr])          # thr 1.0
    best = mean_std([max(x["subject_acc"] for x in r) * 100 for r in tr])
    bcost = np.mean([min(x["compute_pct"] for x in r) for r in tr])
    mincost = mean_std([r[0]["subject_acc"] * 100 for r in tr])        # thr 0.5
    solo_best = mean_std([max(x["subject_acc"] for x in r) * 100 for r in so]) if so else (0, 0)
    summary[sch] = dict(full=full, best=best, mincost=mincost, solo=solo_best)
    print(f"{sch:>9}{full[0]:>10.1f} ±{full[1]:<4.1f}"
          f"{best[0]:>12.1f} ±{best[1]:<4.1f}"
          f"{mincost[0]:>10.1f} ±{mincost[1]:<4.1f}"
          f"{solo_best[0]:>9.1f} ±{solo_best[1]:<4.1f}")

if len(summary) >= 2:
    key = "best"
    hi = max(summary.items(), key=lambda kv: kv[1][key][0])
    lo = min(summary.items(), key=lambda kv: kv[1][key][0])
    n = 3
    se = np.sqrt(hi[1][key][1] ** 2 / n + lo[1][key][1] ** 2 / n)
    d = hi[1][key][0] - lo[1][key][0]
    t = d / se if se > 0 else 0
    print(f"\n  최적 운영점 기준: {hi[0]}({hi[1][key][0]:.1f}) vs "
          f"{lo[0]}({lo[1][key][0]:.1f}) = {d:+.1f}%p, t = {t:.2f}")
    print(f"  ⇒ {'유의미' if t >= 2 else '판단 보류' if t >= 1 else '차이 없음'}")

# ══════════════════════════════════════════════════════════════ 그림
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# (1) 출구별 단독 성능
ax = axes[0]
for sch, accs in solo_stats.items():
    x = np.arange(1, len(accs) + 1)
    m = [a for a, _ in accs]; s = [b for _, b in accs]
    ax.errorbar(x, m, yerr=s, marker="o", capsize=3, lw=2,
                color=COLORS[sch], label=sch)
ax.set_xlabel("출구 번호 (깊이)"); ax.set_ylabel("피험자 정확도 (%)")
ax.set_title("Q1. 출구별 단독 성능\n(조기종료 없이 그 출구만 사용)")
ax.grid(alpha=0.3); ax.legend()

# (2) PABEE vs softmax
ax = axes[1]
for sch in SCHEMES:
    pr, tr = pabee.get(sch, []), thr.get(sch, [])
    if not pr:
        continue
    pc = [np.mean([r[k]["compute_pct"] for r in pr]) for k in range(len(pr[0]))]
    pa = [np.mean([r[k]["subject_acc"] * 100 for r in pr]) for k in range(len(pr[0]))]
    ax.plot(pc, pa, "s--", color=COLORS[sch], alpha=0.85, label=f"{sch} PABEE")
    L = min(len(r) for r in tr)
    tc = [np.mean([r[i]["compute_pct"] for r in tr]) for i in range(L)]
    ta = [np.mean([r[i]["subject_acc"] * 100 for r in tr]) for i in range(L)]
    ax.plot(tc, ta, "-", color=COLORS[sch], lw=2, label=f"{sch} softmax")
ax.set_xlabel("평균 연산량 (%)"); ax.set_ylabel("피험자 정확도 (%)")
ax.set_title("Q2. 종료 기준 비교\n(실선=max softmax, 점선=PABEE)")
ax.grid(alpha=0.3); ax.legend(fontsize=7.5)

# (3) 손실 가중 비교
ax = axes[2]
labels = ["전층\n(thr 1.0)", "최적\n운영점", "최소연산\n(thr 0.5)"]
w, x = 0.25, np.arange(3)
for i, sch in enumerate(SCHEMES):
    if sch not in summary:
        continue
    d = summary[sch]
    m = [d["full"][0], d["best"][0], d["mincost"][0]]
    s = [d["full"][1], d["best"][1], d["mincost"][1]]
    ax.bar(x + (i - 1) * w, m, w, yerr=s, capsize=3, color=COLORS[sch], label=sch)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("피험자 정확도 (%)")
ax.set_title("Q3. 출구 손실 가중 방식")
ax.grid(alpha=0.3, axis="y"); ax.legend()

fig.tight_layout()
fig.savefig("results/earlyexit_analysis.png", dpi=140, facecolor="white")
print(f"\n그림 저장: results/earlyexit_analysis.png")
