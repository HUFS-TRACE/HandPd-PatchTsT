"""공동 학습의 대가 — 백본이 일치하는 정적 모델과 EE 출구를 비교한다.

주의: EE 실험(earlyexit.py)은 patch 100/50 · d64 · head 4 · d_ff 128로 돌아갔다.
DEPTH_CONFIGS의 p16d128 계열은 patch 16/8 · d128이라 백본이 다르므로,
그쪽과 EE 출구를 비교하면 연산량이 28배 차이 나는 모델을 견주게 된다.
여기서는 SIZE_CONFIGS의 d64L* 계열(EE와 동일 설정)만 짝짓는다.

실행: python analyze_joint_cost.py
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

PAIRS = [(1, "d64L1"), (2, "d64L2"), (3, "d64L3"), (4, "d64L4"), (6, "d64L6")]
KEYS = ("window_acc", "roc_auc", "subject_acc", "mflops")


def collect_ee():
    out = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob("results/earlyexit_L6*_solo.csv")):
        if "deep" in f or "shallow" in f:          # uniform 가중만
            continue
        for r in load_csv(f):
            for k in KEYS:
                out[int(r["exit"])][k].append(r[k])
    return out


def collect_static():
    out = defaultdict(lambda: defaultdict(list))
    # 파일명 두 형태를 모두 잡는다.
    #   전체 suite 실행분: pareto_size_2s.csv
    #   --configs 지정분:  pareto_size_d64L6_2s.csv
    for f in sorted(glob.glob("results/pareto_size*.csv")):
        b = os.path.basename(f)
        if re.search(r"_lr|_bs\d|fold\d|pat\d|strat|autocw|_ls0|val0", b):
            continue                                # 프로토콜 실험분 제외
        for r in load_csv(f):
            for k in KEYS:
                out[r["name"]][k].append(r[k])
    return out


def welch(a, b):
    a, b = np.array(a, float), np.array(b, float)
    se = np.sqrt(a.std() ** 2 / len(a) + b.std() ** 2 / len(b))
    d = a.mean() - b.mean()
    return d, (d / se if se > 0 else 0.0)


ee, static = collect_ee(), collect_static()

print("=" * 72)
print("공동 학습의 대가 — 백본 일치 비교 (patch 100/50 · d64)")
print("=" * 72)

print(f"\n{'출구':>5}{'EE MFLOPs':>12}{'정적':>9}{'MFLOPs':>10}  백본 일치")
usable = []
for e, name in PAIRS:
    if name not in static or e not in ee:
        print(f"{e:>5}{'—':>12}{name:>9}{'없음':>10}")
        continue
    a, b = np.mean(ee[e]["mflops"]), np.mean(static[name]["mflops"])
    ok = abs(a - b) < 0.1
    print(f"{e:>5}{a:>12.2f}{name:>9}{b:>10.2f}  {'✅' if ok else '❌ 불일치'}")
    if ok:
        usable.append((e, name))

print(f"\n{'비교':<20}{'정적':>9}{'EE':>9}{'차이':>10}{'t':>7}   지표")
sig = tot = 0
for e, name in usable:
    for k, lab in (("window_acc", "윈도우"), ("roc_auc", "AUC")):
        s, v = static[name][k], ee[e][k]
        d, t = welch(s, v)
        tot += 1
        mark = ""
        if abs(t) >= 2:
            sig += 1
            mark = " ★"
        print(f"{name + ' vs 출구' + str(e):<20}{np.mean(s):>9.4f}{np.mean(v):>9.4f}"
              f"{d:>+10.4f}{t:>+7.2f}   {lab}{mark}")

print(f"\n  유의미(t≥2): {sig}/{tot}")
if usable:
    ds = [welch(static[n]["window_acc"], ee[e]["window_acc"])[0] * 100
          for e, n in usable]
    print(f"  윈도우 정확도 차이: {min(ds):+.1f} ~ {max(ds):+.1f}%p "
          f"(평균 {np.mean(ds):+.1f}%p)")
    print(f"  → 정적이 진 경우: {sum(1 for d in ds if d < 0)}건")

# ── 그림 ──
if usable:
    es = [e for e, _ in usable]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sm = [np.mean(static[n]["window_acc"]) * 100 for _, n in usable]
    ss = [np.std(static[n]["window_acc"]) * 100 for _, n in usable]
    em = [np.mean(ee[e]["window_acc"]) * 100 for e, _ in usable]
    es_ = [np.std(ee[e]["window_acc"]) * 100 for e, _ in usable]
    ax.errorbar(es, sm, yerr=ss, marker="o", capsize=4, lw=2.2,
                color="#1e5aa8", label="정적 (독립 학습)")
    ax.errorbar(es, em, yerr=es_, marker="s", capsize=4, lw=2.2,
                color="#c0392b", label="Early Exit 출구 (공동 학습)")
    ax.set_xlabel("층 수 / 출구 번호"); ax.set_ylabel("윈도우 정확도 (%)")
    ax.set_title("공동 학습의 대가 — 백본 일치 비교 (patch 100/50 · d64)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig("results/joint_cost.png", dpi=140, facecolor="white")
    print(f"\n그림 저장: results/joint_cost.png")
