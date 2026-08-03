"""E1 — 통제된 정적 깊이 대조군 집계.

Early Exit과 동일한 백본(patch 16/8, d128, head 8, d_ff 256)에서 층 수만 바꿔
독립 학습한 정적 모델을, EE 출구별 단독 성능과 나란히 놓는다.

두 가지를 가른다.
  (1) 정적 L1 vs 정적 L6  — 깊이 자체가 무용한가? (논문 §5.5 사전 등록 규칙)
  (2) 정적 Lk vs EE 출구 k — 공동 학습이 손해를 주는가?

실행: python analyze_depth.py
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

EXPECT_SEEDS = 3


def collect_static():
    """{층수: {지표: [seed별 값]}} — 정적 대조군.

    파일명이 두 종류다. seed 42 초기 실행분은 --configs 태그가 없던 시절이라
    수동 백업한 pareto_depthL{n}_2s*.csv 이고, 이후는 설정 이름이 들어간
    pareto_depth_p16d128L{n}_2s*.csv 다. 둘 다 읽되 (층, seed)로 중복을 제거한다.
    """
    out = defaultdict(lambda: defaultdict(dict))     # [층][지표][seed]
    for f in glob.glob("results/pareto_depth*.csv"):
        base = os.path.basename(f)
        m = re.search(r"L(\d)", base)
        if not m:
            continue
        L = int(m.group(1))
        sm = re.search(r"_s(\d+)\.csv$", base)
        seed = sm.group(1) if sm else "42"
        for r in load_csv(f):
            if not r["name"].startswith("p16d128L"):
                continue
            for k in ("window_acc", "roc_auc", "subject_acc",
                      "mean_epochs", "overfit_gap", "mflops"):
                out[L][k][seed] = r[k]
    return out


def collect_ee():
    """{출구: {지표: [seed별 값]}} — EE 출구별 단독 성능 (uniform 가중만)."""
    out = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob("results/earlyexit_L6*_solo.csv")):
        if "deep" in f or "shallow" in f:           # uniform만
            continue
        for r in load_csv(f):
            e = int(r["exit"])
            for k in ("window_acc", "roc_auc", "subject_acc", "mflops"):
                out[e][k].append(r[k])
    return out


def ms(vals):
    v = np.array(list(vals), dtype=float)
    return v.mean(), v.std(), len(v)


def welch_t(a, b):
    """(평균차, t) — 평균의 표준오차 기준. 표준편차가 아니라 std/√n을 쓴다."""
    (ma, sa, na), (mb, sb, nb) = ms(a), ms(b)
    se = np.sqrt(sa ** 2 / max(na, 1) + sb ** 2 / max(nb, 1))
    return ma - mb, (ma - mb) / se if se > 0 else 0.0


static, ee = collect_static(), collect_ee()
if not static:
    sys.exit("정적 대조군 결과가 없습니다 (results/pareto_depth*.csv)")

print("=" * 74)
print("E1 — 통제된 정적 깊이 대조군 (patch 16/8, d128, windows_2s)")
print("=" * 74)

# 표본 수 검증 — 병렬 실행 크래시로 표본이 1개인 걸 모르고 집계한 적이 있다
short = {L: len(d["window_acc"]) for L, d in static.items()
         if len(d["window_acc"]) < EXPECT_SEEDS}
if short:
    print("\n⚠️  seed 표본 부족 (기대 %d개):" % EXPECT_SEEDS)
    for L, n in sorted(short.items()):
        print(f"     L{L}: {n}개 — 표준편차를 신뢰할 수 없음")

print(f"\n{'층':>3}{'MFLOPs':>9}{'윈도우 정확도':>20}{'ROC-AUC':>18}"
      f"{'피험자acc':>16}{'gap':>7}{'seed':>6}")
for L in sorted(static):
    d = static[L]
    wm, ws, n = ms(d["window_acc"].values())
    am, as_, _ = ms(d["roc_auc"].values())
    sm_, ss, _ = ms(d["subject_acc"].values())
    gm, _, _ = ms(d["overfit_gap"].values())
    mf = list(d["mflops"].values())[0]
    print(f"{L:>3}{mf:>9.1f}{wm:>13.4f} ± {ws:<5.3f}{am:>11.4f} ± {as_:<5.3f}"
          f"{sm_:>10.4f} ± {ss:<4.3f}{gm:>7.2f}{n:>6}")

# ── (1) 깊이 자체 — 논문 §5.5 판정 ──
if 1 in static and 6 in static:
    print(f"\n### 판정 1 — 깊이 자체가 무용한가 (논문 §5.5 사전 등록 규칙)")
    for key, label in (("window_acc", "윈도우 정확도"), ("roc_auc", "ROC-AUC")):
        d, t = welch_t(static[6][key].values(), static[1][key].values())
        print(f"  {label:<12} 정적 L6 − 정적 L1 = {d:+.4f}   t = {t:+.2f}")
    _, t = welch_t(static[6]["window_acc"].values(), static[1]["window_acc"].values())
    if abs(t) < 2:
        print("  ⇒ 구별되지 않음 → **깊이 자체가 원인**. §5.1 결론 강화")
    elif t > 0:
        print("  ⇒ L6이 유의하게 높음 → 공동 학습이 원인. §5.1 결론 약화")
    else:
        print("  ⇒ L6이 유의하게 낮음 → 깊이가 해로움. 주장 강화")

# ── (2) 공동 학습의 대가 ──
if ee:
    print(f"\n### 판정 2 — 공동 학습이 손해를 주는가 (정적 Lk vs EE 출구 k)")
    print(f"{'층/출구':>8}{'정적':>10}{'EE':>10}{'차이':>9}{'t':>7}   지표")
    for L in sorted(set(static) & set(ee)):
        for key, label in (("window_acc", "윈도우"), ("roc_auc", "AUC")):
            sv = list(static[L][key].values())
            ev = ee[L][key]
            d, t = welch_t(sv, ev)
            mark = " ★" if abs(t) >= 2 else ""
            print(f"{L:>8}{np.mean(sv):>10.4f}{np.mean(ev):>10.4f}"
                  f"{d:>+9.4f}{t:>+7.2f}   {label}{mark}")

# ── 그림 ──
Ls = sorted(static)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

sw = [ms(static[L]["window_acc"].values()) for L in Ls]
ax1.errorbar(Ls, [m * 100 for m, _, _ in sw], yerr=[s * 100 for _, s, _ in sw],
             marker="o", capsize=4, lw=2.2, color="#1e5aa8", label="정적 (독립 학습)")
if ee:
    Es = sorted(ee)
    ax1.errorbar(Es, [np.mean(ee[e]["window_acc"]) * 100 for e in Es],
                 yerr=[np.std(ee[e]["window_acc"]) * 100 for e in Es],
                 marker="s", capsize=4, lw=2.2, color="#c0392b",
                 label="Early Exit 출구 (공동 학습)")
ax1.set_xlabel("층 수 / 출구 번호"); ax1.set_ylabel("윈도우 정확도 (%)")
ax1.set_title("E1 — 정적 깊이 vs Early Exit 출구")
ax1.grid(alpha=0.3); ax1.legend()

gaps = [ms(static[L]["overfit_gap"].values())[0] for L in Ls]
ax2.plot(Ls, gaps, "o-", lw=2.2, color="#b26a00")
ax2.set_xlabel("층 수"); ax2.set_ylabel("train/val loss 간극")
ax2.set_title("과적합 gap — 깊을수록 커진다")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("results/depth_e1.png", dpi=140, facecolor="white")
print(f"\n그림 저장: results/depth_e1.png")
