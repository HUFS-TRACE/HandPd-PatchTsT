"""평가 프로토콜·정규화 실험 집계.

기준선(GroupKFold + 상수 클래스가중 + 스무딩 없음) 대비 각 변경의 효과를
seed 방향 평균±표준편차와 t값으로 판정한다.

실행: python analyze_protocol.py
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

# 파일명 태그 → 사람이 읽는 이름
def label_of(fname):
    b = os.path.basename(fname)[len("pareto_size_2s"):-len(".csv")]
    b = re.sub(r"_s\d+$", "", b).strip("_")        # seed 제거
    return b if b else "기준선"


rows = defaultdict(list)
for f in sorted(glob.glob("results/pareto_size_2s*.csv")):
    lab = label_of(f)
    # 이번 실험과 무관한 축(lr/bs/fold/pat)은 제외
    if re.search(r"lr|bs\d|fold\d|pat\d", lab):
        continue
    for r in load_csv(f):
        if r["name"] != "d64L2":                   # 이번 스윕은 d64L2 고정
            continue
        rows[lab].append(r)

if "기준선" not in rows:
    sys.exit("기준선(results/pareto_size_2s.csv)이 없습니다.")

# ── 표본 수 검증 ──
# 이전에 병렬 실행이 절반 이상 크래시했는데 CSV 개수를 안 세서 모르고
# "표준편차 0.0"을 안정성으로 오독한 적이 있다. 집계 전에 반드시 확인한다.
EXPECT = 3
short = {lab: len(v) for lab, v in rows.items() if len(v) < EXPECT}
if short:
    print("⚠️  seed 표본이 부족한 설정이 있습니다 (기대 %d개):" % EXPECT)
    for lab, n in sorted(short.items()):
        print(f"     {lab:<20} {n}개  ← 표준편차를 신뢰할 수 없음")
    print()

def stat(lab, key="subject_acc"):
    v = np.array([r[key] * 100 for r in rows[lab]])
    return v.mean(), v.std(), len(v)


base_m, base_s, base_n = stat("기준선")
print("=" * 70)
print("평가 프로토콜 · 정규화 실험 (d64L2, windows_2s, seed 3개)")
print("=" * 70)
print(f"\n기준선: GroupKFold + 클래스가중 2.0/1.0 + 스무딩 없음 + val 0.1")
print(f"        피험자 정확도 {base_m:.1f} ± {base_s:.1f}  (seed {base_n}개)\n")

print(f"{'설정':<16}{'피험자acc':>18}{'기준선 대비':>12}{'t':>7}  판정")
results = []
for lab in sorted(rows, key=lambda x: (x == "기준선", x)):
    if lab == "기준선":
        continue
    m, s, n = stat(lab)
    d = m - base_m
    se = np.sqrt(s ** 2 / max(n, 1) + base_s ** 2 / max(base_n, 1))
    t = d / se if se > 0 else 0
    verdict = ("개선" if t >= 2 else "악화" if t <= -2 else
               "판단 보류" if abs(t) >= 1 else "차이 없음")
    print(f"{lab:<16}{m:>11.1f} ± {s:<4.1f}{d:>+11.1f}{t:>7.2f}  {verdict}")
    results.append((lab, m, s, d, t))

# ── 안정성(표준편차) 관점 ──
print(f"\n{'설정':<16}{'표준편차':>10}{'기준선 대비':>12}")
print(f"{'기준선':<16}{base_s:>10.1f}")
for lab, m, s, d, t in sorted(results, key=lambda x: x[2]):
    print(f"{lab:<16}{s:>10.1f}{s-base_s:>+12.1f}")
print("  → 표준편차가 작을수록 seed를 덜 타서 재현성이 좋다")

# ── val_size 축만 따로 (n_folds 미스터리 검증) ──
val_labs = [l for l in rows if l.startswith("val") or l == "기준선"]
if len(val_labs) >= 2:
    print(f"\n### val 비율 (n_folds가 유의미했던 원인 검증)")
    print(f"{'val 비율':>10}{'val 인원(추정)':>15}{'피험자acc':>18}")
    order = {"val0.05": 0.05, "기준선": 0.10, "val0.2": 0.20}
    for lab in sorted(val_labs, key=lambda x: order.get(x, 0)):
        m, s, _ = stat(lab)
        vs = order.get(lab, 0.1)
        print(f"{vs:>10.2f}{49 * vs:>15.1f}명{m:>13.1f} ± {s:<4.1f}")
    print("  가설: 10-fold에서 val이 5~6명까지 줄어 조기중단이 불안정해진다")
    print("        → val 비율을 낮추면(0.05) 같은 현상이 재현되어야 한다")

# ── 그림 ──
labs = ["기준선"] + [r[0] for r in results]
means = [base_m] + [r[1] for r in results]
stds = [base_s] + [r[2] for r in results]
colors = ["#666"] + ["#1a7a3a" if r[4] >= 1 else "#c0392b" if r[4] <= -1 else "#1e5aa8"
                     for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                               gridspec_kw={"width_ratios": [1.5, 1]})
x = np.arange(len(labs))
ax1.bar(x, means, yerr=stds, capsize=4, color=colors)
ax1.axhline(base_m, ls="--", color="#666", lw=1.5, alpha=0.8, label="기준선")
ax1.set_xticks(x); ax1.set_xticklabels(labs, rotation=30, ha="right", fontsize=9)
ax1.set_ylabel("피험자 정확도 (%)")
ax1.set_ylim(min(np.array(means) - np.array(stds)) - 2,
             max(np.array(means) + np.array(stds)) + 2)
ax1.set_title("평가 프로토콜·정규화 변경 효과 (seed 3개 평균±표준편차)")
ax1.legend(); ax1.grid(alpha=0.3, axis="y")

ax2.barh(x, stds, color=colors)
ax2.axvline(base_s, ls="--", color="#666", lw=1.5, label=f"기준선 {base_s:.1f}")
ax2.set_yticks(x); ax2.set_yticklabels(labs, fontsize=9)
ax2.set_xlabel("표준편차 (%p) — 작을수록 재현성 좋음")
ax2.set_title("seed 안정성")
ax2.legend(); ax2.grid(alpha=0.3, axis="x")

fig.tight_layout()
fig.savefig("results/protocol_effects.png", dpi=140, facecolor="white")
print(f"\n그림 저장: results/protocol_effects.png")
