"""results/의 모든 스윕 결과를 파라미터 축별로 집계한다.

파일명에서 (스윕, 데이터, lr, batch, folds, patience, seed)를 파싱해
같은 조건끼리 묶고, seed 방향으로 평균±표준편차를 낸다.

핵심 질문: "이 파라미터를 바꾼 효과가 seed 노이즈보다 큰가?"
  설정 간 차이 < seed 변동 이면 그 파라미터는 튜닝 가치가 없다.

실행: python analyze_all.py
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

DEFAULTS = dict(lr="1e-3", bs="64", fold="5", pat="10", seed="42")


def parse(fname):
    """pareto_size_2s_lr0.0003_s1.csv → 조건 dict"""
    b = os.path.basename(fname)[len("pareto_"):-len(".csv")]
    c = dict(DEFAULTS)
    c["suite"] = "patch" if b.startswith("patch") else "size" if b.startswith("size") \
        else "head" if b.startswith("head") else "default"
    for key, pat in (("lr", r"lr([\d.e-]+)"), ("bs", r"bs(\d+)"),
                     ("fold", r"fold(\d+)"), ("pat", r"pat(\d+)"), ("seed", r"s(\d+)$")):
        m = re.search(pat, b)
        if m:
            c[key] = m.group(1)
    c["data"] = "2s" if "_2s" in b else "1s"
    return c


# ── 수집: {(축이름, 축값, 설정명): [정확도들]} ──
rows = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob("results/pareto_*.csv")):
    c = parse(f)
    if c["data"] != "2s":                    # 2초 데이터만 (1초는 조건이 달라 섞으면 안 됨)
        continue
    for r in load_csv(f):
        key = (c["lr"], c["bs"], c["fold"], c["pat"], r["name"])
        rows[key]["hard"].append(r["subject_acc"] * 100)
        rows[key]["soft"].append(r["subject_acc_soft"] * 100)
        rows[key]["auc"].append(r["roc_auc"])
        rows[key]["ep"].append(r["mean_epochs"])
        rows[key]["gap"].append(r["overfit_gap"])
        rows[key]["mflops"].append(r["mflops"])


def show(title, axis_name, filt, label_of):
    """한 축을 표로 출력하고 (라벨, 평균, 표준편차) 리스트를 반환."""
    groups = defaultdict(list)
    for key, d in rows.items():
        if not filt(key):
            continue
        groups[label_of(key)].extend(d["hard"])
    if len(groups) < 2:
        return []

    print(f"\n### {title}")
    print(f"{axis_name:>10}{'seed수':>7}{'hard 평균±표준편차':>20}{'최소':>7}{'최대':>7}")
    out = []
    for lab in sorted(groups, key=lambda x: (len(str(x)), str(x))):
        v = np.array(groups[lab])
        print(f"{str(lab):>10}{len(v):>7}{v.mean():>13.1f} ± {v.std():<5.1f}"
              f"{v.min():>7.1f}{v.max():>7.1f}")
        out.append((str(lab), v.mean(), v.std(), len(v)))

    # 최고 그룹 vs 최저 그룹을 Welch t 근사로 비교.
    #   표준편차가 아니라 "평균의 표준오차"(std/√n)를 써야 한다.
    #   seed를 늘릴수록 평균은 안정되므로, 표준편차만 보면 효과를 과대평가한다.
    hi = max(out, key=lambda x: x[1])
    lo = min(out, key=lambda x: x[1])
    se = np.sqrt((hi[2] ** 2) / max(hi[3], 1) + (lo[2] ** 2) / max(lo[3], 1))
    spread = hi[1] - lo[1]
    t = spread / se if se > 0 else float("inf")
    verdict = ("효과 있음" if t >= 2 else
               "판단 보류(seed 더 필요)" if t >= 1 else "노이즈에 묻힘")
    print(f"  → 최고 {hi[0]}({hi[1]:.1f}) vs 최저 {lo[0]}({lo[1]:.1f}) = {spread:.1f}%p")
    print(f"    차이의 표준오차 {se:.1f}%p → t = {t:.2f}  ⇒ {verdict}")
    return [(lab, m, s) for lab, m, s, _ in out]


print("=" * 62)
print("파라미터 축별 집계 (windows_2s.npz, 피험자 61명)")
print("=" * 62)

base = lambda k: k[0] == "1e-3" and k[1] == "64" and k[2] == "5" and k[3] == "10"

axes = {}
# 설정 축(패치/크기/head)은 기본 하이퍼파라미터 조건에서만
axes["패치 크기"] = show("패치 크기 (patch_len)", "설정",
                      lambda k: base(k) and k[4].startswith("p"), lambda k: k[4])
axes["모델 크기"] = show("모델 크기 (d_model·n_layers)", "설정",
                      lambda k: base(k) and k[4].startswith("d"), lambda k: k[4])
axes["head 수"] = show("어텐션 head 수", "설정",
                     lambda k: base(k) and k[4].startswith("h"), lambda k: k[4])
# 학습 하이퍼파라미터는 d64L2 고정
d64 = lambda k: k[4] == "d64L2"
axes["학습률"] = show("학습률 (lr)", "lr",
                   lambda k: d64(k) and k[1] == "64" and k[2] == "5" and k[3] == "10",
                   lambda k: k[0])
axes["배치 크기"] = show("배치 크기 (batch_size)", "batch",
                     lambda k: d64(k) and k[0] == "1e-3" and k[2] == "5" and k[3] == "10",
                     lambda k: k[1])
axes["fold 수"] = show("fold 수 (n_folds)", "folds",
                     lambda k: d64(k) and k[0] == "1e-3" and k[1] == "64" and k[3] == "10",
                     lambda k: k[2])
axes["patience"] = show("patience (epochs 상한 실증)", "patience",
                        lambda k: d64(k) and k[0] == "1e-3" and k[1] == "64" and k[2] == "5",
                        lambda k: k[3])

# ── seed 노이즈 자체의 크기 ──
per_cfg = [np.array(d["hard"]) for d in rows.values() if len(d["hard"]) >= 3]
if per_cfg:
    print(f"\n### seed 노이즈 (같은 조건, seed만 다름)")
    ranges = [v.max() - v.min() for v in per_cfg]
    print(f"  조건 {len(per_cfg)}개  평균 변동폭 {np.mean(ranges):.1f}%p  "
          f"최대 {np.max(ranges):.1f}%p")

# ── 축별 효과 크기 비교 그림 ──
valid = {k: v for k, v in axes.items() if v}
if valid:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = list(valid)
    spreads = [max(m for _, m, _ in v) - min(m for _, m, _ in v) for v in valid.values()]
    noise = float(np.mean(ranges)) if per_cfg else 0
    colors = ["#1a7a3a" if s > noise else "#999" for s in spreads]
    ax.barh(names, spreads, color=colors)
    ax.axvline(noise, color="#c0392b", lw=2, ls="--",
               label=f"seed 노이즈 {noise:.1f}%p")
    for i, s in enumerate(spreads):
        ax.text(s + 0.1, i, f"{s:.1f}", va="center", fontsize=9)
    ax.set_xlabel("이 축을 바꿔서 얻는 정확도 차이 (%p)")
    ax.set_title("파라미터 축별 효과 크기 vs seed 노이즈")
    ax.legend(); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig("results/axis_effects.png", dpi=140, facecolor="white")
    print(f"\n그림 저장: results/axis_effects.png")
    print(f"  빨간 선(seed 노이즈)보다 짧은 축 = 튜닝해도 의미 없음")
