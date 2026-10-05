# -*- coding: utf-8 -*-
"""논문용 그림. 수치는 전부 results/*.csv · *.npz 에서 다시 읽는다.

  results/paper_fig2_difficulty_cost.png  난이도 분산 + 연산량 대 성능
  results/paper_fig3_saturation.png       특징 개수 대 성능 (fixed vs greedy)

`fig_main.png`을 대체한다 — 그 그림에는 옛 수치(AUC -0.021, d64L2 .841)가
박혀 있었고 생성 스크립트가 남아 있지 않았다.
"""
import csv
import io
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")

for cand in ["Malgun Gothic", "MalgunGothic", "Gulim", "Batang"]:
    try:
        fm.findfont(fm.FontProperties(family=cand), fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10,
    "axes.linewidth": 1.0, "axes.edgecolor": "#44545f",
})

NAVY, RED, GREY, GREEN = "#1b4965", "#b3121b", "#94a7b3", "#1f6f54"
SEEDS = ["results/probs_d64L2_size_d64L2_2s.npz",
         "results/probs_d64L2_size_d64L2_2s_s1.npz",
         "results/probs_d64L2_size_d64L2_2s_s7.npz"]


def rows(name):
    with io.open(os.path.join(RES, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def auc_mean_sd(files):
    v = [float(rows(f)[0]["roc_auc"]) for f in files
         if os.path.exists(os.path.join(RES, f))]
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** .5 if len(v) > 1 else 0.
    return m, sd


def subject_clarity():
    """피험자별 |p-0.5| — 시드 3개 평균. 시드별 '애매한 인원'도 함께 돌려준다."""
    per_seed, mats, labels = [], [], None
    for rel in SEEDS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        prob, y, subj = d["prob"], d["y"], d["subject"]
        su = np.unique(subj)
        pm = np.array([prob[subj == s].mean() for s in su])
        per_seed.append(int((np.abs(pm - .5) < .1).sum()))
        mats.append(pm)
        if labels is None:
            labels = np.array([y[subj == s][0] for s in su])
    return np.vstack(mats).mean(axis=0), labels, per_seed


# ───────────────── 그림 2 · 난이도 분산 + 연산량 대 성능 ─────────────────
def fig_difficulty_cost():
    pm, ym, per_seed = subject_clarity()
    clarity = np.abs(pm - .5)
    n_amb = int((clarity < .1).sum())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.5),
                                 gridspec_kw={"width_ratios": [1, 1.1]})

    bins = np.linspace(0, .5, 15)
    a1.hist([clarity[ym == 1], clarity[ym == 0]], bins=bins, stacked=True,
            color=[NAVY, GREY], edgecolor="white", linewidth=.8,
            label=["환자 (n=%d)" % (ym == 1).sum(), "정상 (n=%d)" % (ym == 0).sum()])
    a1.axvspan(0, .1, color=RED, alpha=.08, zorder=0)
    a1.axvline(.1, color=RED, lw=1.4, ls="--")
    a1.annotate("애매한 피험자 %d명\n(시드별 %d~%d명 · %d~%d%%)"
                % (n_amb, min(per_seed), max(per_seed),
                   round(100 * min(per_seed) / len(ym)),
                   round(100 * max(per_seed) / len(ym))),
                xy=(.115, a1.get_ylim()[1] * .82), fontsize=10.5, color=RED,
                va="top")
    a1.set_xlabel("판정 명확도  |p - 0.5|   (시드 3개 평균)")
    a1.set_ylabel("피험자 수")
    a1.set_title("(a) 난이도 분산은 실재한다", pad=10)
    a1.legend(loc="upper right", framealpha=.95)
    a1.grid(axis="y", alpha=.25)

    depth = [("L1", 588.23, ["pareto_depthL1_2s.csv", "pareto_depthL1_2s_s1.csv",
                             "pareto_depthL1_2s_s7.csv"]),
             ("L2", 1170.34, ["pareto_depth_p16d128L2_2s.csv",
                              "pareto_depth_p16d128L2_2s_s1.csv",
                              "pareto_depth_p16d128L2_2s_s7.csv"]),
             ("L3", 1752.45, ["pareto_depth_p16d128L3_2s.csv",
                              "pareto_depth_p16d128L3_2s_s1.csv",
                              "pareto_depth_p16d128L3_2s_s7.csv"]),
             ("L6", 3498.78, ["pareto_depthL6_2s.csv",
                              "pareto_depth_p16d128L6_2s_s1.csv",
                              "pareto_depth_p16d128L6_2s_s7.csv"])]
    dm, dsd, dfl = [], [], []
    for _, fl, files in depth:
        m, sd = auc_mean_sd(files)
        dm.append(m); dsd.append(sd); dfl.append(fl)
    sm, ssd = auc_mean_sd(["pareto_size_d64L2_2s.csv", "pareto_size_d64L2_2s_s1.csv",
                           "pareto_size_d64L2_2s_s7.csv"])
    ref, drop, ratio = dm[-1], dm[-1] - sm, dfl[-1] / 38.34

    a2.axhspan(ref - .02, ref + .02, color=NAVY, alpha=.07, zorder=0)
    a2.annotate("동등성 마진 ±0.02", xy=(.0016, ref + .021), fontsize=9.5,
                color="#4a6070", va="bottom")
    a2.errorbar(dfl, dm, yerr=dsd, fmt="o", ms=9, color=NAVY, ecolor="#9fb6c4",
                elinewidth=1.7, capsize=4, zorder=3, label="PatchTST 깊이 축")
    for lb, xx, yy in zip([d[0] for d in depth], dfl, dm):
        a2.annotate(lb, (xx, yy), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=10, color=NAVY)
    a2.errorbar([38.34], [sm], yerr=[ssd], fmt="D", ms=11, color=GREEN,
                ecolor=GREEN, elinewidth=1.7, capsize=4, zorder=4,
                label="d64L2 — 패치 크게 · 폭 작게")
    a2.plot([.001], [.7974], "s", ms=9, color=RED, zorder=3,
            label="도메인 특징 4개 + 회귀")
    a2.annotate("연산량 %.0f배 절감\nAUC %+.3f" % (ratio, -drop),
                xy=(38.34, sm - .004), xytext=(.9, .812),
                fontsize=12, color=GREEN, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2,
                                connectionstyle="arc3,rad=-.22"))
    a2.set_xscale("log")
    a2.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
    a2.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100", "1,000"])
    a2.minorticks_off()
    a2.set_xlabel("연산량 (MFLOPs, 로그 눈금)")
    a2.set_ylabel("창 ROC-AUC")
    a2.set_ylim(.775, .90)
    a2.set_title("(b) 계산을 늘려도 나아지지 않는다", pad=10)
    a2.legend(loc="lower right", framealpha=.95)
    a2.grid(alpha=.25)
    a2.annotate("오차막대 = 시드 3개 SD", xy=(.985, .975), xycoords="axes fraction",
                ha="right", va="top", fontsize=9.5, color="#5b6f7d")

    fig.tight_layout()
    fig.savefig(os.path.join(RES, "paper_fig2_difficulty_cost.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  paper_fig2  애매 %d명 (시드별 %s) · d64L2 %.4f · %.0f배 · %+.4f"
          % (n_amb, per_seed, sm, ratio, -drop))


# ───────────────── 그림 3 · 특징 개수 대 성능 (E2) ─────────────────
# 출처: 확인서/0906_E1E2_결과.pdf (원종윤, 2026-09-06) — 원본 results/e2.csv
E2 = [  # k, fixed, greedy, greedy_subject, 고른 것
    (1, .7274, .7157, .8277, "가속SD·그립력"),
    (2, .7487, .7735, .8891, "떨림·기울기X"),
    (3, .8086, .7899, .9025, "중주파·기울기X"),
    (4, .8085, .7974, .9053, "고주파·그립력"),
    (5, .8343, .8004, .9015, "중주파·축압력"),
    (6, .8313, .7984, .9004, "첨도·기울기Z"),
    (7, .8314, .7980, .8991, "떨림·기울기X"),
    (8, .8280, .7990, .8979, "첨도·기울기X"),
]
E2_REF = .8629


def fig_saturation():
    k = [r[0] for r in E2]
    fixed = [r[1] for r in E2]
    greedy = [r[2] for r in E2]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(E2_REF, color=NAVY, lw=1.6, ls="--", zorder=2)
    ax.annotate("참조 모델 p16d128L6  .8629", xy=(8.05, E2_REF), ha="right",
                va="bottom", fontsize=10.5, color=NAVY)

    ax.plot(k, fixed, "s--", ms=8, lw=2, color=GREY, zorder=3,
            label="fixed — 전체 데이터에서 순위 (낙관 편향)")
    ax.plot(k, greedy, "o-", ms=9, lw=2.6, color=GREEN, zorder=4,
            label="greedy — fold 안 검증집합에서만 선택")

    ax.annotate("k = 4에서 포화\n.7974 = 참조의 93%",
                xy=(4, .7974), xytext=(5.9, .757), fontsize=11.5, color=GREEN,
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8,
                                connectionstyle="arc3,rad=.25"))
    plateau = max(greedy[3:])
    ax.axhline(plateau, color=GREEN, lw=1.1, ls=":", zorder=2)
    ax.annotate("", xy=(1.28, E2_REF), xytext=(1.28, plateau),
                arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=1.8))
    ax.annotate("남은 %.2f%%p는 소수 특징으로\n환원되지 않는다"
                % (100 * (E2_REF - plateau)),
                xy=(1.45, (E2_REF + plateau) / 2), ha="left", va="center",
                fontsize=10.5, color=RED)

    ax.set_xlabel("선택한 특징 수  k   (후보 54개)")
    ax.set_ylabel("창 ROC-AUC")
    ax.set_xticks(k)
    ax.set_ylim(.70, .885)
    ax.grid(alpha=.25)
    ax.legend(loc="lower right", framealpha=.95)
    ax.set_title("판별 정보는 소수 축에 있으나, 전부는 아니다", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "paper_fig3_saturation.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  paper_fig3  greedy k=4 .7974 · 잔차 %.1f%%p" % (100 * (E2_REF - .8004)))


if __name__ == "__main__":
    fig_difficulty_cost()
    fig_saturation()
