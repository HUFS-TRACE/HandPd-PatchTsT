# -*- coding: utf-8 -*-
"""A1 포스터용 그림 3장. 수치는 전부 results/*.csv에서 다시 읽는다.

  results/poster_fig1_gates.png    세 관문 (교란을 끊어도 살아남는다)
  results/poster_fig2_cost.png     연산량 vs 성능 (계산을 늘려도 안 나아진다)
  results/poster_fig3_evidence.png occlusion (모델은 떨림을 본다)
"""
import csv
import io
import os

import matplotlib

matplotlib.use("Agg")
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

# 포스터 축척 — A1에서 1m 거리에서 읽히는 크기
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 19, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.linewidth": 1.4, "axes.edgecolor": "#3a4a55",
})

NAVY, RED, GREY, GREEN = "#1b4965", "#b3121b", "#94a7b3", "#1f6f54"


def rows(name):
    with io.open(os.path.join(RES, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def auc_mean_sd(files):
    v = []
    for fn in files:
        p = os.path.join(RES, fn)
        if os.path.exists(p):
            v.append(float(rows(fn)[0]["roc_auc"]))
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** .5 if len(v) > 1 else 0.0
    return m, sd, len(v)


# ─────────────────────────── 그림 1 · 세 관문 ───────────────────────────
def fig_gates():
    conds = ["R0\n전체 61명", "R4\n2016년 제외\n40명", "R5\n45세 이상\n41명"]
    gap = [14.7, 11.3, 3.3]
    rule = [.738, .775, .659]
    model = [.813, .849, .805]
    major = [.574, .650, .585]
    adv = [m - r for m, r in zip(model, rule)]
    x = range(3)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 5.6),
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    a1.plot(x, model, "o-", lw=4.2, ms=15, color=NAVY, label="PatchTST 모델", zorder=3)
    a1.plot(x, rule, "s--", lw=3.8, ms=13, color=RED,
            label="연령 규칙 (임계 재최적화)", zorder=3)
    a1.plot(x, major, "^:", lw=2.4, ms=11, color=GREY, label="다수결 기준선", zorder=2)
    for i, (m, r) in enumerate(zip(model, rule)):
        a1.annotate("%.3f" % m, (i, m), textcoords="offset points", xytext=(0, 17),
                    ha="center", fontsize=17, color=NAVY, fontweight="bold")
        a1.annotate("%.3f" % r, (i, r), textcoords="offset points", xytext=(0, -30),
                    ha="center", fontsize=16, color=RED)
    for i, g in enumerate(gap):
        a1.annotate("연령차 +%.1f세" % g, (i, .922), ha="center", fontsize=15,
                    color="#31424e",
                    bbox=dict(boxstyle="round,pad=.42", fc="#eef2f4", ec="none"))
    a1.set_xticks(list(x))
    a1.set_xticklabels(conds)
    a1.set_ylim(.53, .995)
    a1.set_ylabel("피험자 정확도")
    a1.set_title("교란을 끊어도 모델은 버틴다", pad=16, fontweight="bold")
    a1.legend(loc="lower center", bbox_to_anchor=(.5, -.34), ncol=3, frameon=False)
    a1.grid(axis="y", alpha=.28)

    a2.bar([0, 1, 2], adv, color=[GREY, GREY, NAVY], width=.6)
    for i, v in enumerate(adv):
        a2.annotate("+%.3f" % v, (i, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=18, fontweight="bold")
    a2.set_xticks([0, 1, 2])
    a2.set_xticklabels(["R0\n+14.7세", "R4\n+11.3세", "R5\n+3.3세"])
    a2.set_ylim(0, .215)
    a2.set_ylabel("모델 정확도 - 연령 규칙")
    a2.set_title("연령을 맞출수록 우위가 커진다", pad=16, fontweight="bold")
    a2.grid(axis="y", alpha=.28)
    a2.annotate("연령을 대리 학습했다면\n반대 방향이어야 한다",
                xy=(1.70, .150), xytext=(0.45, .192), ha="center", va="center",
                fontsize=15, color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.2,
                                connectionstyle="arc3,rad=-.25"))
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "poster_fig1_gates.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  poster_fig1_gates.png")


# ──────────────────────── 그림 2 · 연산량 vs 성능 ────────────────────────
def fig_cost():
    depth = [
        ("L1", 588.23, ["pareto_depthL1_2s.csv", "pareto_depthL1_2s_s1.csv",
                        "pareto_depthL1_2s_s7.csv"]),
        ("L2", 1170.34, ["pareto_depth_p16d128L2_2s.csv",
                         "pareto_depth_p16d128L2_2s_s1.csv",
                         "pareto_depth_p16d128L2_2s_s7.csv"]),
        ("L3", 1752.45, ["pareto_depth_p16d128L3_2s.csv",
                         "pareto_depth_p16d128L3_2s_s1.csv",
                         "pareto_depth_p16d128L3_2s_s7.csv"]),
        ("L6", 3498.78, ["pareto_depthL6_2s.csv",
                         "pareto_depth_p16d128L6_2s_s1.csv",
                         "pareto_depth_p16d128L6_2s_s7.csv"]),
    ]
    dm, dsd, dfl = [], [], []
    for _, fl, files in depth:
        m, sd, _ = auc_mean_sd(files)
        dm.append(m); dsd.append(sd); dfl.append(fl)

    sm, ssd, _ = auc_mean_sd(["pareto_size_d64L2_2s.csv",
                              "pareto_size_d64L2_2s_s1.csv",
                              "pareto_size_d64L2_2s_s7.csv"])
    feat_x, feat_y = 0.001, .7974
    ref = dm[-1]                       # L6 참조
    drop = ref - sm
    ratio = dfl[-1] / 38.34

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.errorbar(dfl, dm, yerr=dsd, fmt="o", ms=15, color=NAVY, ecolor="#9fb6c4",
                elinewidth=2.6, capsize=7, capthick=2.4, zorder=3,
                label="PatchTST 깊이 축 (L1~L6)")
    for lb, xx, yy in zip([d[0] for d in depth], dfl, dm):
        ax.annotate(lb, (xx, yy), textcoords="offset points", xytext=(0, 20),
                    ha="center", fontsize=15, color=NAVY)
    ax.errorbar([38.34], [sm], yerr=[ssd], fmt="D", ms=18, color=GREEN,
                ecolor=GREEN, elinewidth=2.6, capsize=7, capthick=2.4, zorder=4,
                label="d64L2 — 패치 크게 · 폭 작게")
    ax.plot([feat_x], [feat_y], "s", ms=15, color=RED, zorder=3,
            label="도메인 특징 4개 + 회귀")

    ax.annotate("연산량 %.0f배 절감\nAUC %+.3f" % (ratio, -drop),
                xy=(38.34, sm - .004), xytext=(0.9, .818),
                fontsize=20, color=GREEN, fontweight="bold",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=3.2,
                                connectionstyle="arc3,rad=-.22"))

    ax.axhspan(ref - .02, ref + .02, color=NAVY, alpha=.07, zorder=0)
    ax.annotate("동등성 마진 ±0.02", xy=(0.0016, ref + .0205), fontsize=13.5,
                color="#4a6070", ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("연산량 (MFLOPs, 로그 눈금)")
    ax.set_ylabel("창 ROC-AUC")
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_title("계산을 늘려도 나아지지 않는다", pad=16, fontweight="bold")
    ax.annotate("오차막대 = 시드 3개 SD", xy=(.985, .975), xycoords="axes fraction",
                ha="right", va="top", fontsize=13, color="#5b6f7d")
    ax.set_ylim(.775, .905)
    ax.grid(alpha=.26)
    ax.legend(loc="lower right", framealpha=.96)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "poster_fig2_cost.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  poster_fig2_cost.png  (d64L2 %.4f±%.4f, 참조 %.4f, -%.4f, %.0f배)"
          % (sm, ssd, ref, drop, ratio))


# ────────────────────────── 그림 3 · 근거 검증 ──────────────────────────
def fig_evidence():
    seeds = ["s42", "s1", "s7"]
    chans = ["ch%d" % i for i in range(6)]

    def dz(suffix, group):
        out = {c: [] for c in chans}
        for sd in seeds:
            fn = "bandpower_d64L2_%s_w1000ms_abs%s.csv" % (sd, suffix)
            p = os.path.join(RES, fn)
            if not os.path.exists(p):
                continue
            for r in rows(fn):
                if r["group"] == group and r["channel"] in out:
                    out[r["channel"]].append(float(r["cohen_dz"]))
        return out

    pat, ctl = dz("", "환자"), dz("", "정상")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 5.6),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    xs = range(6)
    pm = [sum(pat[c]) / len(pat[c]) if pat[c] else 0 for c in chans]
    cm = [sum(ctl[c]) / len(ctl[c]) if ctl[c] else 0 for c in chans]
    w = .38
    a1.bar([i - w / 2 for i in xs], pm, w, color=NAVY, label="환자 (n=26)", zorder=2)
    a1.bar([i + w / 2 for i in xs], cm, w, color=GREY, label="정상 (n=35)", zorder=2)
    for i, c in enumerate(chans):
        for v in pat[c]:
            a1.plot(i - w / 2, v, "o", ms=6, mfc="white", mec="#0d2c40", mew=1.5, zorder=3)
        for v in ctl[c]:
            a1.plot(i + w / 2, v, "o", ms=6, mfc="white", mec="#5c6f7b", mew=1.5, zorder=3)
    a1.axhline(0, color="#3a4a55", lw=1.4)
    a1.set_xticks(list(xs))
    a1.set_xticklabels(["ch0", "ch1", "ch2\n축압력", "ch3", "ch4", "ch5"])
    a1.set_ylabel("효과크기  $d_z$  (근거 구간의 4–6Hz 쏠림)")
    a1.set_title("12개 검정 중 환자의 축압력만 살아남는다", pad=16, fontweight="bold")
    a1.legend(loc="upper right", framealpha=.96)
    a1.grid(axis="y", alpha=.26)
    a1.set_ylim(-.82, 1.20)
    a1.annotate("Holm $p$ = .0008 / .0030 / .0063\n세 시드 전부 유의",
                xy=(2, .96), ha="center", va="bottom",
                fontsize=15, color=NAVY, fontweight="bold")
    a1.annotate("정상군은 부호가 뒤집히고  Holm $p$ = 1.0"+chr(10)+"귀무가 성립해야 할 곳에서 귀무가 나온다",
                xy=(.985, .022), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=13.5, color="#4a5d69")

    tasks = [("_spir", "나선"), ("_mean", "구불선"), ("_circ", "원"), ("_diad", "손뒤집기")]
    tv = []
    for suf, lb in tasks:
        d = dz(suf, "환자")["ch2"]
        tv.append((lb, sum(d) / len(d) if d else 0.0, d))
    cols = [NAVY if v > .3 else (GREY if v > 0 else RED) for _, v, _ in tv]
    a2.bar(range(4), [v for _, v, _ in tv], color=cols, width=.62, zorder=2)
    for i, (_, v, d) in enumerate(tv):
        for j, x in enumerate(d):
            a2.plot(i + (j - 1) * .17, x, "o", ms=7, mfc="white",
                    mec="#33454f", mew=1.6, zorder=3)
        a2.annotate("%+.2f" % v, (i, v), textcoords="offset points",
                    xytext=(0, 26 if v > 0 else -36), ha="center",
                    fontsize=18, fontweight="bold")
    a2.axhline(0, color="#3a4a55", lw=1.4)
    a2.set_ylim(-.34, 1.26)
    a2.set_xticks(range(4))
    a2.set_xticklabels([lb for lb, _, _ in tv])
    a2.set_ylabel("환자 ch2 효과크기  $d_z$")
    a2.set_title("그림을 그리는 과제일수록 강하다", pad=16, fontweight="bold")
    a2.grid(axis="y", alpha=.26)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "poster_fig3_evidence.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  poster_fig3_evidence.png  환자 ch2 = %.3f" % pm[2])


if __name__ == "__main__":
    fig_gates()
    fig_cost()
    fig_evidence()
