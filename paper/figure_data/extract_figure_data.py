# -*- coding: utf-8 -*-
"""논문 그림 1~4를 손으로 다시 그리기 위한 데이터 추출.
결과는 이 폴더에 CSV(UTF-8 BOM, 엑셀에서 한글 깨짐 없음)로 저장된다.
실행: python paper/figure_data/extract_figure_data.py  (저장소 루트에서)
"""
import csv
import io
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results")
SEEDS = ["s42", "s1", "s7"]


def rows(name):
    with io.open(os.path.join(RES, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save(name, header, data):
    with io.open(os.path.join(HERE, name), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)
    print("저장:", name, len(data), "행")


def r4(x):
    return round(float(x), 4)


def mean_sd(v):
    v = np.asarray(v, float)
    return r4(v.mean()), r4(v.std(ddof=1)) if len(v) > 1 else 0.0


# ── 그림 1 · 교란 통제 (표 1과 같은 값) ──
conds = [("R0", "전체 61명", 61, 14.7, .813, .738, .574),
         ("R4", "2016년 제외 40명", 40, 11.3, .849, .775, .650),
         ("R5", "45세 이상 41명", 41, 3.3, .805, .659, .585)]
save("fig1a_정확도.csv",
     ["조건", "설명", "인원", "연령차(세)", "PatchTST 모델", "연령 규칙", "다수결 기준선"],
     [list(c) for c in conds])
save("fig1b_모델-규칙.csv", ["조건", "연령차(세)", "모델 - 연령 규칙"],
     [[c[0], c[3], round(c[4] - c[5], 3)] for c in conds])

# ── 그림 2(a) · 피험자별 판정 명확도 ──
npz = ["probs_d64L2_size_d64L2_2s.npz", "probs_d64L2_size_d64L2_2s_s1.npz",
       "probs_d64L2_size_d64L2_2s_s7.npz"]
mats, labels = [], None
for fn in npz:
    d = np.load(os.path.join(RES, fn), allow_pickle=True)
    prob, y, subj = d["prob"], d["y"], d["subject"]
    su = np.unique(subj)
    mats.append(np.array([prob[subj == s].mean() for s in su]))
    if labels is None:
        labels = np.array([int(y[subj == s][0]) for s in su])
m = np.vstack(mats)
pm = m.mean(axis=0)
clar = np.abs(pm - .5)
save("fig2a_피험자별_명확도.csv",
     ["피험자 번호", "군", "p(시드42)", "p(시드1)", "p(시드7)", "p 평균", "명확도 |p-0.5|", "애매(<0.1)"],
     [[i + 1, "환자" if labels[i] == 1 else "정상", r4(m[0, i]), r4(m[1, i]), r4(m[2, i]),
       r4(pm[i]), r4(clar[i]), "O" if clar[i] < .1 else ""] for i in np.argsort(clar)])
edges = np.linspace(0, .5, 15)
hp, _ = np.histogram(clar[labels == 1], edges)
hc, _ = np.histogram(clar[labels == 0], edges)
save("fig2a_히스토그램_구간.csv",
     ["구간 시작", "구간 끝", "환자 수", "정상 수", "합계(막대 전체 높이)"],
     [[r4(edges[i]), r4(edges[i + 1]), int(hp[i]), int(hc[i]), int(hp[i] + hc[i])]
      for i in range(14)])
per_seed = [int((np.abs(mm - .5) < .1).sum()) for mm in m]
print("  애매한 피험자(평균) %d명, 시드별 %s" % ((clar < .1).sum(), per_seed))

# ── 그림 2(b) · 연산량 대 성능 ──
models = [
    ("L1 (p16d128L1)", "PatchTST 깊이 축", 588.23,
     ["pareto_depthL1_2s.csv", "pareto_depthL1_2s_s1.csv", "pareto_depthL1_2s_s7.csv"]),
    ("L2 (p16d128L2)", "PatchTST 깊이 축", 1170.34,
     ["pareto_depth_p16d128L2_2s.csv", "pareto_depth_p16d128L2_2s_s1.csv", "pareto_depth_p16d128L2_2s_s7.csv"]),
    ("L3 (p16d128L3)", "PatchTST 깊이 축", 1752.45,
     ["pareto_depth_p16d128L3_2s.csv", "pareto_depth_p16d128L3_2s_s1.csv", "pareto_depth_p16d128L3_2s_s7.csv"]),
    ("L6 (p16d128L6, 참조)", "PatchTST 깊이 축", 3498.78,
     ["pareto_depthL6_2s.csv", "pareto_depth_p16d128L6_2s_s1.csv", "pareto_depth_p16d128L6_2s_s7.csv"]),
    ("d64L2", "패치 크게 · 폭 작게", 38.34,
     ["pareto_size_d64L2_2s.csv", "pareto_size_d64L2_2s_s1.csv", "pareto_size_d64L2_2s_s7.csv"]),
]
out = []
for name, kind, fl, files in models:
    a = [float(rows(f)[0]["roc_auc"]) for f in files]
    mu, sd = mean_sd(a)
    out.append([name, kind, fl, r4(a[0]), r4(a[1]), r4(a[2]), mu, sd])
out.append(["도메인 특징 4개 + 회귀", "표현 축", 0.001, "", "", "", .7974, ""])
save("fig2b_연산량_AUC.csv",
     ["모델", "종류", "MFLOPs", "AUC(시드42)", "AUC(시드1)", "AUC(시드7)", "AUC 평균", "SD(오차막대)"], out)

# ── 그림 3 · 특징 수 대 성능 (make_paper_figs.py E2 표와 동일) ──
E2 = [(1, .7274, .7157, .8277, "가속SD·그립력"), (2, .7487, .7735, .8891, "떨림·기울기X"),
      (3, .8086, .7899, .9025, "중주파·기울기X"), (4, .8085, .7974, .9053, "고주파·그립력"),
      (5, .8343, .8004, .9015, "중주파·축압력"), (6, .8313, .7984, .9004, "첨도·기울기Z"),
      (7, .8314, .7980, .8991, "떨림·기울기X"), (8, .8280, .7990, .8979, "첨도·기울기X")]
save("fig3_특징수_AUC.csv",
     ["k", "fixed 창 AUC", "greedy 창 AUC", "greedy 피험자 AUC", "k번째로 추가된 특징"],
     [list(r) for r in E2])

# ── 그림 4(a) · 채널별 효과크기 d_z ──
CH = {"ch0": "마이크", "ch1": "그립력", "ch2": "축압력", "ch3": "기울기X", "ch4": "기울기Y", "ch5": "기울기Z"}


def dz_table(suffix):
    t = {}
    for sd in SEEDS:
        for r in rows("bandpower_d64L2_%s_w1000ms_abs%s.csv" % (sd, suffix)):
            if r["band"] != "4-6Hz":
                continue
            t.setdefault((r["group"], r["channel"]), []).append(
                (float(r["cohen_dz"]), float(r["perm_z"]), float(r["p_holm"])))
    return t


t = dz_table("")
out = []
for ch, kr in CH.items():
    for g in ["환자", "정상"]:
        v = t[(g, ch)]
        mu, sd = mean_sd([x[0] for x in v])
        out.append([kr, ch, g] + [r4(x[0]) for x in v] + [mu, sd] +
                   [r4(x[2]) for x in v])
save("fig4a_채널별_dz.csv",
     ["채널", "채널코드", "군", "dz(시드42)", "dz(시드1)", "dz(시드7)", "dz 평균(막대)", "SD(오차막대)",
      "Holm p(시드42)", "Holm p(시드1)", "Holm p(시드7)"], out)

# ── 그림 4(b) · 과제별 (환자 · 축압력) ──
out = []
for suf, kr in [("_spir", "나선"), ("_mean", "구불선"), ("_circ", "원"), ("_diad", "손뒤집기")]:
    v = [x[0] for x in dz_table(suf)[("환자", "ch2")]]
    mu, sd = mean_sd(v)
    out.append([kr, suf[1:], r4(v[0]), r4(v[1]), r4(v[2]), mu, sd])
save("fig4b_과제별_dz.csv",
     ["과제", "코드", "dz(시드42)", "dz(시드1)", "dz(시드7)", "dz 평균(막대)", "SD(오차막대)"], out)
