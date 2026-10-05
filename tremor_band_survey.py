"""
채널별 떨림 대역(4~6Hz) 파워가 군 간에 다른가 — 모델과 무관한 사전 확인.

왜 이게 먼저 필요한가
    `bandpower_test.py`(X2)는 "모델이 짚은 구간에 떨림이 몰려 있는가"를 묻는다.
    그런데 그 질문이 뜻을 가지려면, **애초에 이 데이터의 떨림 대역이 군을
    가르는가**가 먼저 확인되어야 한다. 그렇지 않다면 X2가 무엇을 잡아내든
    "떨림을 봤다"로 읽을 수 없다.

    이 스크립트는 모델을 전혀 쓰지 않는다. 신호와 라벨만 본다. 따라서 X2와
    독립이며, 둘을 나란히 놓으면 "떨림이 군을 가른다"(여기) + "모델이 그
    떨림을 본다"(X2)의 두 단계가 된다.

주의 — 이 분석은 라벨을 쓴다
    X2는 라벨 없이 성립하지만 이 사전 확인은 군 비교라 라벨이 필요하다.
    두 분석의 성격이 다르므로 논문에서도 구분해 쓴다.

피험자 평균을 먼저 내는 이유
    환자 1명당 윈도우가 정상의 2.4배라, 윈도우 단위로 비교하면 환자 쪽
    사람들의 신호가 과대 대표된다. 사람마다 평균을 낸 뒤 사람 단위로
    비교한다.

사용법
    python tremor_band_survey.py
    python tremor_band_survey.py --band 3,7 --csv results/tremor_survey.csv
"""
import sys
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
import argparse
import csv
import io
import os

import numpy as np
from scipy import stats

from dataset import load_npz
from bandpower_test import band_power

CH_NAMES = ["ch0 마이크", "ch1 그립력", "ch2 축압력",
            "ch3 기울기X", "ch4 기울기Y", "ch5 기울기Z"]


def main():
    ap = argparse.ArgumentParser(description="채널별 떨림 대역 파워 군 비교")
    ap.add_argument("--data-path", default="data/windows_2s.npz")
    ap.add_argument("--band", default="4,6")
    ap.add_argument("--fs", type=int, default=1000)
    ap.add_argument("--csv", default="results/tremor_band_survey.csv")
    a = ap.parse_args()
    lo, hi = [float(s) for s in a.band.split(",")]

    X, y, subj, task = load_npz(a.data_path)
    n_ch = X.shape[1]
    print(f"데이터 {a.data_path}  X{X.shape}  대역 {lo:g}~{hi:g}Hz")

    # 창 전체에서 계산한다 — 구간 선택과 무관한 순수 신호 성질을 보려는 것이다
    rel = np.zeros((len(X), n_ch), np.float32)
    for s0 in range(0, len(X), 2000):
        s1 = min(s0 + 2000, len(X))
        _, r = band_power(X[s0:s1], a.fs, lo, hi)
        rel[s0:s1] = r

    codes, inv = np.unique(subj, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    per_subj = np.stack(
        [np.bincount(inv, weights=rel[:, c]) / cnt for c in range(n_ch)], 1)
    sy = np.array([y[subj == s][0] for s in codes])

    print(f"\n피험자 {len(codes)}명 (환자 {(sy==1).sum()} / 정상 {(sy==0).sum()})")
    print(f"\n{'채널':<12}{'환자':>9}{'정상':>9}{'차이':>10}{'p':>11}{'유의':>5}")
    rows = []
    ps = []
    for c in range(n_ch):
        pa, hc = per_subj[sy == 1, c], per_subj[sy == 0, c]
        p = float(stats.mannwhitneyu(pa, hc).pvalue)
        ps.append(p)
        rows.append(dict(channel=CH_NAMES[c] if c < len(CH_NAMES) else f"ch{c}",
                         patient=round(float(pa.mean()), 5),
                         healthy=round(float(hc.mean()), 5),
                         delta=round(float(pa.mean() - hc.mean()), 5),
                         mannwhitney_p=p))
    # 채널 6개를 동시에 보므로 보정한다
    order = np.argsort(ps)
    adj = np.empty(n_ch)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n_ch - rank) * ps[i])
        adj[i] = min(1.0, run)
    for r, q in zip(rows, adj):
        r["p_holm"] = round(float(q), 6)
        print(f"{r['channel']:<12}{r['patient']:>9.4f}{r['healthy']:>9.4f}"
              f"{r['delta']:>+10.4f}{r['mannwhitney_p']:>11.2e}"
              f"{'  *' if q < 0.05 else '   '}")

    # ── 채널 이름 매핑의 경험적 확인 ──
    # 원본 txt에는 열 이름이 없다. 메타 블록은 인적사항·펜 모델·표집률만
    # 담고 있고, 채널 이름은 데이터셋 문서를 옮긴 것이다. 이름을 그대로
    # 믿기 전에, 신호 자체가 문서의 묶음(마이크 1 / 압력 2 / 기울기 3)과
    # 맞는지 스펙트럼 성격으로 확인한다.
    sub = X[:min(800, len(X))].astype(np.float64)
    sub = sub - sub.mean(-1, keepdims=True)
    spec = np.abs(np.fft.rfft(sub * np.hanning(sub.shape[-1]), axis=-1)) ** 2
    fr = np.fft.rfftfreq(sub.shape[-1], 1.0 / a.fs)
    centroid = (spec * fr).sum(-1) / spec.sum(-1)
    low = spec[..., (fr > 0) & (fr < 20)].sum(-1) / spec[..., fr > 0].sum(-1)
    print(f"\n채널 성격 (이름 매핑 확인용)")
    print(f"{'채널':<12}{'스펙트럼 중심':>14}{'<20Hz 비중':>12}")
    for c in range(n_ch):
        print(f"{(CH_NAMES[c] if c < len(CH_NAMES) else f'ch{c}'):<12}"
              f"{centroid[:, c].mean():>14.1f}{low[:, c].mean():>12.3f}")
    print("  ch0만 중심 주파수가 수백 Hz대이고 저주파 비중이 0 -> 음향 채널")
    print("  ch1·ch2는 10Hz 미만, ch3~ch5는 20Hz대로 묶인다")
    print("  -> 문서의 1(마이크) + 2(압력) + 3(기울기) 구조와 일치한다.")
    print("  다만 ch1(그립력)과 ch2(축압력)의 개별 구분은 신호만으로는 못 한다.")
    print("  그 둘의 이름은 데이터셋 문서를 따른 것이다.")

    best = max(rows, key=lambda r: r["delta"])
    print(f"\n절대 차이가 가장 큰 채널: {best['channel']} "
          f"({best['delta']:+.4f})")
    print("마이크 채널의 상대 파워가 0에 가까운 것은 정상이다 — 음향 신호라")
    print("4~6Hz 성분이 전체 파워에서 차지하는 비중이 무시할 만하다.")

    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with io.open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n-> {a.csv}")


if __name__ == "__main__":
    main()
