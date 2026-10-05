"""
필기 획 리듬의 주파수를 직접 잰다 (C).

왜 필요한가
    XAI 결과에서 정상군도 기울기Z 채널에서 4~6Hz 대역이 유의하게 나왔다
    (시드평균 dz +0.58). 이를 "정상적인 필기의 획 리듬도 4~6Hz대이므로,
    이 대역은 떨림의 전유물이 아니다"로 해석했다. **그런데 그 해석은 문헌
    지식에 기댄 것이고, 이 데이터에서 획 주파수를 잰 적이 없다.**

    이 스크립트가 그 빈틈을 메운다. 신호에서 지배 주파수를 직접 찾아,
    정상군의 그것이 실제로 4~6Hz대인지 확인한다.

무엇을 재는가
    창마다 채널별로 스펙트럼 최대점(지배 주파수)을 찾는다. 직류와 초저주파는
    필기의 느린 이동(획이 아니라 손 전체가 옮겨가는 성분)이라 1Hz 미만은
    제외한다.

    그리고 군별로 그 분포를 본다.

읽는 법
    정상군의 지배 주파수가 4~6Hz에 몰린다
        -> "4~6Hz는 정상 필기 리듬이기도 하다"가 데이터로 뒷받침된다.
           그러면 판별의 근거는 "4~6Hz가 있는가"가 아니라 "어느 채널에
           실리는가"라는 해석이 힘을 얻는다.
    정상군의 지배 주파수가 그보다 낮거나 높다
        -> 그 해석을 버리고, 정상군 기울기Z 결과는 다른 설명을 찾아야 한다.

과제별로 나누는 이유
    diadochokinesis(손 빠르게 뒤집기)는 동작 자체가 반복 주기를 가진다.
    spiral·meander(그림 그리기)와 획 리듬이 다를 수밖에 없다. XAI에서도
    이 과제만 부호가 뒤집혔으므로 반드시 분리해서 본다.

사용법
    python stroke_rhythm.py
    python stroke_rhythm.py --channel 5        # 기울기Z만
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

CH = ["ch0 마이크", "ch1 그립력", "ch2 축압력",
      "ch3 기울기X", "ch4 기울기Y", "ch5 기울기Z"]


def dominant_freq(X, fs, fmin, fmax):
    """창별·채널별 지배 주파수 (스펙트럼 최대점).

    1Hz 미만은 제외한다 — 필기 중 손 전체가 종이 위를 옮겨가는 느린 성분이며
    획 리듬이 아니다.
    """
    n = X.shape[-1]
    seg = X - X.mean(axis=-1, keepdims=True)
    spec = np.abs(np.fft.rfft(seg * np.hanning(n), axis=-1)) ** 2
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    band = (fr >= fmin) & (fr <= fmax)
    idx = spec[..., band].argmax(axis=-1)
    return fr[band][idx]


def main():
    ap = argparse.ArgumentParser(description="필기 획 리듬 주파수 측정")
    ap.add_argument("--data-path", default="data/windows_2s.npz")
    ap.add_argument("--fs", type=int, default=1000)
    ap.add_argument("--fmin", type=float, default=1.0)
    ap.add_argument("--fmax", type=float, default=30.0)
    ap.add_argument("--channel", type=int, default=None)
    ap.add_argument("--csv", default="results/stroke_rhythm.csv")
    a = ap.parse_args()

    X, y, subj, task = load_npz(a.data_path)
    n_ch = X.shape[1]
    chans = [a.channel] if a.channel is not None else list(range(n_ch))
    print(f"데이터 {a.data_path}  X{X.shape}  탐색 대역 {a.fmin:g}~{a.fmax:g}Hz")

    # 청크로 계산 (스펙트럼이 크다)
    dom = np.zeros((len(X), n_ch), np.float32)
    step = max(1, int(3e7 // (n_ch * X.shape[2])))
    for c0 in range(0, len(X), step):
        c1 = min(c0 + step, len(X))
        dom[c0:c1] = dominant_freq(X[c0:c1], a.fs, a.fmin, a.fmax)

    # 피험자 평균 후 군 비교 — 환자 1명당 윈도우가 2.4배라 윈도우 단위는 편중
    codes, inv = np.unique(subj, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    sy = np.array([y[subj == s][0] for s in codes])

    rows = []
    print(f"\n{'채널':<12}{'환자 중앙':>10}{'정상 중앙':>10}"
          f"{'정상 4~6Hz 비율':>16}{'p':>10}")
    for c in chans:
        per = np.bincount(inv, weights=dom[:, c], minlength=len(codes)) / cnt
        pa, hc = per[sy == 1], per[sy == 0]
        # 정상군 윈도우 중 지배 주파수가 4~6Hz인 비율
        hw = dom[(y == 0), c]
        frac = float(((hw >= 4) & (hw <= 6)).mean())
        p = float(stats.mannwhitneyu(pa, hc).pvalue)
        rows.append(dict(channel=CH[c], patient_median=round(float(np.median(pa)), 2),
                         healthy_median=round(float(np.median(hc)), 2),
                         healthy_frac_4_6=round(frac, 3), mannwhitney_p=round(p, 5)))
        print(f"{CH[c]:<12}{np.median(pa):>10.2f}{np.median(hc):>10.2f}"
              f"{frac:>16.3f}{p:>10.4f}")

    # 과제별 — diadochokinesis는 동작 자체가 반복 주기를 가진다
    print(f"\n과제별 정상군 지배 주파수 중앙값 (Hz)")
    print(f"{'과제':<18}" + "".join(f"{CH[c][:6]:>9}" for c in chans))
    for t in np.unique(task):
        m = (task == t) & (y == 0)
        print(f"{t:<18}" + "".join(f"{np.median(dom[m, c]):>9.2f}" for c in chans))

    hi = [r for r in rows if r["healthy_frac_4_6"] > 0.25]
    print()
    if hi:
        print("  정상군 윈도우의 25% 이상이 4~6Hz를 지배 주파수로 갖는 채널:")
        for r in hi:
            print(f"    {r['channel']} ({r['healthy_frac_4_6']*100:.0f}%)")
        print("  -> '4~6Hz는 정상 필기 리듬이기도 하다'가 데이터로 뒷받침된다.")
    else:
        print("  정상군의 지배 주파수가 4~6Hz에 몰리지 않는다.")
        print("  -> '정상 필기 획 리듬' 해석을 그대로 쓸 수 없다. 다른 설명이 필요하다.")

    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with io.open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n-> {a.csv}")


if __name__ == "__main__":
    main()
