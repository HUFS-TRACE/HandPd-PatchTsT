"""
occlusion 중요도가 RevIN 재정규화의 부산물인지 잰다 (B).

무엇이 문제인가
    이 모델은 RevIN을 쓴다 — 창 하나를 채널별로 시간축 평균·표준편차로
    정규화한 뒤 인코더에 넣는다. 그런데 occlusion은 구간 하나를 평균값으로
    덮는다. **덮인 구간이 창의 표준편차에 기여하던 만큼 그 통계가 바뀌고,
    RevIN이 창 전체를 다시 스케일한다.** 즉 가린 구간뿐 아니라 가리지 않은
    구간까지 모델에 다르게 보인다.

    그래서 "진폭 SD가 큰 구간의 중요도가 높다"는 관측이, 모델이 그 구간을
    실제로 중요하게 쓰기 때문인지 아니면 그 구간을 가릴 때 정규화가 크게
    흔들리기 때문인지 구별되지 않는다.

무엇을 재는가
    구간 g를 가렸을 때 창의 표준편차가 얼마나 변하는지를 순수 산술로
    계산한다(모델을 쓰지 않는다).

        delta_sigma(g) = | sigma(가리기 전) - sigma(가린 뒤) | / sigma(가리기 전)

    그리고 이것이 occlusion 중요도 |dp|를 얼마나 설명하는지 본다.
    윈도우 안에서 위치들 사이의 순위 상관(Spearman)을 쓴다 — 윈도우마다
    척도가 다르므로 절대값 비교는 뜻이 없다.

읽는 법
    상관이 높다  -> 중요도의 상당 부분이 정규화 흔들림으로 설명된다.
                    절차적 교란이며 방법 절에 반드시 적어야 한다.
    상관이 낮다  -> 중요도는 정규화와 별개다. 교란 걱정을 덜 수 있다.

    어느 쪽이든 **군 간 차이는 이 교란으로 설명되지 않는다.** 마스킹 절차는
    라벨을 모르므로 절차적 효과는 군에 무관하게 나타나기 때문이다.

창 길이 주의 — 1000ms로 돌리지 말 것
    Spearman은 순위 상관이라 표본이 적으면 값이 이산적으로 튄다. 1000ms 창은
    위치가 3개뿐이어서 상관이 {-1, -0.5, +0.5, +1} 네 값밖에 못 가지고,
    중앙값이 기계적으로 0.500에 고정된다. **200ms 창(위치 19개)을 쓴다.**

사용법
    python revin_confound.py --configs d64L2 --seed 42 --win-ms 200
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

RESULT_DIR = "results"
CH = ["ch0 마이크", "ch1 그립력", "ch2 축압력",
      "ch3 기울기X", "ch4 기울기Y", "ch5 기울기Z"]


def sigma_shift(X, pos, L):
    """구간을 평균으로 덮었을 때 창 표준편차의 상대 변화량.

    RevIN은 창 전체의 표준편차로 나누므로, 이 값이 클수록 가리지 않은
    구간까지 크게 다시 스케일된다. 모델을 쓰지 않는 순수 산술이다.
    """
    n_win, n_ch, T = X.shape
    out = np.zeros((n_win, n_ch, len(pos)), np.float32)
    base = X.std(axis=-1)                                  # (N, C)
    for pi, p in enumerate(pos):
        m = X.copy()
        seg_mean = X.mean(axis=-1, keepdims=True)          # occlusion과 같은 채움
        m[:, :, p:p + L] = seg_mean
        out[:, :, pi] = np.abs(base - m.std(axis=-1)) / (base + 1e-12)
        del m
    return out


def main():
    ap = argparse.ArgumentParser(description="RevIN 재정규화 교란 정량화")
    ap.add_argument("--configs", default="d64L2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--win-ms", type=int, default=200,
                    help="위치 수가 많아야 순위 상관이 뜻을 가진다. "
                         "1000ms는 위치 3개라 값이 0.5로 고정된다")
    ap.add_argument("--data-path", default="data/windows_2s.npz")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    npz = os.path.join(RESULT_DIR,
                       f"occlusion_{a.configs}_s{a.seed}_w{a.win_ms}ms.npz")
    if not os.path.exists(npz):
        raise SystemExit(f"없음: {npz}")
    d = np.load(npz, allow_pickle=True)
    drop, pos, L = d["drop"], d["pos_start"], int(d["win_len"])
    subj, yy = d["subject"], d["y"]

    from dataset import load_npz
    X, y, subject_id, task = load_npz(a.data_path)
    if len(X) != len(drop):
        X = X[np.isin(subject_id, np.unique(subj))]
    assert len(X) == len(drop)
    n_ch = drop.shape[1] - 1
    print(f"[{a.configs} s{a.seed} {a.win_ms}ms] 윈도우 {len(X)} · 위치 {len(pos)}")

    # 청크로 나눠 계산 (X.copy()가 커서 메모리를 먹는다)
    ss = np.zeros((len(X), n_ch, len(pos)), np.float32)
    step = max(1, int(4e7 // (n_ch * X.shape[2])))
    for c0 in range(0, len(X), step):
        c1 = min(c0 + step, len(X))
        ss[c0:c1] = sigma_shift(X[c0:c1], pos, L)

    imp = np.abs(drop[:, :n_ch, :])          # 주 중요도 정의 |dp|
    rows = []
    print(f"\n{'채널':<12}{'환자 rho':>10}{'정상 rho':>10}{'전체 rho':>10}"
          f"{'설명력 R2':>11}")
    for c in range(n_ch):
        r_all = []
        per = {}
        for grp, name in ((1, "환자"), (0, "정상")):
            m = yy == grp
            # 윈도우마다 위치 순위 상관을 구하고 그 분포의 중앙값을 쓴다.
            # 윈도우를 뭉쳐 한 번에 상관을 내면 윈도우 간 척도 차이가 섞인다.
            rs = []
            for i in np.where(m)[0]:
                if len(pos) < 3:
                    continue
                r = stats.spearmanr(imp[i, c], ss[i, c]).statistic
                if np.isfinite(r):
                    rs.append(r)
            per[name] = float(np.median(rs)) if rs else np.nan
            r_all.extend(rs)
        med = float(np.median(r_all)) if r_all else np.nan
        rows.append(dict(channel=CH[c], rho_patient=round(per["환자"], 3),
                         rho_healthy=round(per["정상"], 3),
                         rho_all=round(med, 3), r2=round(med ** 2, 3)))
        print(f"{CH[c]:<12}{per['환자']:>10.3f}{per['정상']:>10.3f}"
              f"{med:>10.3f}{med**2:>11.3f}")

    mx = max(abs(r["rho_all"]) for r in rows)
    print(f"\n최대 |rho| = {mx:.3f}  (설명력 R2 = {mx**2:.3f})")
    if mx ** 2 > 0.3:
        print("  -> 중요도의 상당 부분이 정규화 흔들림으로 설명된다.")
        print("     절차적 교란이며 방법 절에 반드시 적는다.")
    else:
        print("  -> 정규화 흔들림으로 설명되는 몫이 작다.")
        print("     중요도는 이 교란과 대체로 별개다.")
    print("\n  어느 쪽이든 군 간 차이는 이것으로 설명되지 않는다 —")
    print("  마스킹 절차는 라벨을 모르므로 절차적 효과는 군에 무관하다.")

    out = a.csv or os.path.join(
        RESULT_DIR, f"revin_confound_{a.configs}_s{a.seed}_w{a.win_ms}ms.csv")
    with io.open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
