"""V2 — 사람마다 난이도가 다른가.

가이드 05 §M1-V2. ③단계의 본체다.

    깊이·시간 축  →  "메커니즘이 작동하지 않았다"
    사람 축      →  "애초에 배분할 난이도 차이가 없었다"

두 번째가 훨씬 강한 진술이고, 지금까지 직접 잰 적이 없다.

세 가지를 센다
    1. 피험자별 예측 확률의 분포와 표준편차
    2. L1과 L6에서 예측 라벨이 뒤집히는 피험자 비율
    3. 오분류가 소수에게 집중되는지, 전체에 퍼져 있는지

⚠️ 피험자당 윈도우가 82~801개로 10배 차이 난다. 윈도우가 많은 사람의 확률
   분포가 더 안정적으로 보이는 것은 당연한 일이므로, 분산을 그대로 비교하면
   안 된다. 표본 수의 영향을 뺀 기준선을 함께 낸다.

입력은 evaluate.py --save-probs 가 남긴 results/probs_<cfg>_depth_2s.npz 다.
학습을 다시 돌리지 않는다.

실행
    python exp_v2.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load(cfg, sfx="_depth_2s"):
    p = ROOT / "results" / f"probs_{cfg}{sfx}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def per_subject(d):
    """사람 단위로 접는다. (피험자, 평균확률, 라벨, 창수, 창내 표준편차)"""
    su = np.unique(d["subject"])
    out = []
    for u in su:
        m = d["subject"] == u
        out.append((u, d["prob"][m].mean(), int(d["y"][m][0]),
                    int(m.sum()), float(d["prob"][m].std())))
    return out


def gini(x):
    """오분류 집중도. 0이면 완전 균등, 1이면 한 사람에게 몰림."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    ap = argparse.ArgumentParser(description="V2 — 사람 축 난이도 분산")
    ap.add_argument("--shallow", default="p16d128L1")
    ap.add_argument("--deep", default="p16d128L6")
    a = ap.parse_args()

    D = {c: load(c) for c in ("p16d128L1", "p16d128L2", "p16d128L3", "p16d128L6")}
    D = {k: v for k, v in D.items() if v is not None}
    print(f"확률 파일 {list(D)}\n")

    ref = D[a.deep]
    S = per_subject(ref)
    nwin = np.array([s[3] for s in S])
    print(f"피험자 {len(S)}명 · 창 {nwin.sum():,}개 "
          f"(사람당 {nwin.min()}~{nwin.max()}, 중앙 {int(np.median(nwin))})")

    # ── 1. 피험자별 확률 분포 ──
    print("\n[1] 피험자별 평균 확률의 분포")
    P = np.array([s[1] for s in S])
    Y = np.array([s[2] for s in S])
    for lab, nm in ((0, "정상"), (1, "환자")):
        v = P[Y == lab]
        print(f"  {nm}  n={len(v):>2}  평균 {v.mean():.3f}  SD {v.std(ddof=1):.3f}  "
              f"범위 {v.min():.3f}~{v.max():.3f}")
    print(f"  두 군 분리(피험자 단위 AUC 대용): "
          f"Mann-Whitney p={stats.mannwhitneyu(P[Y==1], P[Y==0]).pvalue:.2e}")

    # 경계(0.5)까지의 거리 = 그 사람이 얼마나 애매한가
    margin = np.abs(P - 0.5)
    print(f"\n  경계까지 거리 |p-0.5|  평균 {margin.mean():.3f}  SD {margin.std(ddof=1):.3f}")
    print(f"    0.1 미만(애매)  {(margin<0.1).sum():>2}명 ({(margin<0.1).mean():.1%})")
    print(f"    0.3 이상(확실)  {(margin>=0.3).sum():>2}명 ({(margin>=0.3).mean():.1%})")

    # ── 창 내 산포: 표본 수 영향을 뺀다 ──
    print("\n[1-b] 창 내 확률 표준편차 — 창 수 편차를 보정한다")
    sd_in = np.array([s[4] for s in S])
    r, p = stats.spearmanr(nwin, sd_in)
    print(f"  창 수 ↔ 창내 SD  Spearman r={r:+.3f} (p={p:.3f})")
    print(f"  창내 SD  평균 {sd_in.mean():.3f}  범위 {sd_in.min():.3f}~{sd_in.max():.3f}")
    print("  → r 이 0 근처면 창 수 편차가 분산 해석을 왜곡하지 않는다는 뜻")

    # ── 2. L1 ↔ L6 뒤집힘 ──
    print(f"\n[2] {a.shallow} ↔ {a.deep} 예측 뒤집힘")
    if a.shallow in D:
        A = dict((s[0], s[1]) for s in per_subject(D[a.shallow]))
        B = dict((s[0], s[1]) for s in per_subject(D[a.deep]))
        lab = dict((s[0], s[2]) for s in per_subject(D[a.deep]))
        keys = sorted(set(A) & set(B))
        fa = np.array([A[k] > 0.5 for k in keys])
        fb = np.array([B[k] > 0.5 for k in keys])
        yy = np.array([lab[k] for k in keys])
        flip = fa != fb
        print(f"  뒤집힌 피험자 {flip.sum()} / {len(keys)}명 ({flip.mean():.1%})")
        if flip.any():
            gain = ((fb == yy) & flip).sum()
            loss = ((fa == yy) & flip).sum()
            print(f"    깊은 쪽이 고친 것 {gain}명 · 망친 것 {loss}명 → 순이득 {gain-loss:+d}명")
        d_abs = np.abs(np.array([A[k] for k in keys]) - np.array([B[k] for k in keys]))
        print(f"  확률 변화폭 |L1-L6|  평균 {d_abs.mean():.3f}  최대 {d_abs.max():.3f}")

    # ── 3. 오분류 집중도 ──
    print("\n[3] 오분류가 소수에 몰리는가")
    print(f"  {'구성':<12}{'틀린 사람':>9}{'창 오류율 Gini':>16}{'상위20% 점유':>14}")
    for cfg, d in D.items():
        s = per_subject(d)
        wrong_sub = sum(1 for u, p_, y_, n_, sd_ in s if (p_ > 0.5) != bool(y_))
        # 사람별 창 단위 오류율
        err = []
        for u, p_, y_, n_, sd_ in s:
            m = d["subject"] == u
            err.append(((d["prob"][m] > 0.5).astype(int) != d["y"][m]).mean())
        err = np.array(err)
        top = np.sort(err)[::-1][:max(1, len(err) // 5)].sum() / max(err.sum(), 1e-9)
        print(f"  {cfg:<12}{wrong_sub:>6}/{len(s):<3}{gini(err):>16.3f}{top:>14.1%}")
    print("  → Gini 가 낮고 상위20% 점유가 20% 근처면 오류가 고르게 퍼진 것")

    # ── 과제별 (05 §② — diadochokinesis 는 성격이 다르다) ──
    print("\n[4] 과제별 경계 거리 (참조 모델)")
    for t in sorted(set(ref["task"].tolist())):
        m = ref["task"] == t
        su = np.unique(ref["subject"][m])
        pm = np.array([ref["prob"][m & (ref["subject"] == u)].mean() for u in su])
        print(f"  {t:<18}피험자 {len(su):>2}  |p-0.5| 평균 {np.abs(pm-0.5).mean():.3f}")


if __name__ == "__main__":
    main()
