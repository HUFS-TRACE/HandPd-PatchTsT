"""[B] 사전 점검 — 시간 축 적응적 종료가 성립하는가.

`paper/exp2_protocol.md` §2. 학습 없이 numpy 재계산만으로 실험 2의 성립 여부가
판정된다. 여기서 나쁘면 [C] 캘리브레이션 이후를 실행하지 않는다.

계산
    윈도우별 증거      e_i  = logit(p_i) - c
    드리프트           d_s  = mean(e_i)        한 윈도우당 평균 증거
    필요 관측량        n*_s = A / |d_s|        Wald 근사, A = log((1-β)/α)

유형 (§2-2)
    A 몰표    부호 일치 · n* <= 10     빨리 종료. 시간 절감이 나오는 곳
    B 접전    부호 일치 · n* > 10      오래 봐야 함. 유보 대상
    C 반대    부호 불일치              확신을 갖고 오분류

사전 등록된 판정 기준 (§2-4, 결과 보기 전에 고정)
    C 0~3명    진행
    C 4~7명    조건부 진행 — 유보 옵션과 비대칭 경계 필수
    C 8명 이상  재설계

확률 공급원은 **정적 L1**이다(§1-1). EE 출구보다 일관되게 낫고, 실험 2가
earlyexit.py 에서 독립한다.

실행
    python analyze_drift.py
    python analyze_drift.py --cfg p16d128L6 --alpha 0.05
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load(cfg, sfx="_depth_2s"):
    p = ROOT / "results" / f"probs_{cfg}{sfx}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def drift_table(d, c, A):
    """피험자별 드리프트와 필요 관측량."""
    out = []
    for u in np.unique(d["subject"]):
        m = d["subject"] == u
        e = logit(d["prob"][m]) - c
        ds = e.mean()
        y = int(d["y"][m][0])
        # 라벨이 1이면 증거가 양이어야 맞는 방향이다
        agree = (ds > 0) == (y == 1)
        nstar = A / abs(ds) if abs(ds) > EPS else np.inf
        out.append(dict(subject=u, y=y, n=int(m.sum()), drift=ds,
                        nstar=nstar, agree=agree))
    return out


def classify(rows, thr=10):
    A = [r for r in rows if r["agree"] and r["nstar"] <= thr]
    B = [r for r in rows if r["agree"] and r["nstar"] > thr]
    C = [r for r in rows if not r["agree"]]
    return A, B, C


def main():
    ap = argparse.ArgumentParser(description="[B] 드리프트 사전 점검")
    ap.add_argument("--cfg", default="p16d128L1", help="확률 공급원 (기본 정적 L1)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    a = ap.parse_args()

    d = load(a.cfg)
    if d is None:
        sys.exit(f"확률 파일이 없다: results/probs_{a.cfg}_depth_2s.npz")

    A_bound = np.log((1 - a.beta) / a.alpha)
    # 사전확률 보정 c — 학습 유병률의 로짓. val 이 없으므로 전체 유병률로 근사한다.
    prev = d["y"].mean()
    c = logit(np.array([prev]))[0]
    print(f"확률 공급원 {a.cfg} · 창 {len(d['y']):,} · 피험자 "
          f"{len(np.unique(d['subject']))}명")
    print(f"경계 A = log((1-β)/α) = {A_bound:.3f}  ·  사전확률 보정 c = {c:+.3f} "
          f"(창 양성률 {prev:.3f})\n")

    rows = drift_table(d, c, A_bound)
    n_sub = len(rows)

    # ── 유형 분류: 민감도 확인을 위해 5/10/20 세 기준 (§2-2) ──
    print("유형 분류 — n* 기준을 바꿔 가며")
    print(f"  {'기준':>6}{'A 몰표':>10}{'B 접전':>10}{'C 반대':>10}")
    for thr in (5, 10, 20):
        Aa, Bb, Cc = classify(rows, thr)
        print(f"  {f'n*<={thr}':>6}{len(Aa):>10}{len(Bb):>10}{len(Cc):>10}")
    Aa, Bb, Cc = classify(rows, 10)

    # ── 사전 등록된 판정 (§2-4) ──
    nC = len(Cc)
    verdict = ("진행" if nC <= 3 else
               "조건부 진행 — 유보 옵션과 비대칭 경계 필수" if nC <= 7 else
               "재설계 — 정지 규칙이 아니라 모델·특징 문제")
    print(f"\n사전 등록 판정: C 유형 {nC}명 / {n_sub}명  →  **{verdict}**")

    # ── C 유형 명단 ──
    if Cc:
        print(f"\nC 유형(확신을 갖고 오분류) {len(Cc)}명")
        print(f"  {'구분':<5}{'창수':>7}{'드리프트':>11}{'n*':>9}")
        for r in sorted(Cc, key=lambda r: -abs(r["drift"])):
            print(f"  {'환자' if r['y'] else '정상':<5}{r['n']:>7}"
                  f"{r['drift']:>+11.3f}{r['nstar']:>9.1f}")

    # ── 드리프트 분포: 단봉이면 적응적 종료의 이론적 이득이 0 (§2-3) ──
    dr = np.array([r["drift"] for r in rows])
    ab = np.abs(dr)
    print(f"\n드리프트 |d_s| 분포")
    print(f"  평균 {ab.mean():.3f}  중앙 {np.median(ab):.3f}  "
          f"SD {ab.std(ddof=1):.3f}  범위 {ab.min():.3f}~{ab.max():.3f}")
    q = np.percentile(ab, [10, 25, 50, 75, 90])
    print(f"  분위 10/25/50/75/90  " + " / ".join(f"{v:.2f}" for v in q))
    print(f"  최대/중앙 비 {ab.max()/np.median(ab):.1f}배  "
          f"· 상위25%/하위25% 비 {q[3]/q[1]:.1f}배")

    # ── 필요 관측량 분포 ──
    ns = np.array([min(r["nstar"], 1e4) for r in rows])
    nwin = np.array([r["n"] for r in rows])
    print(f"\n필요 관측량 n* 분포 (사람당 실제 창 {nwin.min()}~{nwin.max()}, "
          f"중앙 {int(np.median(nwin))})")
    for thr in (5, 10, 20, 50):
        print(f"  n* <= {thr:<3} {int((ns<=thr).sum()):>3}명 ({(ns<=thr).mean():>5.1%})")
    print(f"  n* 중앙 {np.median(ns):.1f}  ·  실제 창 중앙의 "
          f"{np.median(ns)/np.median(nwin):.1%}")

    # ── 이론적 절감 상한 ──
    save = 1 - np.minimum(ns, nwin).sum() / nwin.sum()
    print(f"\n이론적 관측량 절감 상한 {save:.1%}")
    print("  (각자 n* 개만 보고 멈춘다고 가정. 실제 SPRT 는 이보다 못하다)")

    # ── |d_s| 가 신뢰도의 대리 지표로 작동하는가 (§2-3) ──
    from scipy import stats
    correct = np.array([r["agree"] for r in rows])
    r_pb, p_pb = stats.pointbiserialr(correct.astype(int), ab)
    print(f"\n|d_s| ↔ 판정 적중  점이연 상관 r={r_pb:+.3f} (p={p_pb:.3f})")
    print("  → 양수여야 드리프트가 신뢰도의 대리 지표로 쓸모 있다")


if __name__ == "__main__":
    main()
