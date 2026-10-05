"""
논문 §1·§3에 들어가는 수치를 원자료에서 다시 계산한다.

왜 필요한가
    기준선 세 줄(57.4 / 63.9 / 73.8%)과 교란 구조 수치는 논문에서 가장 자주
    인용되는 값이고, 심사에서 "그 숫자 어디서 나왔나"를 가장 먼저 묻는 곳이다.
    가이드 문서에 적힌 값을 그대로 옮기는 대신 `subject_meta.csv`와
    `windows_2s.npz`에서 매번 다시 계산해, 문서와 데이터가 어긋나면
    즉시 드러나게 한다.

사용법
    python verify_baselines.py
    python verify_baselines.py --json results/baselines.json
"""
import sys
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
import argparse
import csv
import io
import json
import os

import numpy as np

EXPECTED = {          # 가이드 01·02에 적힌 값. 어긋나면 경고한다.
    "n_subjects": 61, "n_healthy": 35, "n_patient": 26,
    "subject_majority_pct": 57.4, "window_majority_pct": 63.9,
    "age_rule_pct": 73.8, "age_gap_years": 14.7,
    "year2016_n": 21, "r4_n": 40, "r5_n": 41,
}


def main():
    ap = argparse.ArgumentParser(description="기준선·교란 수치 재현")
    ap.add_argument("--meta", default="data/subject_meta.csv")
    ap.add_argument("--npz", default="data/windows_2s.npz")
    ap.add_argument("--age-threshold", type=float, default=52.0)
    ap.add_argument("--json", default=None, help="결과를 JSON으로도 저장")
    a = ap.parse_args()

    rows = list(csv.DictReader(io.open(a.meta, encoding="utf-8")))
    y = np.array([int(r["label"]) for r in rows])
    age = np.array([float(r["age"]) for r in rows])
    yr = np.array([int(r["year"]) for r in rows])
    n = len(rows)

    out = {}
    out["n_subjects"] = n
    out["n_healthy"] = int((y == 0).sum())
    out["n_patient"] = int((y == 1).sum())
    out["subject_majority_pct"] = round(100 * max((y == 0).mean(), (y == 1).mean()), 1)

    # 연령 규칙 — "임계값 이상이면 환자"
    out["age_rule_pct"] = round(100 * (((age >= a.age_threshold).astype(int) == y).mean()), 1)
    best_acc, best_t = max((((age >= t).astype(int) == y).mean(), t)
                           for t in sorted(set(age)))
    out["age_rule_best_pct"] = round(100 * best_acc, 1)
    out["age_rule_best_threshold"] = float(best_t)
    out["age_healthy_mean"] = round(float(age[y == 0].mean()), 1)
    out["age_patient_mean"] = round(float(age[y == 1].mean()), 1)
    out["age_gap_years"] = round(float(age[y == 1].mean() - age[y == 0].mean()), 1)

    # 수집 시기 교란
    m16 = yr == 2016
    out["year2016_n"] = int(m16.sum())
    out["year2016_all_healthy"] = bool((y[m16] == 0).all())
    out["year2016_share_pct"] = round(100 * m16.mean(), 1)

    # 서브셋
    m4 = ~m16
    m5 = age >= 45
    for tag, m in (("r4", m4), ("r5", m5), ("r4r5", m4 & m5)):
        out[f"{tag}_n"] = int(m.sum())
        out[f"{tag}_healthy"] = int((y[m] == 0).sum())
        out[f"{tag}_patient"] = int((y[m] == 1).sum())
        if (y[m] == 0).any() and (y[m] == 1).any():
            out[f"{tag}_age_gap"] = round(
                float(age[m & (y == 1)].mean() - age[m & (y == 0)].mean()), 1)

    # 윈도우 단위 (npz가 있을 때만)
    if os.path.exists(a.npz):
        d = np.load(a.npz, allow_pickle=True)
        wy = d["y"]
        out["n_windows"] = int(len(wy))
        out["window_healthy"] = int((wy == 0).sum())
        out["window_patient"] = int((wy == 1).sum())
        out["window_majority_pct"] = round(
            100 * max((wy == 0).mean(), (wy == 1).mean()), 1)
        out["windows_per_patient_ratio"] = round(
            float((wy == 1).sum() / out["n_patient"]) /
            float((wy == 0).sum() / out["n_healthy"]), 2)

    # ── 출력 ──
    print(f"피험자 {out['n_subjects']}명 "
          f"(정상 {out['n_healthy']} / 환자 {out['n_patient']})")
    print()
    print("기준선 세 줄 — 성능표 맨 위에 둘 것")
    print(f"  피험자 다수결      {out['subject_majority_pct']:5.1f}%")
    if "window_majority_pct" in out:
        print(f"  윈도우 다수결      {out['window_majority_pct']:5.1f}%")
    print(f"  연령 규칙(>={a.age_threshold:g}세) {out['age_rule_pct']:5.1f}%"
          f"   <- 모델이 실제로 넘어야 하는 선")
    print(f"  (임계값 전수 탐색 최적: >={out['age_rule_best_threshold']:g}세에서 "
          f"{out['age_rule_best_pct']:.1f}%)")
    print()
    if "window_majority_pct" in out:
        print(f"지표 역전: 피험자로 세면 정상이 다수이나 윈도우로 세면 환자가 다수 "
              f"(환자 1명당 윈도우가 {out['windows_per_patient_ratio']:.1f}배)")
        print()
    print("교란 구조")
    print(f"  2016년 수집 {out['year2016_n']}명 "
          f"({out['year2016_share_pct']:.0f}%), 전원 정상: {out['year2016_all_healthy']}")
    print(f"  연령 정상 {out['age_healthy_mean']}세 / 환자 "
          f"{out['age_patient_mean']}세 (차 {out['age_gap_years']:+.1f})")
    print()
    print("서브셋")
    for tag, name in (("r4", "R4 (2016 제외)"), ("r5", "R5 (45세 이상)"),
                      ("r4r5", "R4 AND R5")):
        gap = out.get(f"{tag}_age_gap")
        print(f"  {name:16s} {out[tag+'_n']:3d}명 "
              f"(정상 {out[tag+'_healthy']:2d} / 환자 {out[tag+'_patient']:2d})"
              + (f"  연령차 {gap:+.1f}" if gap is not None else ""))
    print(f"\n  -> R4 AND R5는 정상이 {out['r4r5_healthy']}명뿐이라 "
          f"두 교란의 동시 통제가 불가능하다")

    # ── 문서와 대조 ──
    print("\n가이드 문서 값과 대조")
    bad = 0
    for k, exp in EXPECTED.items():
        got = out.get(k)
        if got is None:
            print(f"  {k:24s} 계산 안 됨 (npz 없음?)"); continue
        ok = abs(got - exp) < 0.15 if isinstance(exp, float) else got == exp
        print(f"  {k:24s} 문서 {exp} / 계산 {got}  {'일치' if ok else '<-- 불일치'}")
        bad += 0 if ok else 1
    print(f"\n{'전부 일치한다.' if bad == 0 else f'{bad}건 불일치 - 문서를 고칠 것.'}")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with io.open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n-> {a.json}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
