"""V2-b — 틀리는 사람이 누구인가.

V2에서 오분류가 61명 중 5명에 몰린다는 것을 봤다(Gini .45, 상위 20%가 오류의 44%).
그 5명이 병리적으로 어려운 사람인지, 아니면 교란(수집 시기·나이) 쪽 사람인지를
가린다.

    5명이 2016년·젊은 정상군에 몰려 있으면   → 교란 신호
    나이·연도와 무관하게 흩어져 있으면        → 진짜 어려운 표본

②단계(R4·R5)의 예고편이다. 학습을 다시 돌리지 않는다.

⚠️ 5명 규모라 통계적 검정이 성립하지 않는다. 순열검정으로 "우연히 이 정도로
   몰릴 확률"만 낸다. 시드 42 하나의 결과이므로 시드 1·7이 나오면 갱신한다.

실행
    python exp_v2b.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_meta():
    """subject_meta.csv → npz의 subject_id 를 키로."""
    m = {}
    with open(ROOT / "data" / "subject_meta.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[r["npz_subject_id"]] = dict(age=float(r["age"]),
                                          year=int(r["year"]),
                                          label=int(r["label"]),
                                          folders=r.get("folders", ""))
    return m


def per_subject(d):
    su = np.unique(d["subject"])
    return {u: (d["prob"][d["subject"] == u].mean(),
                int(d["y"][d["subject"] == u][0]),
                int((d["subject"] == u).sum())) for u in su}


def perm_p(values, flags, n=20000, seed=0):
    """flags 로 고른 부분집합의 평균이 우연보다 극단적일 확률 (양측)."""
    rng = np.random.default_rng(seed)
    k = flags.sum()
    if k == 0:
        return np.nan
    obs = values[flags].mean() - values[~flags].mean()
    cnt = 0
    for _ in range(n):
        idx = rng.permutation(len(values))
        d = values[idx[:k]].mean() - values[idx[k:]].mean()
        if abs(d) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser(description="V2-b — 오분류 피험자의 정체")
    ap.add_argument("--cfg", default="p16d128L6")
    a = ap.parse_args()

    z = np.load(ROOT / "results" / f"probs_{a.cfg}_depth_2s.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    meta = load_meta()
    S = per_subject(d)

    subs = sorted(S)
    P = np.array([S[u][0] for u in subs])
    Y = np.array([S[u][1] for u in subs])
    N = np.array([S[u][2] for u in subs])
    age = np.array([meta[u]["age"] for u in subs])
    yr = np.array([meta[u]["year"] for u in subs])
    wrong = (P > 0.5).astype(int) != Y

    print(f"참조 모델 {a.cfg} · 피험자 {len(subs)}명 · 틀린 사람 {wrong.sum()}명\n")

    print("틀린 피험자")
    print(f"  {'구분':<5}{'나이':>5}{'연도':>7}{'창수':>7}{'평균확률':>10}   폴더")
    for i in np.where(wrong)[0]:
        print(f"  {'환자' if Y[i] else '정상':<5}{age[i]:>5.0f}{yr[i]:>7}"
              f"{N[i]:>7}{P[i]:>10.3f}   {meta[subs[i]]['folders']}")

    print("\n맞힌 사람과 비교")
    print(f"  {'':10}{'틀린 %d명' % wrong.sum():>12}{'맞힌 %d명' % (~wrong).sum():>12}"
          f"{'전체':>10}{'순열 p':>10}")
    for nm, v in (("나이", age), ("수집연도", yr.astype(float)),
                  ("창 수", N.astype(float))):
        print(f"  {nm:<10}{v[wrong].mean():>12.1f}{v[~wrong].mean():>12.1f}"
              f"{v.mean():>10.1f}{perm_p(v, wrong):>10.3f}")

    print("\n라벨 구성")
    print(f"  틀린 사람 중 정상 {int((Y[wrong]==0).sum())} / 환자 {int((Y[wrong]==1).sum())}")
    print(f"  전체       정상 {int((Y==0).sum())} / 환자 {int((Y==1).sum())}")

    print("\n연도별 오분류율")
    print(f"  {'연도':>6}{'인원':>6}{'틀림':>6}{'비율':>8}")
    for u in sorted(set(yr.tolist())):
        m = yr == u
        print(f"  {u:>6}{m.sum():>6}{int(wrong[m].sum()):>6}{wrong[m].mean():>8.1%}")

    print("\n나이 구간별 오분류율")
    print(f"  {'구간':>10}{'인원':>6}{'틀림':>6}{'비율':>8}")
    for lo, hi in ((0, 45), (45, 55), (55, 65), (65, 100)):
        m = (age >= lo) & (age < hi)
        if m.sum():
            print(f"  {f'{lo}~{hi}':>10}{m.sum():>6}{int(wrong[m].sum()):>6}"
                  f"{wrong[m].mean():>8.1%}")

    # 경계 근처(애매한) 사람도 같이 본다 — 틀린 것보다 표본이 크다
    margin = np.abs(P - 0.5)
    amb = margin < 0.1
    print(f"\n경계 근처(|p-0.5|<0.1) {amb.sum()}명 — 표본이 더 크므로 함께 본다")
    print(f"  {'':10}{'애매 %d명' % amb.sum():>12}{'명확 %d명' % (~amb).sum():>12}{'순열 p':>10}")
    for nm, v in (("나이", age), ("수집연도", yr.astype(float))):
        print(f"  {nm:<10}{v[amb].mean():>12.1f}{v[~amb].mean():>12.1f}"
              f"{perm_p(v, amb):>10.3f}")

    print("\n⚠️ 시드 42 하나, 틀린 사람 %d명 규모다. 순열 p 는 참고용이다."
          % wrong.sum())


if __name__ == "__main__":
    main()
