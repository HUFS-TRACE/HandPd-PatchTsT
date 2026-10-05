"""
X2 결과를 시드 3개에 걸쳐 종합한다.

왜 단순 평균이 아닌가 — 시드는 독립 표본이 아니다
    `GroupKFold`는 shuffle을 하지 않으므로 **시드를 바꿔도 test fold 구성이
    그대로다.** 달라지는 것은 모델 초기화와 val 분할뿐이다. 따라서 시드 3개는
    독립 3표본이 아니라 **같은 피험자 집합에 대한 반복 측정**이다.

    시드별 dz를 3개 모아 "n=3"으로 검정하면 표준오차를 크게 과소추정한다.
    그래서 두 가지를 따로 낸다.

    1) 시드별 dz 표 — 재현성을 눈으로 확인한다. 부호가 흔들리면 결론이 아니다
    2) 시드 평균 후 단일 검정 — 피험자별 차이값을 시드에 걸쳐 평균한 뒤
       그 하나의 벡터로 검정한다. 반복 측정을 평균으로 접는 것이라
       독립 단위는 여전히 피험자다

사용법
    python summarize_xai_seeds.py --configs d64L2,p16d128L1 --seeds 42,1,7
    python summarize_xai_seeds.py --channel ch2 --win-ms 1000

산출물
    results/xai_seed_summary.csv       설정×군×채널별 시드별 dz + 종합
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


def load_subj(cfg, seed, win, importance="abs"):
    """피험자별 (선택 − 비선택) 차이값을 {(군, 채널): {피험자: 값}}으로."""
    suf = "_abs" if importance == "abs" else ""
    path = os.path.join(
        RESULT_DIR, f"bandpower_{cfg}_s{seed}_w{win}ms{suf}_subj.csv")
    if not os.path.exists(path):
        return None
    out = {}
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        key = (r["group"], r["channel"])
        out.setdefault(key, {})[r["subject"]] = (
            float(r["sel_rel"]) - float(r["non_rel"]))
    return out


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser(description="X2 시드 종합")
    ap.add_argument("--configs", default="d64L2,p16d128L1")
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--win-ms", type=int, default=1000)
    ap.add_argument("--importance", default="abs", choices=["abs", "signed"])
    ap.add_argument("--channel", default=None,
                    help="한 채널만 볼 때. 비우면 전체")
    ap.add_argument("--csv", default="results/xai_seed_summary.csv")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    rows = []

    for cfg in [c.strip() for c in a.configs.split(",")]:
        per_seed = {}
        missing = []
        for s in seeds:
            d = load_subj(cfg, s, a.win_ms, a.importance)
            if d is None:
                missing.append(s)
            else:
                per_seed[s] = d
        if not per_seed:
            print(f"[{cfg}] 시드 결과가 하나도 없음 - 건너뜀")
            continue
        if missing:
            print(f"[{cfg}] 시드 {missing} 결과 없음 - 있는 것만 종합한다")

        keys = sorted(set().union(*[set(d) for d in per_seed.values()]))
        if a.channel:
            keys = [k for k in keys if k[1] == a.channel]

        for grp, ch in keys:
            # 시드별 dz
            dzs = {}
            for s, d in per_seed.items():
                v = np.array(list(d.get((grp, ch), {}).values()))
                dzs[s] = (v.mean() / v.std(ddof=1)) if len(v) > 1 and v.std(ddof=1) > 0 else np.nan

            # 시드 평균 후 단일 검정 — 피험자별로 시드에 걸쳐 평균
            subs = sorted(set().union(
                *[set(d.get((grp, ch), {})) for d in per_seed.values()]))
            avg = []
            for sub in subs:
                vals = [d[(grp, ch)][sub] for d in per_seed.values()
                        if sub in d.get((grp, ch), {})]
                if vals:
                    avg.append(np.mean(vals))
            avg = np.array(avg)
            if len(avg) > 1 and avg.std(ddof=1) > 0:
                dz_avg = float(avg.mean() / avg.std(ddof=1))
                w_p = float(stats.wilcoxon(avg, np.zeros_like(avg)).pvalue)
            else:
                dz_avg, w_p = np.nan, np.nan
            pos = int((avg > 0).sum())

            vals = [dzs[s] for s in seeds if s in dzs and np.isfinite(dzs[s])]
            same_sign = (len(vals) > 0 and
                         (all(v > 0 for v in vals) or all(v < 0 for v in vals)))

            rows.append(dict(
                config=cfg, win_ms=a.win_ms, importance=a.importance,
                group=grp, channel=ch, n_subjects=len(avg),
                **{f"dz_s{s}": (round(float(dzs[s]), 3)
                                if s in dzs and np.isfinite(dzs[s]) else "")
                   for s in seeds},
                dz_seedavg=round(dz_avg, 3) if np.isfinite(dz_avg) else "",
                wilcoxon_p=w_p, subj_positive=f"{pos}/{len(avg)}",
                sign_stable=same_sign, n_seeds=len(vals)))

    if not rows:
        print("종합할 결과가 없다"); return

    # 시드 평균 검정에 Holm 보정 (설정별로 채널×군 가족)
    for cfg in {r["config"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["config"] == cfg]
        ps = [rows[i]["wilcoxon_p"] for i in idx]
        ok = [i for i, p in zip(idx, ps) if np.isfinite(p)]
        if ok:
            adj = holm([rows[i]["wilcoxon_p"] for i in ok])
            for i, q in zip(ok, adj):
                rows[i]["p_holm"] = round(float(q), 6)
                rows[i]["sig"] = "*" if q < 0.05 else ""
    for r in rows:
        r.setdefault("p_holm", ""); r.setdefault("sig", "")
        r["wilcoxon_p"] = (round(r["wilcoxon_p"], 6)
                           if np.isfinite(r["wilcoxon_p"]) else "")

    sc = a.seeds.split(",")
    print(f"\n창 {a.win_ms}ms · 중요도 {a.importance}"
          + (f" · 채널 {a.channel}" if a.channel else ""))
    print(f"{'설정':<11}{'군':>5}{'ch':>5}"
          + "".join(f"{'dz s'+s:>9}" for s in sc)
          + f"{'시드평균':>9}{'p(Holm)':>10}{'부호일치':>9}{'양의피험자':>11}")
    for r in rows:
        print(f"{r['config']:<11}{r['group']:>5}{r['channel'][2:]:>5}"
              + "".join(f"{str(r.get('dz_s'+s, '')):>9}" for s in sc)
              + f"{str(r['dz_seedavg']):>9}{str(r['p_holm']):>10}"
              + f"{('일치' if r['sign_stable'] else '흔들림'):>9}"
              + f"{r['subj_positive']:>11}{r['sig']}")

    print("\n판정 규칙")
    print("  시드 3개에서 부호가 흔들리면 결론으로 쓰지 않는다.")
    print("  시드는 fold 분할을 바꾸지 않으므로 독립 3표본이 아니다 —")
    print("  '5 fold x 3 반복'으로 쓰고, 검정은 시드 평균 후 피험자 단위로 한다.")

    if a.csv:
        with io.open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n-> {a.csv}")


if __name__ == "__main__":
    main()
