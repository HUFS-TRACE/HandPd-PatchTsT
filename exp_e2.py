"""E2 — 피처를 늘리면 언제 멈추나.

가이드 05 §M1-E2. E1이 "하나로 얼마나 되나"라면 이건 "몇 개면 충분한가"다.
곡선이 평평해지는 지점이 딥러닝 성능과 만나면 저차원 주장이 완성되고,
한참 아래에서 멈추면 갈래 3이 굳는다.

⚠️ 전진 선택의 편향
    같은 폴드로 고르고 같은 폴드로 평가하면 성능이 부풀려진다. 그래서 두 가지를
    함께 보고한다.

        greedy   fold 안에서만 고른다(중첩). 편향 없는 곡선
        fixed    단일 성능 순서로 고정해 넣는다. 순서가 데이터에 안 휘둘림

    둘이 크게 다르면 선택 자체가 노이즈를 타고 있다는 뜻이다.

프로토콜은 E1과 동일 — protocol.get_folds(), SEEDS = 42, 1, 7,
fold 안에서 시드 평균 → fold 사이 산포로 신뢰구간.

실행
    python exp_e2.py
    python exp_e2.py --max-feat 12 --out results/e2.csv
"""
import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dataset import load_npz                    # noqa: E402
from protocol import SEEDS, get_folds           # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FS = 1000
CH = ["마이크", "그립력", "축압력", "기울기X", "기울기Y", "기울기Z"]


def build_features(X):
    """후보 피처 은행. 가이드 05 §E2의 목록을 그대로 만든다."""
    f, P = signal.welch(X, fs=FS, nperseg=min(1024, X.shape[-1]), axis=-1)
    names, cols = [], []

    def band(lo, hi, tag):
        m = (f >= lo) & (f <= hi)
        v = np.log10(np.trapezoid(P[..., m], f[m], axis=-1) + 1e-20)
        for c in range(X.shape[1]):
            names.append(f"{tag}·{CH[c]}")
            cols.append(v[:, c])

    band(0.5, 4, "저주파")
    band(4, 6, "떨림")
    band(6, 12, "중주파")
    band(12, 50, "고주파")

    d = np.diff(X, axis=-1)                      # 속도
    dd = np.diff(d, axis=-1)                     # 가속도
    for tag, arr, fn in (("속도SD", d, np.std), ("가속SD", dd, np.std),
                         ("진폭SD", X, np.std), ("첨도", X, None)):
        for c in range(X.shape[1]):
            if fn is None:
                z = X[:, c] - X[:, c].mean(-1, keepdims=True)
                s = z.std(-1) + 1e-12
                v = ((z / s[:, None]) ** 4).mean(-1)
            else:
                v = fn(arr[:, c], axis=-1)
            names.append(f"{tag}·{CH[c]}")
            cols.append(v)

    # 정지 시간 비율 — 속도가 그 기록의 10% 분위 아래인 표본의 비율
    thr = np.percentile(np.abs(d), 10, axis=-1, keepdims=True)
    for c in range(X.shape[1]):
        names.append(f"정지비율·{CH[c]}")
        cols.append((np.abs(d[:, c]) < thr[:, c]).mean(-1))

    return np.column_stack(cols).astype(np.float32), names


def subject_auc(prob, y, sid, te):
    su = np.unique(sid[te])
    p = np.array([prob[sid[te] == u].mean() for u in su])
    t = np.array([y[te][sid[te] == u][0] for u in su])
    return roc_auc_score(t, p) if len(set(t)) > 1 else np.nan


def fit_auc(F, y, tr, te, sid):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000,
                                           class_weight="balanced"))
    clf.fit(F[tr], y[tr])
    p = clf.predict_proba(F[te])[:, 1]
    return roc_auc_score(y[te], p), subject_auc(p, y, sid, te)


def folds_all(sid, y, n_folds):
    """(seed, fold, tr, va, te) 를 한 번만 만들어 재사용한다."""
    out = []
    for seed in SEEDS:
        a = SimpleNamespace(n_folds=n_folds, val_size=0.1,
                            stratified=False, seed=seed)
        for k, (tr, va, te) in enumerate(get_folds(sid, y, a)):
            out.append((seed, k, tr, va, te))
    return out


def summarize(vals):
    """vals: {(seed,fold): 값}. fold 안 시드평균 → fold 사이 90% CI."""
    ks = sorted({f for _, f in vals})
    per = np.array([np.nanmean([v for (s, f), v in vals.items() if f == k])
                    for k in ks])
    m, sd = np.nanmean(per), np.nanstd(per, ddof=1)
    return m, stats.t.ppf(0.95, len(per) - 1) * sd / np.sqrt(len(per))


def main():
    ap = argparse.ArgumentParser(description="E2 — 피처 수 대비 성능 곡선")
    ap.add_argument("--data", default="data/windows_2s.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-feat", type=int, default=10)
    ap.add_argument("--out", default="results/e2.csv")
    a = ap.parse_args()

    X, y, sid, task = load_npz(a.data)
    F, names = build_features(X)
    print(f"창 {len(y):,} · 피험자 {len(np.unique(sid))} · 후보 피처 {len(names)}개")
    print(f"fold {a.folds} × seed {SEEDS}\n")

    FD = folds_all(sid, y, a.folds)
    rows = []

    # ── fixed: 단일 성능 순위로 고정 순서 ──
    single = {}
    for j in range(len(names)):
        v = {(s, k): fit_auc(F[:, [j]], y, tr, te, sid)[0]
             for s, k, tr, va, te in FD}
        single[j] = summarize(v)[0]
    order = sorted(single, key=single.get, reverse=True)
    print("단일 성능 상위 8개")
    for j in order[:8]:
        print(f"  {names[j]:<18}{single[j]:.4f}")

    # ── greedy: fold 안에서만 고른다(중첩). 한 번 전진하며 매 단계 기록한다 ──
    #    k 마다 처음부터 다시 고르면 같은 계산을 max_feat 번 한다. 한 번이면 충분하다.
    gw = {k: {} for k in range(1, a.max_feat + 1)}
    gs = {k: {} for k in range(1, a.max_feat + 1)}
    picks = {k: [] for k in range(1, a.max_feat + 1)}
    for s, f, tr, va, te in FD:
        sel = []
        for k in range(1, a.max_feat + 1):
            best, bj = -1, None
            for j in range(len(names)):
                if j in sel:
                    continue
                au, _ = fit_auc(F[:, sel + [j]], y, tr, va, sid)   # 고르기는 val 로
                if au > best:
                    best, bj = au, j
            sel.append(bj)
            picks[k].append(names[bj])
            w, sb = fit_auc(F[:, sel], y, tr, te, sid)             # 평가는 test 로
            gw[k][(s, f)], gs[k][(s, f)] = w, sb
        print(f"  greedy seed {s} fold {f} 완료", flush=True)

    print(f"\n{'k':>3}{'fixed 창AUC':>16}{'greedy 창AUC':>18}"
          f"{'greedy 피험자':>16}   greedy 가 자주 고른 피처")
    for k in range(1, a.max_feat + 1):
        chosen_fixed = order[:k]
        vf = {(s, f): fit_auc(F[:, chosen_fixed], y, tr, te, sid)[0]
              for s, f, tr, va, te in FD}
        fm, fc = summarize(vf)
        gm, gc = summarize(gw[k])
        sm, sc = summarize(gs[k])
        top = max(set(picks[k]), key=picks[k].count)
        print(f"{k:>3}{fm:>9.4f} ±{fc:<5.4f}{gm:>11.4f} ±{gc:<5.4f}"
              f"{sm:>9.4f} ±{sc:<5.4f}   {top}")
        rows.append(dict(k=k, fixed_win=fm, fixed_ci=fc,
                         greedy_win=gm, greedy_ci=gc,
                         greedy_subj=sm, greedy_subj_ci=sc,
                         fixed_feats="|".join(names[j] for j in chosen_fixed)))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장: {a.out}")
    print("참조 모델 p16d128L6 윈도우 AUC .8629 (08 §3-2) — 곡선이 이 선에 닿는지가 판정")


if __name__ == "__main__":
    main()
