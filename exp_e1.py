"""E1 — 떨림 대역 파워 하나로 어디까지 되나.

가이드 05 §M1-E1. ④단계(왜 난이도 차이가 없나)의 핵심 실험이다.
피처 하나짜리 로지스틱 회귀가 딥러닝에 근접하면 이 과제가 저차원이라는 직접
증거가 되고, 그것이 Early Exit 실패를 설명한다.

프로토콜 (08 §2)
    분할   protocol.get_folds() 를 반드시 거친다. sklearn 쪽에서 GroupKFold 를
           따로 만들면 val 분리 방식이 달라져 딥러닝과 나란히 놓을 수 없다.
    시드   SEEDS = 42, 1, 7
    보고   fold 안에서 시드 평균 → fold 사이 산포로 신뢰구간 (독립 단위는 fold)

피처
    창마다 채널별 4~6Hz 대역 파워. Welch PSD 로 구한다.
    ⚠️ 마이크(채널 1)는 떨림 대역의 물리적 의미가 다르므로 별도 보고한다.

판정 (05 §E1)
    딥러닝 근접(AUC .82 이상) → 저차원 가설 지지 → 갈래 1
    크게 낮음(.70대)          → 저차원 가설 기각 → 갈래 3

실행
    python exp_e1.py
    python exp_e1.py --band 4 6 --out results/e1.csv
"""
import argparse
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

FS = 1000                                        # 전 파일 동일 (헤더 Samplerate)
CH = ["마이크", "그립력", "축압력", "기울기X", "기울기Y", "기울기Z"]


def band_power(X, lo, hi, fs=FS):
    """창마다 채널별 lo~hi Hz 대역 파워. (N, C) 로 돌려준다."""
    f, P = signal.welch(X, fs=fs, nperseg=min(1024, X.shape[-1]), axis=-1)
    m = (f >= lo) & (f <= hi)
    return np.trapezoid(P[..., m], f[m], axis=-1)


def subject_auc(prob, y, sid, te):
    """피험자 단위 AUC. 사람마다 창 확률을 평균해 한 점으로 만든다."""
    su = np.unique(sid[te])
    p = np.array([prob[sid[te] == u].mean() for u in su])
    t = np.array([y[te][sid[te] == u][0] for u in su])
    return roc_auc_score(t, p) if len(set(t)) > 1 else np.nan


def run(F, y, sid, n_folds=5):
    """(fold, seed) 원본을 남긴다. 08 §2 보고 규칙."""
    rows = []
    for seed in SEEDS:
        args = SimpleNamespace(n_folds=n_folds, val_size=0.1,
                               stratified=False, seed=seed)
        for k, (tr, va, te) in enumerate(get_folds(sid, y, args)):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000,
                                                   class_weight="balanced"))
            clf.fit(F[tr], y[tr])
            prob = clf.predict_proba(F[te])[:, 1]
            rows.append(dict(seed=seed, fold=k,
                             win_auc=roc_auc_score(y[te], prob),
                             subj_auc=subject_auc(prob, y, sid, te)))
    return rows


def summarize(rows, key):
    """fold 안에서 시드 평균 → fold 사이 산포로 90% CI (독립 단위는 fold)."""
    folds = sorted({r["fold"] for r in rows})
    per = np.array([np.nanmean([r[key] for r in rows if r["fold"] == f])
                    for f in folds])
    m, sd = np.nanmean(per), np.nanstd(per, ddof=1)
    ci = stats.t.ppf(0.95, len(per) - 1) * sd / np.sqrt(len(per))
    seed_sd = np.nanstd([np.nanmean([r[key] for r in rows if r["seed"] == s])
                         for s in SEEDS], ddof=1)
    return m, ci, seed_sd


def main():
    ap = argparse.ArgumentParser(description="E1 — 떨림 대역 파워 단일 피처")
    ap.add_argument("--data", default="data/windows_2s.npz")
    ap.add_argument("--band", nargs=2, type=float, default=[4, 6])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results/e1.csv")
    a = ap.parse_args()

    X, y, sid, task = load_npz(a.data)
    print(f"창 {len(y):,} · 피험자 {len(np.unique(sid))} · "
          f"대역 {a.band[0]:g}~{a.band[1]:g}Hz · fold {a.folds} × seed {SEEDS}")

    BP = band_power(X, *a.band)                  # (N, 6)
    logBP = np.log10(BP + 1e-20)                 # 파워는 로그 스케일이 자연스럽다

    out = []
    print(f"\n{'구성':<26}{'창 AUC':>18}{'피험자 AUC':>20}{'시드SD':>9}")
    combos = ([(f"단일: {c}", [i]) for i, c in enumerate(CH)]
              + [("기울기 3축", [3, 4, 5]),
                 ("압력 2종(그립+축)", [1, 2]),
                 ("마이크 제외 5채널", [1, 2, 3, 4, 5]),
                 ("전 채널 6개", list(range(6)))])
    for name, idx in combos:
        rows = run(logBP[:, idx], y, sid, a.folds)
        wm, wc, ws = summarize(rows, "win_auc")
        sm, sc, _ = summarize(rows, "subj_auc")
        print(f"{name:<26}{wm:>9.4f} ±{wc:<7.4f}{sm:>11.4f} ±{sc:<7.4f}{ws:>9.4f}")
        for r in rows:
            out.append(dict(config=name, n_feat=len(idx), **r))

    # 과제 4종 분리 (05 §② — diadochokinesis 는 성격이 다르다)
    print(f"\n{'과제별 (마이크 제외 5채널)':<26}{'창 AUC':>18}{'피험자 AUC':>20}")
    for t in sorted(set(task)):
        m = task == t
        if len(np.unique(sid[m])) < a.folds * 2:
            continue
        rows = run(logBP[m][:, [1, 2, 3, 4, 5]], y[m], sid[m], a.folds)
        wm, wc, _ = summarize(rows, "win_auc")
        sm, sc, _ = summarize(rows, "subj_auc")
        print(f"{t:<26}{wm:>9.4f} ±{wc:<7.4f}{sm:>11.4f} ±{sc:<7.4f}")
        for r in rows:
            out.append(dict(config=f"task:{t}", n_feat=5, **r))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\n원본 저장: {a.out}  ({len(out)}행)")
    print("참조 모델 p16d128L6 윈도우 AUC .8629 · 피험자 정확도 .8790 (08 §3-2)")


if __name__ == "__main__":
    main()
