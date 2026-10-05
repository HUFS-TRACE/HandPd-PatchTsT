"""R0 · R4 · R5 를 d64L2 한 모델로 통일해 표 1 행을 만든다.

왜 필요한가 — 표 1 은 원래 ②단계 담당자가 다른 하이퍼파라미터로 돌린 값이었다.
⑤단계(설명 분석)가 쓰는 모델과 표 1 의 모델이 다르면 "우리가 설명한 그 모델이
교란을 통과했다"고 말할 수 없다. 세 조건을 모두 d64L2 · 3-fold · autocw 로
맞춰 다시 재고, 연령 규칙과 다수결을 같은 피험자 집합에서 계산한다.

fold 가 3개뿐이므로 fold 간 표준편차를 함께 낸다. 시드는 반복 측정이지
독립 표본이 아니다 (GroupKFold 가 shuffle 하지 않아 분할이 시드에 안 바뀐다).
"""
import sys
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
import csv, glob, io, os
import numpy as np

RES = "results"
META = "data/subject_meta.csv"


def folds(pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        with io.open(p, encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    return rows


def meta():
    with io.open(META, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def subset(path):
    if path is None:
        return None
    with io.open(path, encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip()}


def baseline(keep):
    """연령 규칙(>=52세)·다수결·연령차를 같은 피험자 집합에서 계산한다."""
    m = [r for r in meta()]
    # subset 파일은 npz 쪽 ID(H_/P_ 접두)를 쓴다 — 그 열로 맞춰야 걸린다
    # subset 파일은 npz 쪽 ID(H_/P_ 접두)를 쓴다 — 그 열로 맞춰야 걸린다
    idc = ("npz_subject_id" if "npz_subject_id" in m[0]
           else next(k for k in m[0] if "id" in k.lower()))
    agec = next(k for k in m[0] if "age" in k.lower())
    labc = next(k for k in m[0] if k.lower() in ("label", "y", "class", "group"))
    if keep is not None:
        m = [r for r in m if r[idc].strip() in keep]
    y = np.array([int(float(r[labc])) for r in m])
    age = np.array([float(r[agec]) for r in m])
    # 연령 규칙에 최선의 조건을 준다 — 그 부분집합에서 임계값을 다시 고른다.
    # 고정 임계(>=52)를 쓰면 부분집합에서 규칙이 불리해져 우리 모델이 과대평가된다.
    rule, thr = max(((float(((age >= t).astype(int) == y).mean()), t)
                     for t in range(35, 85)))
    major = float(max((y == 0).mean(), (y == 1).mean()))
    gap = float(age[y == 1].mean() - age[y == 0].mean())
    return len(m), gap, rule, thr, major


CONDS = [
    ("R0 전체",        "folds_size_d64L2_2s_fold3_autocw*.csv",    None),
    ("R4 2016년 제외", "folds_size_d64L2_2s_R4_fold3_autocw*.csv", "data/subset_R4.txt"),
    ("R5 45세 이상",   "folds_size_d64L2_2s_R5_fold3_autocw*.csv", "data/subset_R5.txt"),
]

print(f"{'조건':<15}{'인원':>5}{'연령차':>8}{'규칙':>7}{'임계':>6}{'다수결':>8}"
      f"{'모델(피험자)':>13}{'모델-규칙':>10}{'창AUC':>9}")
out = []
for name, pat, sfile in CONDS:
    fr = folds(pat)
    if not fr:
        print(f"{name:<15}  (결과 없음 — 아직 안 돌았다: {pat})")
        continue
    keep = subset(sfile)
    n, gap, rule, thr, major = baseline(keep)
    acc = np.array([float(r["subject_acc"]) for r in fr])
    auc = np.array([float(r["roc_auc"]) for r in fr])
    seeds = sorted({r["seed"] for r in fr})
    print(f"{name:<15}{n:>5}{gap:>+8.1f}{rule:>7.3f}{thr:>5}세{major:>8.3f}"
          f"{acc.mean():>13.3f}{acc.mean()-rule:>+10.3f}{auc.mean():>9.4f}"
          f"   fold {len(fr)} · 시드 {','.join(seeds)}")
    out.append(dict(cond=name, n=n, age_gap=round(gap, 1), rule=round(rule, 3), rule_threshold=thr,
                    majority=round(major, 3), model_subj=round(float(acc.mean()), 3),
                    model_subj_sd=round(float(acc.std(ddof=1)), 3),
                    advantage=round(float(acc.mean() - rule), 3),
                    win_auc=round(float(auc.mean()), 4),
                    win_auc_sd=round(float(auc.std(ddof=1)), 4),
                    n_folds=len(fr), seeds=",".join(seeds)))

if out:
    p = os.path.join(RES, "table1_d64L2.csv")
    with io.open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\n-> {p}")
    print("\n  '모델-규칙'이 R5 에서 가장 커야 연령 대리 학습이 아니라는 논증이 선다.")
    print("  fold 3개 × 시드 3개 = 9개 값이지만 분할은 시드에 안 바뀌므로")
    print("  독립 표본 9개가 아니라 '3 fold 반복 측정 3회'로 읽어야 한다.")
