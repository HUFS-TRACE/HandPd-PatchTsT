"""
저장된 체크포인트로 성능을 다시 계산한다.

왜 필요한가
    학습 로그는 사라질 수 있고(Colab VM 회수처럼), 로그의 숫자를 논문에
    옮겨 적으면 출처가 로그 파일 하나에 묶인다. 체크포인트에는 그 fold의
    test 피험자 명단이 함께 저장되어 있으므로, 모델만 있으면 성능을 언제든
    다시 만들 수 있다.

    XAI(occlusion)와 같은 모델·같은 fold 분할을 쓰므로, 여기서 나온 성능은
    설명 결과와 짝이 맞는다.

지표
    윈도우 정확도 / ROC-AUC  — 주 지표
    피험자 정확도 (hard·soft) — 보조. 1명이 1.64%p라 노이즈가 크다

사용법
    python eval_ckpts.py --configs d64L2,p16d128L1 --seed 42
"""
import sys
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
import argparse
import csv
import glob
import io
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dataset import load_npz
from model import PatchTSTClassifier

RESULT_DIR = "results"


def build(ck, n_cls):
    cfg = ck["config"]
    return PatchTSTClassifier(
        seq_len=ck["seq_len"], num_channels=ck["num_channels"],
        num_classes=n_cls, patch_len=cfg["patch_len"], stride=cfg["stride"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
        dropout=ck.get("dropout", 0.2), head_dropout=ck.get("head_dropout", 0.2))


@torch.no_grad()
def predict(model, X, device, bs=64):
    out = []
    model.eval()
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).float().to(device)
        out.append(model(xb).softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description="체크포인트로 성능 재계산")
    ap.add_argument("--configs", default="d64L2,p16d128L1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-path", default="data/windows_2s.npz")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--csv", default="results/ckpt_performance.csv")
    a = ap.parse_args()

    X, y, subject_id, task = load_npz(a.data_path)
    n_cls = int(y.max() + 1)
    device = torch.device(a.device)
    rows = []

    for cfg_name in a.configs.split(","):
        cfg_name = cfg_name.strip()
        paths = sorted(glob.glob(
            os.path.join(RESULT_DIR, f"ckpt_{cfg_name}_f*_s{a.seed}*.pt")))
        if not paths:
            print(f"[{cfg_name}] 체크포인트 없음 - 건너뜀")
            continue

        prob = np.full(len(y), np.nan)
        fold_of = np.full(len(y), -1, np.int8)
        for p in paths:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            te = np.where(np.isin(subject_id, ck["test_subjects"]))[0]
            assert set(subject_id[te]) <= set(ck["test_subjects"])
            m = build(ck, n_cls)
            m.load_state_dict(ck["state_dict"])
            m.to(device)
            prob[te] = predict(m, X[te], device)
            fold_of[te] = ck["fold"]
            del m

        assert not np.isnan(prob).any(), "덮이지 않은 윈도우가 있다"
        pred = (prob >= 0.5).astype(int)
        win_acc = float((pred == y).mean())
        auc = float(roc_auc_score(y, prob))

        # 피험자 단위 — hard는 윈도우 다수결, soft는 확률 평균
        subs = np.unique(subject_id)
        hard = np.array([int(pred[subject_id == s].mean() >= 0.5) for s in subs])
        soft = np.array([int(prob[subject_id == s].mean() >= 0.5) for s in subs])
        truth = np.array([int(y[subject_id == s][0]) for s in subs])
        sa_hard, sa_soft = float((hard == truth).mean()), float((soft == truth).mean())

        # fold별 윈도우 정확도 — 편차를 보기 위해
        per_fold = [float((pred[fold_of == k] == y[fold_of == k]).mean())
                    for k in sorted(set(fold_of.tolist()))]

        cfgd = torch.load(paths[0], map_location="cpu", weights_only=False)["config"]
        rows.append(dict(
            config=cfg_name, n_folds=len(paths),
            patch=f"{cfgd['patch_len']}/{cfgd['stride']}",
            d_model=cfgd["d_model"], n_layers=cfgd["n_layers"],
            window_acc=round(win_acc, 4), roc_auc=round(auc, 4),
            subject_acc_hard=round(sa_hard, 4), subject_acc_soft=round(sa_soft, 4),
            fold_acc_min=round(min(per_fold), 4), fold_acc_max=round(max(per_fold), 4),
            fold_acc_std=round(float(np.std(per_fold)), 4)))
        print(f"[{cfg_name}] 패치 {cfgd['patch_len']}/{cfgd['stride']} "
              f"d{cfgd['d_model']} L{cfgd['n_layers']}")
        print(f"   윈도우 정확도 {win_acc:.4f}   ROC-AUC {auc:.4f}")
        print(f"   피험자 hard {sa_hard:.4f} / soft {sa_soft:.4f}  "
              f"({int((hard==truth).sum())}/{len(subs)}명)")
        print(f"   fold별 윈도우 정확도 {min(per_fold):.3f}~{max(per_fold):.3f} "
              f"(표준편차 {np.std(per_fold):.3f})\n")

    if rows and a.csv:
        with io.open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"-> {a.csv}")
    print("기준선: 피험자 다수결 57.4% / 윈도우 다수결 63.9% / 연령 규칙 73.8%")


if __name__ == "__main__":
    main()
