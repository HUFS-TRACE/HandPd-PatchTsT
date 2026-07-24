"""
FLOPs 측정 + 정확도-연산량 Pareto 곡선.

train.py의 학습 로직(run_epoch)과 dataset/model을 그대로 재사용하고,
설정(config)을 바꿔가며 [FLOPs, 정확도] 점을 모아 Pareto 프론티어를 그린다.

사용법
    python evaluate.py --dry-run                  # FLOPs·파라미터 표만 (학습 없음, 수초)
    python evaluate.py --data-path data/windows_2s.npz
    python evaluate.py --ckpt best_model.pt       # 이미 학습된 체크포인트 1개 평가
    python evaluate.py --plot-only                # CSV에서 곡선만 다시 그리기

산출물
    results/pareto.csv        설정별 FLOPs·파라미터·정확도 (수치 기록용)
    results/pareto.png        정확도-연산량 곡선 + Pareto 프론티어

FLOPs 주석
    torch.utils.flop_counter.FlopCounterMode(PyTorch 내장) 사용.
    관례대로 행렬곱 기준이며 MAC이 아닌 실제 FLOPs(곱셈+덧셈)를 센다.
    softmax·LayerNorm·활성함수는 제외 (전체의 1% 미만, 표준 관례).
    단위는 "샘플 1개당 순전파 FLOPs".
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from dataset import WindowDataset, load_npz, subject_kfold
from model import PatchTSTClassifier
from train import run_epoch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

RESULT_DIR = "results"

# 비교할 설정들. baseline = train.py 기본값.
# 이름 / 패치 / stride / d_model / heads / layers / d_ff
CONFIGS = [
    dict(name="tiny",     patch_len=32, stride=16, d_model=64,  n_heads=4, n_layers=1, d_ff=128),
    dict(name="small",    patch_len=32, stride=16, d_model=64,  n_heads=4, n_layers=2, d_ff=128),
    dict(name="coarse",   patch_len=64, stride=32, d_model=128, n_heads=8, n_layers=3, d_ff=256),
    dict(name="mid",      patch_len=16, stride=8,  d_model=64,  n_heads=4, n_layers=2, d_ff=128),
    dict(name="baseline", patch_len=16, stride=8,  d_model=128, n_heads=8, n_layers=3, d_ff=256),
    dict(name="wide",     patch_len=16, stride=8,  d_model=192, n_heads=8, n_layers=3, d_ff=384),
    # deep(16/8, d128, L6)은 기본 스윕에서 제외.
    #   earlyexit.py 기본 설정과 구조가 동일하고, EE를 thr=1.0으로 돌리면
    #   아무도 조기종료를 안 해 6층 전체를 통과 → 같은 지점이 공짜로 나온다.
    #   또 Early Exit 효과를 주장할 땐 별도 학습한 deep이 아니라
    #   같은 모델의 thr 1.0(조기종료 OFF)이 올바른 대조군이다.
    #   단, deep은 "최종 헤드 손실만"으로 학습하므로 EE의 공동 학습과 완전히
    #   같지는 않다. 공동 학습의 정확도 손해를 재고 싶으면 아래로 실행:
    #       python evaluate.py --configs deep
    # dict(name="deep",   patch_len=16, stride=8,  d_model=128, n_heads=8, n_layers=6, d_ff=256),
]


def parse_args():
    p = argparse.ArgumentParser(description="FLOPs 측정 + Pareto 곡선")
    p.add_argument("--data-path",    default="data/windows.npz")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--n-folds",      type=int,   default=5)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--head-dropout", type=float, default=0.2)
    p.add_argument("--class-weight-healthy", type=float, default=2.0)
    p.add_argument("--class-weight-patient", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--dry-run",   action="store_true", help="학습 없이 FLOPs 표만 출력")
    p.add_argument("--plot-only", action="store_true", help="기존 CSV로 곡선만 다시 그리기")
    p.add_argument("--ckpt",      default=None, help="학습된 체크포인트 1개만 평가")
    p.add_argument("--configs",   default=None,
                   help="쉼표로 구분한 설정 이름 (기본: 전체). 예: tiny,baseline")
    return p.parse_args()


# ─────────────────────────── FLOPs ───────────────────────────

def build_model(cfg, seq_len, num_channels, num_classes, args):
    return PatchTSTClassifier(
        seq_len=seq_len, num_channels=num_channels, num_classes=num_classes,
        patch_len=cfg["patch_len"], stride=cfg["stride"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
        dropout=args.dropout, head_dropout=args.head_dropout,
    )


def measure_flops(model, seq_len, num_channels, max_layer=None):
    """샘플 1개당 순전파 FLOPs.

    max_layer=k를 주면 k층까지만 계산한 FLOPs (Early Exit 출구별 누적 연산량).
    해당 모델이 forward(x, max_layer=...)를 지원할 때만 유효하다.

    주의: eval() 상태로 재면 nn.TransformerEncoder가 fused fast-path로 빠져
    카운터가 인코더 연산을 통째로 놓친다(층 수를 바꿔도 값이 안 변함).
    train() 상태에서 재야 일반 경로를 타서 정확히 집계된다.
    dropout은 FLOPs에 영향이 없으므로 train 모드로 재도 값은 동일하다.
    """
    was_training = model.training
    device = next(model.parameters()).device
    model.cpu().train()
    x = torch.randn(1, num_channels, seq_len)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(x) if max_layer is None else model(x, max_layer=max_layer)
    flops = counter.get_total_flops()
    model.to(device).train(was_training)
    return flops


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ────────────────────────── 학습·평가 ──────────────────────────

def subject_vote(pred, subj, ytrue):
    """윈도우 예측 → 피험자 단위 다수결 정확도.

    윈도우 수(수천)와 달리 실제 사람은 수십 명뿐이라, 사람 단위 집계가
    일반화 성능의 정직한 지표다.
    """
    correct = 0
    uniq = np.unique(subj)
    for s in uniq:
        m = subj == s
        correct += int(round(pred[m].mean()) == ytrue[m][0])
    return correct / len(uniq)


def train_eval_config(cfg, X, y, subject_id, folds, args):
    """한 설정을 n-fold 학습·평가 → (윈도우acc, 피험자acc, AUC, FLOPs, 파라미터)."""
    device = torch.device(args.device)
    ckpt = os.path.join(RESULT_DIR, f"_tmp_{cfg['name']}.pt")
    win_accs, subj_accs, aucs = [], [], []
    flops = params = None

    for fold, (tr, va, te) in enumerate(folds):
        train_loader = DataLoader(WindowDataset(X[tr], y[tr]),
                                  batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(WindowDataset(X[va], y[va]), batch_size=args.batch_size)
        test_loader  = DataLoader(WindowDataset(X[te], y[te]), batch_size=args.batch_size)

        model = build_model(cfg, X.shape[2], X.shape[1], int(y.max() + 1), args).to(device)
        if flops is None:                       # 설정당 1회만 측정 (fold와 무관)
            flops, params = measure_flops(model, X.shape[2], X.shape[1]), count_params(model)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [args.class_weight_healthy, args.class_weight_patient],
            dtype=torch.float32).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3)

        best_val, no_improve = float("inf"), 0
        for epoch in range(1, args.epochs + 1):
            run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, *_ = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val, no_improve = val_loss, 0
                torch.save(model.state_dict(), ckpt)
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    break

        model.load_state_dict(torch.load(ckpt))
        _, acc, preds, labels, probs = run_epoch(model, test_loader, criterion, device)
        win_accs.append(acc)
        subj_accs.append(subject_vote(preds, subject_id[te], labels))
        aucs.append(roc_auc_score(labels, probs))
        print(f"    fold {fold+1}/{len(folds)}  윈도우 {acc:.3f}  "
              f"피험자 {subj_accs[-1]:.3f}  AUC {aucs[-1]:.3f}", flush=True)

    if os.path.exists(ckpt):
        os.remove(ckpt)
    return (float(np.mean(win_accs)), float(np.std(win_accs)),
            float(np.mean(subj_accs)), float(np.mean(aucs)), flops, params)


# ─────────────────────────── Pareto ───────────────────────────

def pareto_mask(flops, acc):
    """더 적은 연산으로 더 높은 정확도를 내는 설정이 없으면 Pareto 최적."""
    flops, acc = np.asarray(flops), np.asarray(acc)
    mask = np.ones(len(flops), dtype=bool)
    for i in range(len(flops)):
        dominated = ((flops <= flops[i]) & (acc >= acc[i]) &
                     ((flops < flops[i]) | (acc > acc[i])))
        mask[i] = not dominated.any()
    return mask


def plot_pareto(rows, out_png, acc_key="subject_acc", acc_label="피험자 정확도 (%)"):
    names = [r["name"] for r in rows]
    mf    = np.array([r["mflops"] for r in rows])
    acc   = np.array([r[acc_key] for r in rows]) * 100
    mask  = pareto_mask(mf, acc)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(mf[~mask], acc[~mask], s=70, color="#999", zorder=3, label="지배됨")
    ax.scatter(mf[mask],  acc[mask],  s=110, color="#c0392b", zorder=4,
               edgecolor="white", linewidth=1.5, label="Pareto 최적")

    order = np.argsort(mf[mask])
    ax.plot(mf[mask][order], acc[mask][order], "--", color="#c0392b",
            lw=1.8, alpha=0.8, zorder=2, label="Pareto 프론티어")

    for n, f, a in zip(names, mf, acc):
        ax.annotate(n, (f, a), fontsize=9, textcoords="offset points", xytext=(6, 6))

    ax.set_xscale("log")
    ax.set_xlabel("샘플당 연산량 (MFLOPs, 로그 스케일)")
    ax.set_ylabel(acc_label)
    ax.set_title("PatchTST 정확도-연산량 Pareto 곡선")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="white")
    print(f"곡선 저장: {out_png}")


def save_csv(rows, path):
    cols = ["name", "patch_len", "stride", "d_model", "n_heads", "n_layers", "d_ff",
            "mflops", "params", "window_acc", "window_std", "subject_acc", "roc_auc"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"수치 저장: {path}")


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        cols = f.readline().strip().split(",")
        for line in f:
            v = line.strip().split(",")
            if len(v) != len(cols):
                continue
            r = dict(zip(cols, v))
            for k, val in r.items():          # 숫자로 읽히는 건 전부 float
                if k == "name":               # (pareto.csv와 earlyexit.csv는 컬럼이 다름)
                    continue
                try:
                    r[k] = float(val)
                except ValueError:
                    pass
            rows.append(r)
    return rows


# ──────────────────────────── main ────────────────────────────

def main():
    args = parse_args()
    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, "pareto.csv")
    png_path = os.path.join(RESULT_DIR, "pareto.png")

    if args.plot_only:
        plot_pareto(load_csv(csv_path), png_path)
        return

    torch.manual_seed(args.seed)
    X, y, subject_id, _ = load_npz(args.data_path)
    seq_len, n_ch, n_cls = X.shape[2], X.shape[1], int(y.max() + 1)
    print(f"데이터 {args.data_path}  X{X.shape}  "
          f"정상 {(y==0).sum()} / 환자 {(y==1).sum()}  "
          f"피험자 {len(set(subject_id))}명\n")

    configs = CONFIGS
    if args.configs:
        want = [s.strip() for s in args.configs.split(",")]
        configs = [c for c in CONFIGS if c["name"] in want]

    # ── 체크포인트 1개만 평가 ──
    if args.ckpt:
        cfg = next(c for c in configs if c["name"] == "baseline")
        model = build_model(cfg, seq_len, n_ch, n_cls, args)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        print(f"{args.ckpt}  FLOPs {measure_flops(model, seq_len, n_ch)/1e6:.1f}M  "
              f"파라미터 {count_params(model):,}")
        return

    # ── FLOPs 표 (학습 없이 즉시) ──
    print(f"{'설정':<10}{'패치/stride':>13}{'d_model':>9}{'층':>4}"
          f"{'MFLOPs':>10}{'파라미터':>11}")
    flop_map = {}
    for cfg in configs:
        m = build_model(cfg, seq_len, n_ch, n_cls, args)
        fl, pa = measure_flops(m, seq_len, n_ch), count_params(m)
        flop_map[cfg["name"]] = (fl, pa)
        print(f"{cfg['name']:<10}{cfg['patch_len']:>7}/{cfg['stride']:<5}"
              f"{cfg['d_model']:>9}{cfg['n_layers']:>4}"
              f"{fl/1e6:>10.1f}{pa:>11,}")

    if args.dry_run:
        print("\n--dry-run: 학습은 건너뜀. 곡선을 그리려면 --dry-run 없이 실행하세요.")
        return

    # ── 설정별 학습·평가 ──
    folds = subject_kfold(subject_id, y, n_splits=args.n_folds, seed=args.seed)
    print(f"\n{len(configs)}개 설정 × {args.n_folds}-fold 학습 시작 "
          f"(device={args.device}). 오래 걸립니다.\n")

    rows = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg['name']}  "
              f"({flop_map[cfg['name']][0]/1e6:.1f} MFLOPs)", flush=True)
        wacc, wstd, sacc, auc, fl, pa = train_eval_config(
            cfg, X, y, subject_id, folds, args)
        rows.append({**cfg, "mflops": round(fl / 1e6, 2), "params": pa,
                     "window_acc": round(wacc, 4), "window_std": round(wstd, 4),
                     "subject_acc": round(sacc, 4), "roc_auc": round(auc, 4)})
        print(f"    → 윈도우 {wacc:.3f}±{wstd:.3f}  피험자 {sacc:.3f}  AUC {auc:.3f}\n")
        save_csv(rows, csv_path)        # 중간에 끊겨도 남도록 매번 저장

    # ── 결과 요약 + 곡선 ──
    mask = pareto_mask([r["mflops"] for r in rows], [r["subject_acc"] for r in rows])
    print(f"\n{'설정':<10}{'MFLOPs':>10}{'윈도우acc':>11}{'피험자acc':>11}{'AUC':>8}  Pareto")
    for r, m in zip(rows, mask):
        print(f"{r['name']:<10}{r['mflops']:>10.1f}{r['window_acc']:>11.3f}"
              f"{r['subject_acc']:>11.3f}{r['roc_auc']:>8.3f}  {'★' if m else ''}")

    plot_pareto(rows, png_path)

    best = max((r for r, m in zip(rows, mask) if m), key=lambda r: r["subject_acc"])
    cheap = min((r for r, m in zip(rows, mask) if m), key=lambda r: r["mflops"])
    print(f"\n정확도 최고(Pareto): {best['name']}  "
          f"{best['subject_acc']*100:.1f}% @ {best['mflops']:.1f} MFLOPs")
    print(f"최소 연산(Pareto):   {cheap['name']}  "
          f"{cheap['subject_acc']*100:.1f}% @ {cheap['mflops']:.1f} MFLOPs")


if __name__ == "__main__":
    main()
