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
]

# 패치 축 스윕 — 시퀀스 길이의 5%(작은 패치)부터 10%(큰 패치)까지.
#   패치 수가 곧 어텐션 비용(제곱)이라, 큰 패치 설정은 baseline보다 10~20배 싸다.
#   50% 겹침(stride = patch_len/2)으로 통일해 "패치 크기"만 변수로 남긴다.
PATCH_CONFIGS = [
    dict(name="p16",  patch_len=16,  stride=8,  d_model=64, n_heads=4, n_layers=2, d_ff=128),
    dict(name="p32",  patch_len=32,  stride=16, d_model=64, n_heads=4, n_layers=2, d_ff=128),
    dict(name="p50",  patch_len=50,  stride=25, d_model=64, n_heads=4, n_layers=2, d_ff=128),
    dict(name="p100", patch_len=100, stride=50, d_model=64, n_heads=4, n_layers=2, d_ff=128),
    dict(name="p200", patch_len=200, stride=100, d_model=64, n_heads=4, n_layers=2, d_ff=128),
]

# 어텐션 head 축 — d_model 64를 몇 갈래로 쪼갤지만 변화(파라미터 수는 동일).
#   head가 많을수록 head당 차원이 좁아진다: 2→32차원, 4→16, 8→8.
HEAD_CONFIGS = [
    dict(name="h2", patch_len=100, stride=50, d_model=64, n_heads=2, n_layers=2, d_ff=128),
    dict(name="h4", patch_len=100, stride=50, d_model=64, n_heads=4, n_layers=2, d_ff=128),
    dict(name="h8", patch_len=100, stride=50, d_model=64, n_heads=8, n_layers=2, d_ff=128),
]

# 모델 크기 축 — 패치는 100/50으로 고정하고 폭·깊이만 변화.
SIZE_CONFIGS = [
    dict(name="d32L2",  patch_len=100, stride=50, d_model=32,  n_heads=4, n_layers=2, d_ff=64),
    dict(name="d64L1",  patch_len=100, stride=50, d_model=64,  n_heads=4, n_layers=1, d_ff=128),
    dict(name="d64L2",  patch_len=100, stride=50, d_model=64,  n_heads=4, n_layers=2, d_ff=128),
    dict(name="d64L4",  patch_len=100, stride=50, d_model=64,  n_heads=4, n_layers=4, d_ff=128),
    dict(name="d128L2", patch_len=100, stride=50, d_model=128, n_heads=8, n_layers=2, d_ff=256),
    dict(name="d128L3", patch_len=100, stride=50, d_model=128, n_heads=8, n_layers=3, d_ff=256),
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

    p.add_argument("--verbose", "-v", action="store_true",
                   help="epoch마다 train/val loss·acc 출력")
    p.add_argument("--dry-run",   action="store_true", help="학습 없이 FLOPs 표만 출력")
    p.add_argument("--plot-only", action="store_true", help="기존 CSV로 곡선만 다시 그리기")
    p.add_argument("--ckpt",      default=None, help="학습된 체크포인트 1개만 평가")
    p.add_argument("--configs",   default=None,
                   help="쉼표로 구분한 설정 이름 (기본: 전체). 예: tiny,baseline")
    p.add_argument("--suite", default="default",
                   choices=["default", "patch", "size", "head"],
                   help="설정 묶음: default(원래 6종) / patch(패치 크기) / "
                        "size(모델 크기) / head(어텐션 head 수)")
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
    """윈도우 예측(0/1) → 피험자 단위 hard voting 정확도.

    윈도우 수(수천)와 달리 실제 사람은 수십 명뿐이라, 사람 단위 집계가
    일반화 성능의 정직한 지표다.

    hard voting: 각 윈도우가 한 표씩 행사. 확신도 0.51이든 0.99든 같은 1표.
    """
    correct = 0
    uniq = np.unique(subj)
    for s in uniq:
        m = subj == s
        correct += int(round(pred[m].mean()) == ytrue[m][0])
    return correct / len(uniq)


def subject_vote_soft(probs, subj, ytrue, threshold=0.5):
    """윈도우 확률 → 피험자 단위 soft voting 정확도.

    soft voting: 각 윈도우의 P(환자)를 평균낸 뒤 임계값과 비교.
    확신도가 표의 무게에 반영되므로, 애매한 윈도우(0.51)가 확실한
    윈도우(0.99)를 상쇄하지 못한다.

    hard와 갈리는 경우: 윈도우 10개 중 6개가 P=0.55(환자), 4개가 P=0.05(정상)
      hard → 6:4로 환자 판정
      soft → 평균 0.35로 정상 판정
    어느 쪽이 옳은지는 데이터가 정한다(→ compare_voting).
    """
    correct = 0
    uniq = np.unique(subj)
    for s in uniq:
        m = subj == s
        correct += int((probs[m].mean() >= threshold) == bool(ytrue[m][0]))
    return correct / len(uniq)


def subject_vote_both(pred, probs, subj, ytrue):
    """(hard, soft) 정확도를 한 번에. 두 방식의 비교용."""
    return (subject_vote(pred, subj, ytrue),
            subject_vote_soft(probs, subj, ytrue))


def train_eval_config(cfg, X, y, subject_id, folds, args):
    """한 설정을 n-fold 학습·평가 → (윈도우acc, 피험자acc, AUC, FLOPs, 파라미터)."""
    device = torch.device(args.device)
    ckpt = os.path.join(RESULT_DIR, f"_tmp_{cfg['name']}.pt")
    win_accs, subj_accs, subj_accs_soft, aucs, gaps = [], [], [], [], []
    curves = []                                    # fold별 학습 곡선 (loss 그래프용)
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

        best_val, no_improve, best_epoch = float("inf"), 0, 0
        hist = []                                  # (epoch, train_loss, train_acc, val_loss, val_acc)
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc, *_ = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc, *_ = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            hist.append((epoch, tr_loss, tr_acc, val_loss, val_acc))

            if val_loss < best_val:
                best_val, no_improve, best_epoch = val_loss, 0, epoch
                torch.save(model.state_dict(), ckpt)
                mark = " *"                        # 체크포인트 갱신
            else:
                no_improve += 1
                mark = ""

            if args.verbose:
                print(f"      ep {epoch:3d} | train {tr_loss:.4f} / {tr_acc:.3f}"
                      f" | val {val_loss:.4f} / {val_acc:.3f}"
                      f" | lr {optimizer.param_groups[0]['lr']:.2e}{mark}", flush=True)

            if no_improve >= args.patience:
                if args.verbose:
                    print(f"      early stop @ ep {epoch} (best ep {best_epoch})", flush=True)
                break

        curves.append(hist)
        model.load_state_dict(torch.load(ckpt))
        _, acc, preds, labels, probs = run_epoch(model, test_loader, criterion, device)
        win_accs.append(acc)
        sa_hard, sa_soft = subject_vote_both(preds, probs, subject_id[te], labels)
        subj_accs.append(sa_hard)
        subj_accs_soft.append(sa_soft)
        aucs.append(roc_auc_score(labels, probs))
        # 과적합 진단: 마지막 epoch의 train/val loss 간극
        gaps.append(hist[-1][3] - hist[-1][1])
        print(f"    fold {fold+1}/{len(folds)}  윈도우 {acc:.3f}  "
              f"피험자 hard {sa_hard:.3f} / soft {sa_soft:.3f}  "
              f"AUC {aucs[-1]:.3f}  ep {len(hist)}(best {best_epoch})  "
              f"gap {gaps[-1]:+.3f}", flush=True)

    if os.path.exists(ckpt):
        os.remove(ckpt)
    return dict(
        window_acc=float(np.mean(win_accs)), window_std=float(np.std(win_accs)),
        subject_acc=float(np.mean(subj_accs)),
        subject_acc_soft=float(np.mean(subj_accs_soft)),
        roc_auc=float(np.mean(aucs)),
        mean_epochs=float(np.mean([len(h) for h in curves])),
        overfit_gap=float(np.mean(gaps)),
        mflops=flops, params=params, curves=curves)


def plot_curves(curves_by_cfg, out_png):
    """설정별 train/val loss 곡선. fold는 연하게, 평균은 진하게.

    val loss가 오르는데 train loss만 내려가면 과적합 — 그 지점이 epoch 상한의
    근거가 된다. 그래서 fold별 원본을 남긴다(평균만 보면 조기중단 시점이 흐려짐).
    """
    n = len(curves_by_cfg)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)

    for ax, (name, curves) in zip(axes.flat, curves_by_cfg.items()):
        for h in curves:                            # fold별 (연하게)
            ep = [r[0] for r in h]
            ax.plot(ep, [r[1] for r in h], color="#1e5aa8", alpha=0.25, lw=1)
            ax.plot(ep, [r[3] for r in h], color="#c0392b", alpha=0.25, lw=1)

        # 평균 (fold마다 조기중단 시점이 달라 짧은 쪽에 맞춤)
        L = min(len(h) for h in curves)
        ep = list(range(1, L + 1))
        tr = [np.mean([h[i][1] for h in curves]) for i in range(L)]
        va = [np.mean([h[i][3] for h in curves]) for i in range(L)]
        ax.plot(ep, tr, color="#1e5aa8", lw=2.4, label="train loss")
        ax.plot(ep, va, color="#c0392b", lw=2.4, label="val loss")

        best = int(np.argmin(va)) + 1
        ax.axvline(best, ls="--", color="#666", lw=1.2, alpha=0.8)
        ax.annotate(f"val 최저 ep{best}", (best, max(va)), fontsize=8, color="#444",
                    textcoords="offset points", xytext=(4, -4))

        ax.set_title(f"{name}  (fold {len(curves)}개)", fontsize=11)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, facecolor="white")
    print(f"학습 곡선 저장: {out_png}")


def plot_voting(rows, out_png):
    """hard vs soft voting 비교 — 설정별 막대 + 차이."""
    names = [r["name"] for r in rows]
    hard = np.array([r["subject_acc"] for r in rows]) * 100
    soft = np.array([r["subject_acc_soft"] for r in rows]) * 100
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   gridspec_kw={"width_ratios": [2, 1]})
    ax1.bar(x - 0.2, hard, 0.4, label="hard voting", color="#1e5aa8")
    ax1.bar(x + 0.2, soft, 0.4, label="soft voting", color="#c0392b")
    for i, (h, s) in enumerate(zip(hard, soft)):
        ax1.text(i - 0.2, h + 0.3, f"{h:.1f}", ha="center", fontsize=8)
        ax1.text(i + 0.2, s + 0.3, f"{s:.1f}", ha="center", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("피험자 정확도 (%)")
    ax1.set_ylim(min(hard.min(), soft.min()) - 4, max(hard.max(), soft.max()) + 3)
    ax1.set_title("hard vs soft voting"); ax1.legend(); ax1.grid(alpha=0.3, axis="y")

    diff = soft - hard
    colors = ["#1a7a3a" if d > 0 else "#c0392b" if d < 0 else "#999" for d in diff]
    ax2.barh(x, diff, color=colors)
    ax2.axvline(0, color="#333", lw=1)
    ax2.set_yticks(x); ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("soft − hard (%p)")
    ax2.set_title("차이 (양수 = soft 우세)"); ax2.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140, facecolor="white")
    print(f"voting 비교 저장: {out_png}")
    return diff


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
            "mflops", "params", "window_acc", "window_std",
            "subject_acc", "subject_acc_soft", "roc_auc", "mean_epochs", "overfit_gap"]
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
    # 스윕·데이터별로 파일을 나눠 이전 결과를 덮어쓰지 않는다
    tag = args.suite if args.suite != "default" else ""
    ds = os.path.basename(args.data_path).replace("windows", "").replace(".npz", "")
    seed_tag = "" if args.seed == 42 else f"s{args.seed}"
    lr_tag = "" if args.lr == 1e-3 else f"lr{args.lr:g}"
    bs_tag = "" if args.batch_size == 64 else f"bs{args.batch_size}"
    fd_tag = "" if args.n_folds == 5 else f"fold{args.n_folds}"
    pt_tag = "" if args.patience == 10 else f"pat{args.patience}"
    sfx = "_".join(s for s in (tag, ds.strip("_"), lr_tag, bs_tag,
                               fd_tag, pt_tag, seed_tag) if s)
    sfx = f"_{sfx}" if sfx else ""
    csv_path = os.path.join(RESULT_DIR, f"pareto{sfx}.csv")
    png_path = os.path.join(RESULT_DIR, f"pareto{sfx}.png")

    if args.plot_only:
        plot_pareto(load_csv(csv_path), png_path)
        return

    torch.manual_seed(args.seed)
    X, y, subject_id, _ = load_npz(args.data_path)
    seq_len, n_ch, n_cls = X.shape[2], X.shape[1], int(y.max() + 1)
    print(f"데이터 {args.data_path}  X{X.shape}  "
          f"정상 {(y==0).sum()} / 환자 {(y==1).sum()}  "
          f"피험자 {len(set(subject_id))}명\n")

    configs = {"default": CONFIGS, "patch": PATCH_CONFIGS,
               "size": SIZE_CONFIGS, "head": HEAD_CONFIGS}[args.suite]
    if args.configs:
        want = [s.strip() for s in args.configs.split(",")]
        configs = [c for c in configs if c["name"] in want]
    # 시퀀스보다 긴 패치는 만들 수 없다 (1초 데이터에서 patch200 등)
    configs = [c for c in configs if (seq_len - c["patch_len"]) // c["stride"] + 1 >= 2]

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

    rows, curves_by_cfg = [], {}
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg['name']}  "
              f"({flop_map[cfg['name']][0]/1e6:.1f} MFLOPs)", flush=True)
        res = train_eval_config(cfg, X, y, subject_id, folds, args)
        curves_by_cfg[cfg["name"]] = res.pop("curves")
        res["mflops"] = round(res["mflops"] / 1e6, 2)
        rows.append({**cfg, **{k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in res.items()}})
        print(f"    → 윈도우 {res['window_acc']:.3f}±{res['window_std']:.3f}  "
              f"피험자 hard {res['subject_acc']:.3f} / soft {res['subject_acc_soft']:.3f}  "
              f"AUC {res['roc_auc']:.3f}  평균 {res['mean_epochs']:.1f}ep  "
              f"과적합gap {res['overfit_gap']:+.3f}\n")
        save_csv(rows, csv_path)        # 중간에 끊겨도 남도록 매번 저장

    # ── 결과 요약 ──
    mask = pareto_mask([r["mflops"] for r in rows], [r["subject_acc"] for r in rows])
    print(f"\n{'설정':<10}{'MFLOPs':>10}{'윈도우':>9}{'hard':>8}{'soft':>8}"
          f"{'AUC':>8}{'ep':>6}{'gap':>8}  Pareto")
    for r, m in zip(rows, mask):
        print(f"{r['name']:<10}{r['mflops']:>10.1f}{r['window_acc']:>9.3f}"
              f"{r['subject_acc']:>8.3f}{r['subject_acc_soft']:>8.3f}"
              f"{r['roc_auc']:>8.3f}{r['mean_epochs']:>6.1f}"
              f"{r['overfit_gap']:>+8.3f}  {'★' if m else ''}")

    plot_pareto(rows, png_path)
    plot_curves(curves_by_cfg, os.path.join(RESULT_DIR, f"loss_curves{sfx}.png"))
    diff = plot_voting(rows, os.path.join(RESULT_DIR, f"voting{sfx}.png"))

    # ── voting 결론 ──
    win_soft = int((diff > 0).sum()); win_hard = int((diff < 0).sum())
    print(f"\n[voting] soft 우세 {win_soft}개 / hard 우세 {win_hard}개 / "
          f"동률 {len(diff)-win_soft-win_hard}개  |  평균 차이 {diff.mean():+.2f}%p")
    n_subj_per_fold = len(set(subject_id)) / args.n_folds
    print(f"         피험자 {len(set(subject_id))}명 기준 1명 ≈ "
          f"{100/len(set(subject_id))*args.n_folds/args.n_folds:.1f}%p "
          f"(fold당 test {n_subj_per_fold:.0f}명) → "
          f"{abs(diff).max():.1f}%p 이하 차이는 노이즈로 봐야 함")

    best = max((r for r, m in zip(rows, mask) if m), key=lambda r: r["subject_acc"])
    cheap = min((r for r, m in zip(rows, mask) if m), key=lambda r: r["mflops"])
    print(f"\n정확도 최고(Pareto): {best['name']}  "
          f"{best['subject_acc']*100:.1f}% @ {best['mflops']:.1f} MFLOPs")
    print(f"최소 연산(Pareto):   {cheap['name']}  "
          f"{cheap['subject_acc']*100:.1f}% @ {cheap['mflops']:.1f} MFLOPs")


if __name__ == "__main__":
    main()
