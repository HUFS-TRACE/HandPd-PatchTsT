"""
Early Exit PatchTST — 층마다 분류 출구(exit)를 달아 쉬운 샘플은 얕게 끝낸다.

핵심 아이디어
    고정깊이 모델은 쉬운 샘플도 항상 전 층을 통과해 연산을 낭비한다.
    각 인코더 층 뒤에 분류 헤드를 붙이고, 확신도(max softmax)가 임계값을
    넘으면 그 층에서 종료한다 → 평균 연산량 감소.

model.py의 PatchTSTClassifier와 백본 구성(RevIN·sinusoidal PE·pre-LN·GELU·
channel-independent 패칭)을 동일하게 맞춰, 오직 "출구 유무"만 다르게 했다.
따라서 두 모델의 비교는 Early Exit 효과만 분리해서 보여준다.

학습:  모든 출구의 손실을 합산해 공동 학습(joint training).
       이래야 얕은 층도 단독으로 판정할 수 있게 된다.
추론:  확신도 >= 임계값인 첫 출구에서 종료. 마지막 층은 무조건 종료.

실행:  python earlyexit.py --data-path data/windows_2s.npz
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import WindowDataset, load_npz, subject_kfold
from model import PositionalEncoding, RevIN


class EarlyExitPatchTST(nn.Module):
    """PatchTSTClassifier와 동일한 백본 + 층마다 분류 출구."""

    def __init__(
        self,
        seq_len,
        num_channels,
        num_classes,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=256,
        dropout=0.2,
        head_dropout=0.2,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.patch_len    = patch_len
        self.stride       = stride
        self.n_layers     = n_layers
        self.num_patches  = (seq_len - patch_len) // stride + 1

        if self.num_patches < 1:
            raise ValueError(
                f"seq_len({seq_len})이 너무 짧아요. "
                f"patch_len({patch_len})과 stride({stride})를 확인하세요."
            )

        self.revin       = RevIN(num_channels)
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_enc     = PositionalEncoding(d_model, max_len=self.num_patches)
        self.dropout     = nn.Dropout(dropout)

        # 층을 리스트로 풀어야 중간에서 꺼낼 수 있다 (baseline은 TransformerEncoder 통짜)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])

        # 층마다 출구. 구조는 baseline의 head와 동일.
        self.exits = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Dropout(head_dropout),
                nn.Linear(num_channels * d_model, num_classes),
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, max_layer=None):
        """max_layer=k면 k번째 층까지만 계산하고 그 출구의 logits만 반환.

        반환값
            max_layer=None : 모든 출구의 logits 리스트 [n_layers][B, C]
            max_layer=k    : k번째 출구의 logits [B, C]  (FLOPs 측정·조기종료용)
        """
        b, c, _ = x.shape
        k = self.n_layers if max_layer is None else max_layer

        x = self.revin(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(b * c, self.num_patches, self.patch_len)
        x = self.patch_embed(x)
        x = self.pos_enc(x)
        x = self.dropout(x)

        outs = []
        for i in range(k):
            x = self.layers[i](x)
            if max_layer is None or i == k - 1:
                z = x.mean(dim=1).reshape(b, c, -1)   # 패치 평균 → (B, C, d_model)
                outs.append(self.exits[i](z))
        return outs if max_layer is None else outs[-1]


# ───────────────────── 조기종료 시뮬레이션 ─────────────────────

def exit_probs(model, loader, device):
    """test 전체에 대해 각 출구의 softmax 확률 → [n_exit, N, n_cls]."""
    model.eval()
    per_exit = [[] for _ in range(model.n_layers)]
    with torch.no_grad():
        for X, _ in loader:
            for e, logits in enumerate(model(X.to(device))):
                per_exit[e].append(F.softmax(logits, dim=1).cpu().numpy())
    return np.stack([np.concatenate(p) for p in per_exit])


def simulate_exit(probs, threshold):
    """확신도 >= threshold인 첫 출구에서 나갔다고 가정.

    모든 출구를 미리 계산해두고 재계산만 하므로, 임계값을 아무리 촘촘히
    스윕해도 학습·추론 비용이 늘지 않는다 (표준 offline 평가 방식).

    반환: (예측, 사용한 출구 인덱스 0-based, 환자 확률)
    """
    n_exit, n = probs.shape[:2]
    conf = probs.max(axis=2)                       # [n_exit, N]
    used = np.full(n, n_exit - 1, dtype=int)
    for e in range(n_exit - 1):                    # 마지막 층은 무조건 종료
        hit = (conf[e] >= threshold) & (used == n_exit - 1)
        used[hit] = e
    idx = np.arange(n)
    chosen = probs[used, idx]                      # [N, n_cls]
    return chosen.argmax(axis=1), used, chosen[:, 1]


# 피험자 집계는 evaluate.py의 구현을 그대로 쓴다(하드/소프트 동일 기준 보장).
# 지연 import — evaluate가 이 모듈을 import하므로 최상단에 두면 순환 참조.
def subject_vote_both(pred, probs, subj, ytrue):
    from evaluate import subject_vote_both as _impl
    return _impl(pred, probs, subj, ytrue)


# ──────────────────────────── 학습 ────────────────────────────

def train_fold(model, train_loader, val_loader, criterion, args, ckpt):
    device = torch.device(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    best_val, no_improve = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            # 공동 손실: 모든 출구의 손실을 동일 가중으로 합산
            loss = sum(criterion(o, y) for o in model(X))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += sum(criterion(o, y) for o in model(X)).item() * X.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break
    model.load_state_dict(torch.load(ckpt))
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Early Exit PatchTST 학습·임계값 스윕")
    p.add_argument("--data-path",    default="data/windows_2s.npz")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--n-folds",      type=int,   default=5)
    p.add_argument("--patch-len",    type=int,   default=16)
    p.add_argument("--stride",       type=int,   default=8)
    p.add_argument("--d-model",      type=int,   default=128)
    p.add_argument("--n-heads",      type=int,   default=8)
    p.add_argument("--n-layers",     type=int,   default=6)
    p.add_argument("--d-ff",         type=int,   default=256)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--head-dropout", type=float, default=0.2)
    p.add_argument("--class-weight-healthy", type=float, default=2.0)
    p.add_argument("--class-weight-patient", type=float, default=1.0)
    p.add_argument("--out",  default="results/earlyexit.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    from evaluate import measure_flops, plot_pareto     # 순환 import 회피용 지연 로드

    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    X, y, subject_id, _ = load_npz(args.data_path)
    seq_len, n_ch, n_cls = X.shape[2], X.shape[1], int(y.max() + 1)
    print(f"데이터 {args.data_path}  X{X.shape}  "
          f"정상 {(y==0).sum()} / 환자 {(y==1).sum()}  "
          f"피험자 {len(set(subject_id))}명")

    def build():
        return EarlyExitPatchTST(
            seq_len=seq_len, num_channels=n_ch, num_classes=n_cls,
            patch_len=args.patch_len, stride=args.stride, d_model=args.d_model,
            n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff,
            dropout=args.dropout, head_dropout=args.head_dropout)

    # 출구별 누적 FLOPs — 조기종료의 실제 연산량 계산에 쓰인다
    probe = build()
    exit_flops = np.array([
        measure_flops(probe, seq_len, n_ch, max_layer=k + 1)
        for k in range(args.n_layers)
    ], dtype=float)
    print(f"\n출구별 누적 MFLOPs: "
          + "  ".join(f"exit{k+1}={f/1e6:.1f}" for k, f in enumerate(exit_flops)))

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(
        [args.class_weight_healthy, args.class_weight_patient],
        dtype=torch.float32).to(device))
    folds = subject_kfold(subject_id, y, n_splits=args.n_folds, seed=args.seed)
    thresholds = np.round(np.arange(0.50, 1.001, 0.025), 3)
    ckpt = "results/_tmp_ee.pt"

    # fold별 학습 1회 → 출구 확률 캐시 → 임계값은 재계산만
    cache = []
    for fold, (tr, va, te) in enumerate(folds, 1):
        print(f"\n[fold {fold}/{args.n_folds}] "
              f"train {len(tr)} / val {len(va)} / test {len(te)}", flush=True)
        model = train_fold(
            build().to(device),
            DataLoader(WindowDataset(X[tr], y[tr]), batch_size=args.batch_size, shuffle=True),
            DataLoader(WindowDataset(X[va], y[va]), batch_size=args.batch_size),
            criterion, args, ckpt)
        loader = DataLoader(WindowDataset(X[te], y[te]), batch_size=args.batch_size)
        cache.append((exit_probs(model, loader, device), y[te], subject_id[te]))
    if os.path.exists(ckpt):
        os.remove(ckpt)                                # 임시 체크포인트 정리

    rows = []
    for t in thresholds:
        wa, sa, ss, au, fl, layer = [], [], [], [], [], []
        for probs, yte, subj in cache:
            pred, used, p1 = simulate_exit(probs, t)
            wa.append((pred == yte).mean())
            h, s = subject_vote_both(pred, p1, subj, yte)   # p1 = 종료 출구의 P(환자)
            sa.append(h); ss.append(s)
            au.append(roc_auc_score(yte, p1))
            fl.append(exit_flops[used].mean())        # 실제 사용 출구의 FLOPs 평균
            layer.append(used.mean() + 1)
        rows.append(dict(
            name=f"thr{t:.3f}", threshold=t,
            mflops=round(float(np.mean(fl)) / 1e6, 2),
            mean_exit=round(float(np.mean(layer)), 3),
            compute_pct=round(float(np.mean(fl)) / exit_flops[-1] * 100, 2),
            window_acc=round(float(np.mean(wa)), 4),
            subject_acc=round(float(np.mean(sa)), 4),
            subject_acc_soft=round(float(np.mean(ss)), 4),
            roc_auc=round(float(np.mean(au)), 4)))

    cols = ["name", "threshold", "mflops", "compute_pct", "mean_exit",
            "window_acc", "subject_acc", "subject_acc_soft", "roc_auc"]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\n수치 저장: {args.out}")

    print(f"\n{'임계값':>8}{'MFLOPs':>10}{'연산%':>8}{'평균출구':>9}"
          f"{'윈도우acc':>11}{'hard':>8}{'soft':>8}{'AUC':>8}")
    for r in rows:
        print(f"{r['threshold']:>8.3f}{r['mflops']:>10.1f}{r['compute_pct']:>8.1f}"
              f"{r['mean_exit']:>9.2f}{r['window_acc']:>11.3f}"
              f"{r['subject_acc']:>8.3f}{r['subject_acc_soft']:>8.3f}"
              f"{r['roc_auc']:>8.3f}")

    png = args.out.replace(".csv", ".png")
    plot_pareto(rows, png)
    full = rows[-1]                                    # thr 1.0 = 조기종료 OFF
    best = max(rows, key=lambda r: (r["subject_acc"], -r["mflops"]))
    print(f"\n조기종료 OFF (thr 1.0): 피험자 {full['subject_acc']*100:.1f}% "
          f"@ {full['mflops']:.1f} MFLOPs")
    print(f"최적 운영점 (thr {best['threshold']:.3f}): 피험자 {best['subject_acc']*100:.1f}% "
          f"@ {best['mflops']:.1f} MFLOPs ({best['compute_pct']:.1f}%)")


if __name__ == "__main__":
    main()
