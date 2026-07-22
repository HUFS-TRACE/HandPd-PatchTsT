import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              roc_auc_score)
from torch.utils.data import DataLoader

from dataset import WindowDataset, load_npz, subject_kfold
from model import PatchTSTClassifier


def parse_args():
    p = argparse.ArgumentParser(description="PatchTST 필기 분류 모델 학습")

    # 데이터
    p.add_argument("--data-path",     default="data/windows.npz")

    # 학습
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--n-folds",       type=int,   default=5)

    # 모델
    p.add_argument("--patch-len",     type=int,   default=16)
    p.add_argument("--stride",        type=int,   default=8)
    p.add_argument("--d-model",       type=int,   default=128)
    p.add_argument("--n-heads",       type=int,   default=8)
    p.add_argument("--n-layers",      type=int,   default=3)
    p.add_argument("--d-ff",          type=int,   default=256)
    p.add_argument("--dropout",       type=float, default=0.2)
    p.add_argument("--head-dropout",  type=float, default=0.2)

    # 클래스 가중치 (정상:환자)
    p.add_argument("--class-weight-healthy", type=float, default=2.0)
    p.add_argument("--class-weight-patient", type=float, default=1.0)

    # 기타
    p.add_argument("--ckpt-path",     default="best_model.pt")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss   = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.detach().cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()
    avg_loss   = total_loss / len(loader.dataset)
    acc        = (all_preds == all_labels).mean()

    return avg_loss, acc, all_preds, all_labels, all_probs


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 데이터 로드
    X, y, subject_id, task = load_npz(args.data_path)
    print(f"X shape: {X.shape}")
    print(f"y 분포: 정상={(y==0).sum()}, 환자={(y==1).sum()}")
    print(f"고유 사용자: {len(set(subject_id))}명")

    # K-Fold 분할
    folds = subject_kfold(subject_id, y,
                          n_splits=args.n_folds,
                          seed=args.seed)

    device       = torch.device(args.device)
    fold_results = []

    for fold, (train_idx, val_idx, test_idx) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1} / {args.n_folds}")
        print(f"train={len(train_idx)} "
              f"val={len(val_idx)} "
              f"test={len(test_idx)}")
        print(f"사용자 수: "
              f"train={len(set(subject_id[train_idx]))} "
              f"val={len(set(subject_id[val_idx]))} "
              f"test={len(set(subject_id[test_idx]))}")

        # Dataset & DataLoader
        train_ds = WindowDataset(X[train_idx], y[train_idx])
        val_ds   = WindowDataset(X[val_idx],   y[val_idx])
        test_ds  = WindowDataset(X[test_idx],  y[test_idx])

        train_loader = DataLoader(train_ds,
                                  batch_size=args.batch_size,
                                  shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

        # 모델 초기화 (매 fold마다 새로)
        model = PatchTSTClassifier(
            seq_len      = X.shape[2],
            num_channels = X.shape[1],
            num_classes  = int(y.max() + 1),
            patch_len    = args.patch_len,
            stride       = args.stride,
            d_model      = args.d_model,
            n_heads      = args.n_heads,
            n_layers     = args.n_layers,
            d_ff         = args.d_ff,
            dropout      = args.dropout,
            head_dropout = args.head_dropout,
        ).to(device)

        # 클래스 가중치
        class_weights = torch.tensor(
            [args.class_weight_healthy, args.class_weight_patient],
            dtype=torch.float32
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_val_loss     = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, *_ = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val_loss, val_acc, *_ = run_epoch(
                model, val_loader, criterion, device
            )
            scheduler.step(val_loss)

            print(f"epoch {epoch:03d} | "
                  f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss     = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), args.ckpt_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Best 모델로 test 평가
        model.load_state_dict(torch.load(args.ckpt_path))
        _, test_acc, preds, labels, probs = run_epoch(
            model, test_loader, criterion, device
        )

        print(f"\nFold {fold+1} 결과:")
        print(f"  test accuracy : {test_acc:.4f}")
        print(f"  test ROC-AUC  : {roc_auc_score(labels, probs):.4f}")
        print(classification_report(
            labels, preds,
            target_names=["healthy", "patient"]
        ))
        print("confusion matrix:")
        print(confusion_matrix(labels, preds))

        fold_results.append({
            'accuracy': test_acc,
            'roc_auc' : roc_auc_score(labels, probs),
        })

    # 최종 결과
    print(f"\n{'='*50}")
    print(f"{args.n_folds}-Fold 최종 결과")
    print(f"{'='*50}")
    accs = [r['accuracy'] for r in fold_results]
    aucs = [r['roc_auc']  for r in fold_results]
    print(f"평균 accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"평균 ROC-AUC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")


if __name__ == "__main__":
    main()
