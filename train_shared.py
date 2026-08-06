"""
공유 인코더 학습 — 필기·음성을 함께 학습하고, 평가는 모달별로 따로 한다.

한 스텝의 흐름
    필기 배치 → 필기 stem → 공유 인코더 → 필기 head → 손실 ┐
                                                            ├→ 합산 → backward 1회
    음성 배치 → 음성 stem → 공유 인코더 → 음성 head → 손실 ┘

    두 기울기가 같은 인코더로 흘러들어간다. 필기 배치로 인코더를 조금 고치고
    음성 배치로 또 고치는 식이 아니라, **한 번의 backward에서 두 신호가 합쳐져**
    인코더를 함께 다듬는다. 각자 배치를 따로 넣으므로 짝 데이터가 필요 없다.

평가
    학습이 끝나면 필기만 넣어도 예측이 나온다(음성에서 배운 것은 이미 가중치
    안에 있다). 그래서 평가 단위는 여전히 모달 하나다 — 필기 test 피험자는
    필기 head로, 음성 test 화자는 음성 head로. 같은 fold·같은 사람에 대해
    단독 모델과 숫자를 직접 맞대면 A/B 비교가 그대로 성립한다.

실행
    # 파이프라인 점검 (실제 데이터 없이, 합성 데이터로 수십 초)
    python train_shared.py --smoke-test

    # 두 모달 함께
    python train_shared.py --hw-path data/windows_2s.npz \
                           --voice-path data/voice_windows.npz

    # 단일 모달 (공유 코드가 단독 학습을 재현하는지 확인용)
    python train_shared.py --modalities hw --hw-path data/windows_2s.npz

⚠️ 이 스크립트는 아직 Early Exit을 포함하지 않는다. 기본 골격이 먼저다.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from evaluate import subject_vote_both
from shared_dataset import (PairedBatches, build_folds, class_weights,
                            load_modality, make_loaders, summarize, synthetic)
from shared_model import SharedPatchTST

RESULT_DIR = "results"

# 모달별 패치 설정 기본값 — 각자의 스윕에서 나온 최적값이다.
#   필기: patch 100/50  (T=2000 → 토큰 39개)
#   음성: patch 24/12   (T=126  → 토큰 9개)
# d_model·층 수는 인코더를 공유해야 하므로 모달별로 다르게 둘 수 없다.
MODALITY_DEFAULTS = {
    "hw":    dict(patch_len=100, stride=50),
    "voice": dict(patch_len=24,  stride=12),
}


def parse_args():
    p = argparse.ArgumentParser(description="Shared PatchTST 학습 (Early Exit 없음)")

    p.add_argument("--modalities", default="hw,voice",
                   help="쉼표 구분. 예: hw / voice / hw,voice")
    p.add_argument("--hw-path",    default="data/windows_2s.npz")
    p.add_argument("--voice-path", default="data/voice_windows.npz")

    # 공유 인코더 — 모달 공통
    p.add_argument("--d-model",  type=int, default=64)
    p.add_argument("--n-heads",  type=int, default=4)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--d-ff",     type=int, default=128)

    # 모달 전용 패칭
    p.add_argument("--hw-patch-len",    type=int, default=MODALITY_DEFAULTS["hw"]["patch_len"])
    p.add_argument("--hw-stride",       type=int, default=MODALITY_DEFAULTS["hw"]["stride"])
    p.add_argument("--voice-patch-len", type=int, default=MODALITY_DEFAULTS["voice"]["patch_len"])
    p.add_argument("--voice-stride",    type=int, default=MODALITY_DEFAULTS["voice"]["stride"])

    # 학습
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--head-dropout", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="CE 라벨 스무딩. softmax 과신을 줄인다")

    # 분할·평가
    p.add_argument("--n-folds",  type=int,   default=5)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--stratified", action="store_true",
                   help="StratifiedGroupKFold 사용 (기본: GroupKFold)")
    p.add_argument("--pair-mode", default="longest", choices=["longest", "shortest"],
                   help="모달 로더 길이가 다를 때. longest=긴 쪽 기준(짧은 쪽 순환)")

    p.add_argument("--smoke-test", action="store_true",
                   help="합성 데이터로 파이프라인만 점검 (실제 npz 불필요)")
    p.add_argument("--out", default=None, help="기본: results/shared_<태그>.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    a = p.parse_args()
    a.modalities = [m.strip() for m in a.modalities.split(",") if m.strip()]
    for m in a.modalities:
        if m not in MODALITY_DEFAULTS:
            p.error(f"모르는 모달: {m} (가능: {list(MODALITY_DEFAULTS)})")
    if a.out is None:
        tags = ["-".join(a.modalities), f"d{a.d_model}L{a.n_layers}"]
        if a.smoke_test:
            tags.insert(0, "smoke")
        if a.seed != 42:
            tags.append(f"s{a.seed}")
        a.out = os.path.join(RESULT_DIR, f"shared_{'_'.join(tags)}.csv")
    return a


# ──────────────────────────── 데이터 준비 ────────────────────────────

def gather_data(args):
    """{모달: dict(X=, y=, subject_id=)}"""
    data = {}
    for i, m in enumerate(args.modalities):
        if args.smoke_test:
            # 실제와 같은 채널 수·길이를 쓰되 피험자만 줄인다. shape이 달라야
            # "채널 수가 다른 모달이 한 인코더를 통과하는가"가 실제로 검증된다.
            spec = dict(hw=dict(n_subjects=20, n_channels=6,  seq_len=2000),
                        voice=dict(n_subjects=24, n_channels=64, seq_len=126))[m]
            # 모달마다 다른 seed — 같은 파형이 두 번 나오면 점검이 무의미하다
            X, y, sid = synthetic(**spec, win_per_subject=8, seed=args.seed + i)
        else:
            path = args.hw_path if m == "hw" else args.voice_path
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[{m}] {path} 가 없습니다. data/에 npz를 두거나 "
                    f"--smoke-test 로 파이프라인만 확인하세요.")
            X, y, sid = load_modality(path)
        data[m] = dict(X=X, y=y, subject_id=sid)
    return data


def build_specs(data, args):
    patch = dict(hw=(args.hw_patch_len, args.hw_stride),
                 voice=(args.voice_patch_len, args.voice_stride))
    specs = {}
    for m, d in data.items():
        pl, st = patch[m]
        specs[m] = dict(num_channels=d["X"].shape[1], seq_len=d["X"].shape[2],
                        patch_len=pl, stride=st)
    return specs


# ──────────────────────────── 학습·평가 ────────────────────────────

def eval_modality(model, modality, loader, criterion, device):
    """(loss, 윈도우acc, preds, labels, probs) — 한 모달을 그 모달 head로 평가."""
    model.eval()
    total, preds, labels, probs = 0.0, [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X, modality)
            total += criterion(logits, y).item() * X.size(0)
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(y.cpu())
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    preds  = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    probs  = torch.cat(probs).numpy()
    return (total / len(loader.dataset), float((preds == labels).mean()),
            preds, labels, probs)


def train_fold(model, train_loaders, val_loaders, criteria, args, ckpt):
    """공동 학습 1 fold. 조기중단 기준은 **모달 val loss의 합**.

    합으로 거는 이유는 인코더가 하나뿐이라 모달별로 다른 시점에 멈출 수 없기
    때문이다. 대신 한 모달이 먼저 과적합해도 다른 모달이 아직 내려가고 있으면
    합이 계속 줄어 멈추지 않는다 — 이건 공유 구조의 구조적 타협이고,
    모달별 val loss를 따로 찍어 눈으로 확인할 수 있게 해 뒀다.
    """
    device    = torch.device(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    paired = PairedBatches(train_loaders, mode=args.pair_mode)

    best_val, no_improve, best_epoch, hist = float("inf"), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = {m: 0.0 for m in train_loaders}
        for batches in paired:
            optimizer.zero_grad()
            loss = 0.0
            for m, (Xb, yb) in batches.items():
                # 모달별 손실을 합산해 backward는 한 번만 → 옵티마이저 스텝 수가
                # 단독 학습과 같아진다 (shared_dataset.PairedBatches 주석 참고)
                lm = criteria[m](model(Xb.to(device), m), yb.to(device))
                run_loss[m] += lm.item()
                loss = loss + lm
            loss.backward()
            optimizer.step()

        val = {m: eval_modality(model, m, val_loaders[m], criteria[m], device)[0]
               for m in val_loaders}
        val_total = sum(val.values())
        scheduler.step(val_total)
        hist.append((epoch, {m: v / len(paired) for m, v in run_loss.items()}, val))

        if val_total < best_val:
            best_val, no_improve, best_epoch = val_total, 0, epoch
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    model.load_state_dict(torch.load(ckpt))
    return model, len(hist), best_epoch, hist


def main():
    args = parse_args()
    os.makedirs(RESULT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── 데이터 ──
    data  = gather_data(args)
    specs = build_specs(data, args)
    print(f"device={args.device}  seed={args.seed}  "
          f"{'[합성 데이터 · 스모크 테스트]' if args.smoke_test else ''}")
    for m, d in data.items():
        print("  " + summarize(m, d["X"], d["y"], d["subject_id"]))

    # ── 모델 개요 (fold와 무관하므로 미리 한 번) ──
    probe = SharedPatchTST(specs, d_model=args.d_model, n_heads=args.n_heads,
                           n_layers=args.n_layers, d_ff=args.d_ff)
    print("\n" + probe.describe())
    rep = probe.param_report()
    del probe
    torch.manual_seed(args.seed)   # probe가 소비한 RNG를 되돌린다 (fold 초기화 재현성)

    # ── fold: 모달마다 독립적으로 자른다 (피험자 수가 다르므로) ──
    folds = {m: build_folds(d["y"], d["subject_id"], args.n_folds, args.seed,
                            val_size=args.val_size, stratified=args.stratified)
             for m, d in data.items()}

    print(f"\n{args.n_folds}-fold 학습 시작  "
          f"(pair-mode={args.pair_mode}, 최대 {args.epochs} epoch)\n")

    acc = {m: dict(window=[], subject=[], subject_soft=[], auc=[]) for m in data}
    epochs_used = []
    ckpt = os.path.join(RESULT_DIR, f"_tmp_shared_{os.getpid()}.pt")

    for fold in range(args.n_folds):
        loaders = {m: make_loaders(data[m]["X"], data[m]["y"], folds[m][fold],
                                   args.batch_size) for m in data}
        criteria = {
            m: nn.CrossEntropyLoss(
                weight=torch.tensor(
                    class_weights(data[m]["y"][folds[m][fold][0]]),
                    dtype=torch.float32).to(device),
                label_smoothing=args.label_smoothing)
            for m in data
        }
        model = SharedPatchTST(
            specs, d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, d_ff=args.d_ff,
            dropout=args.dropout, head_dropout=args.head_dropout).to(device)

        model, n_ep, best_ep, _ = train_fold(
            model,
            {m: l[0] for m, l in loaders.items()},
            {m: l[1] for m, l in loaders.items()},
            criteria, args, ckpt)
        epochs_used.append(n_ep)

        print(f"[fold {fold + 1}/{args.n_folds}]  {n_ep} epoch (best {best_ep})")
        for m in data:
            _, w_acc, preds, labels, probs = eval_modality(
                model, m, loaders[m][2], criteria[m], device)
            te = folds[m][fold][2]
            hard, soft = subject_vote_both(preds, probs, data[m]["subject_id"][te], labels)
            auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
            acc[m]["window"].append(w_acc)
            acc[m]["subject"].append(hard)
            acc[m]["subject_soft"].append(soft)
            acc[m]["auc"].append(auc)
            print(f"    {m:<6} 윈도우 {w_acc:.3f}   피험자 hard {hard:.3f} / "
                  f"soft {soft:.3f}   AUC {auc:.3f}   "
                  f"(test {len(set(data[m]['subject_id'][te].tolist()))}명)")

    if os.path.exists(ckpt):
        os.remove(ckpt)

    # ── 요약 ──
    print(f"\n{'모달':<8}{'윈도우acc':>16}{'피험자 hard':>18}{'피험자 soft':>18}{'AUC':>16}")
    rows = []
    for m in data:
        a = acc[m]
        cells = []
        for k in ("window", "subject", "subject_soft", "auc"):
            v = np.array(a[k], dtype=float)
            cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
        print(f"{m:<8}{cells[0]:>16}{cells[1]:>18}{cells[2]:>18}{cells[3]:>16}")
        rows.append(dict(
            modality=m,
            n_subjects=len(set(data[m]["subject_id"].tolist())),
            n_windows=len(data[m]["y"]),
            d_model=args.d_model, n_layers=args.n_layers,
            patch_len=specs[m]["patch_len"], stride=specs[m]["stride"],
            n_tokens=(specs[m]["seq_len"] - specs[m]["patch_len"]) // specs[m]["stride"] + 1,
            window_acc=round(float(np.mean(acc[m]["window"])), 4),
            window_std=round(float(np.std(acc[m]["window"])), 4),
            subject_acc=round(float(np.mean(acc[m]["subject"])), 4),
            subject_acc_std=round(float(np.std(acc[m]["subject"])), 4),
            subject_acc_soft=round(float(np.mean(acc[m]["subject_soft"])), 4),
            roc_auc=round(float(np.mean(acc[m]["auc"])), 4),
            mean_epochs=round(float(np.mean(epochs_used)), 1),
            params_shared=rep["shared"], params_own=rep["own"][m],
            shared_ratio=round(rep["shared_ratio"], 4),
            pair_mode=args.pair_mode, seed=args.seed))

    cols = list(rows[0])
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\n수치 저장: {args.out}")

    n_min = min(len(set(data[m]["subject_id"].tolist())) for m in data)
    print(f"  ⚠️ 피험자가 가장 적은 모달이 {n_min}명 → 1명 ≈ "
          f"{100 / n_min:.1f}%p. 단일 seed·개별 fold는 과대해석 금지.")
    print(f"  ⚠️ 이 숫자는 **같은 설정(d{args.d_model}/L{args.n_layers})으로 돌린 "
          f"단독 모델**과만 비교해야 한다. 설정이 다르면 공유 효과와 설정 효과가 섞인다.")


if __name__ == "__main__":
    main()
