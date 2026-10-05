"""
학습된 모델을 고정한 채 평가 시점에 대역을 지운다 (R1의 짝).

왜 필요한가 — 재학습 실험만으로는 답이 안 나온다
    임시연 R1(2026-09-08)은 4~6Hz를 제거한 입력으로 **재학습**했고 성능이
    유지됐다(AUC .8664 -> .8674). 그런데 그 결과로는 "원래 모델이 4~6Hz를
    쓰지 않았다"고 말할 수 없다. **재학습 과정에서 모델이 우회로를 찾았을 수
    있기 때문**이다. 정보가 중복돼 있으면 한 경로를 막아도 다른 경로로 같은
    성능이 나온다.

    원래 모델의 의존성을 재려면 **모델을 고정하고 입력만 바꿔야** 한다.
    이 스크립트가 그것이다. R1 보고서 §4가 명시적으로 요청한 확인이다.

무엇을 하는가
    학습된 체크포인트를 그대로 쓰고, 평가 입력에서만 지정 대역을 rFFT로
    0으로 만든 뒤 역변환해 넣는다. 그리고 예측 확률이 얼마나 변하는지 잰다.

    occlusion(§7)이 시간축에서 구간을 지운다면, 이것은 주파수축에서 대역을
    지운다. 같은 질문의 다른 축이다.

읽는 법
    AUC가 크게 떨어진다   -> 고정 모델이 그 대역에 의존한다
    AUC가 그대로다        -> 고정 모델도 그 대역 없이 판별한다.
                            그러면 §7의 "떨림 대역" 관찰은 상관이지 의존이 아니다

    군별로 나눠 본다. 환자에서만 떨어지면 §7의 군 특이적 결과와 이어진다.

주의 — 채널 1(그립력)은 R1의 pen 조건에 없었다
    R1 보고서의 채널 구성을 그대로 따른다:
        all  = 0~5 전체
        pen  = 2(축압력), 3~5(기울기 XYZ)
        mic  = 0(마이크)
    비교 가능하게 하려고 같은 묶음을 쓴다.

사용법
    python band_ablation.py --configs d64L2 --seed 42
    python band_ablation.py --configs d64L2 --seed 42 --band 4,6 --sets all,pen,mic
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
SETS = {"all": [0, 1, 2, 3, 4, 5], "pen": [2, 3, 4, 5], "mic": [0],
        "tilt": [3, 4, 5], "press": [1, 2]}


def band_stop(X, fs, lo, hi, chans):
    """지정 채널의 [lo, hi] 대역을 0으로 만들고 역변환한다.

    R1과 같은 방식이다 — rFFT에서 해당 빈을 0으로 두고 irfft.
    창 길이 2000, fs 1000이면 빈 간격이 0.5Hz이므로 4~6Hz는 빈 5개다.
    """
    out = X.copy()
    n = X.shape[-1]
    F = np.fft.rfft(X[:, chans, :], axis=-1)
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    F[:, :, (fr >= lo) & (fr <= hi)] = 0
    out[:, chans, :] = np.fft.irfft(F, n=n, axis=-1).astype(X.dtype)
    return out


def build(ck, n_cls):
    c = ck["config"]
    return PatchTSTClassifier(
        seq_len=ck["seq_len"], num_channels=ck["num_channels"], num_classes=n_cls,
        patch_len=c["patch_len"], stride=c["stride"], d_model=c["d_model"],
        n_heads=c["n_heads"], n_layers=c["n_layers"], d_ff=c["d_ff"],
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
    ap = argparse.ArgumentParser(description="고정 모델 평가 시점 대역 제거")
    ap.add_argument("--configs", default="d64L2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-path", default="data/windows_2s.npz")
    ap.add_argument("--band", default="4,6")
    ap.add_argument("--fs", type=int, default=1000)
    ap.add_argument("--sets", default="all,pen,mic")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    lo, hi = [float(x) for x in a.band.split(",")]

    X, y, subj, task = load_npz(a.data_path)
    n_cls = int(y.max() + 1)
    dev = torch.device(a.device)

    paths = sorted(glob.glob(
        os.path.join(RESULT_DIR, f"ckpt_{a.configs}_f*_s{a.seed}*.pt")))
    if not paths:
        raise SystemExit(f"체크포인트 없음: {a.configs} seed {a.seed}")

    fr = np.fft.rfftfreq(X.shape[-1], 1.0 / a.fs)
    nbin = int(((fr >= lo) & (fr <= hi)).sum())
    print(f"[{a.configs} seed {a.seed}] 체크포인트 {len(paths)}개 · 장치 {dev}")
    print(f"제거 대역 {lo:g}~{hi:g}Hz · 빈 {nbin}개 (간격 {fr[1]:.2f}Hz)\n")

    variants = {"none": None}
    for s in a.sets.split(","):
        s = s.strip()
        if s in SETS:
            variants[s] = SETS[s]

    # fold별로 그 fold의 test 피험자에만 그 fold 모델을 쓴다 (학습에 쓴 사람 제외)
    prob = {k: np.full(len(y), np.nan) for k in variants}
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        te = np.where(np.isin(subj, ck["test_subjects"]))[0]
        m = build(ck, n_cls); m.load_state_dict(ck["state_dict"]); m.to(dev)
        for name, chans in variants.items():
            Xi = X[te] if chans is None else band_stop(X[te], a.fs, lo, hi, chans)
            prob[name][te] = predict(m, Xi, dev)
        del m

    rows = []
    base = prob["none"]
    print(f"{'조건':<7}{'AUC':>9}{'무제거 대비':>12}{'윈도우acc':>10}"
          f"{'환자 Δp':>10}{'정상 Δp':>10}")
    for name in variants:
        pr = prob[name]
        assert not np.isnan(pr).any()
        auc = float(roc_auc_score(y, pr))
        acc = float(((pr >= .5).astype(int) == y).mean())
        # 정답 클래스 확률의 변화 — 군별로
        d_true = np.where(y == 1, base - pr, pr - base)   # 환자는 p 감소가 +
        dp_p = float(d_true[y == 1].mean())
        dp_h = float(d_true[y == 0].mean())
        rows.append(dict(config=a.configs, seed=a.seed, band=f"{lo:g}-{hi:g}Hz",
                         removed=name, roc_auc=round(auc, 4),
                         delta_auc=round(auc - float(roc_auc_score(y, base)), 4),
                         window_acc=round(acc, 4),
                         dp_patient=round(dp_p, 4), dp_healthy=round(dp_h, 4)))
        d = rows[-1]
        print(f"{name:<7}{auc:>9.4f}{d['delta_auc']:>+12.4f}{acc:>10.4f}"
              f"{dp_p:>+10.4f}{dp_h:>+10.4f}")

    print("\n  Δp는 '정답 클래스 확률이 얼마나 깎였는가'다 (양수 = 판단이 나빠짐).")
    worst = min(r["delta_auc"] for r in rows if r["removed"] != "none")
    if worst > -0.01:
        print("  -> 고정 모델도 이 대역 없이 판별한다. AUC 하락이 0.01 미만이다.")
        print("     R1(재학습)과 같은 방향이며, 재학습이 우회로를 찾은 것이")
        print("     아니라 애초에 의존이 약했음을 뜻한다.")
    else:
        print("  -> 고정 모델은 이 대역에 의존한다. R1의 재학습 결과는")
        print("     모델이 우회로를 찾은 것으로 읽어야 한다.")

    out = a.csv or os.path.join(
        RESULT_DIR, f"band_ablation_{a.configs}_s{a.seed}.csv")
    with io.open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
