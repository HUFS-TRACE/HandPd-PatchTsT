"""
Occlusion 기반 구간 중요도 (X1).

신호의 일부 구간을 가리고 정답 클래스 확률이 얼마나 떨어지는지 잰다.
많이 떨어질수록 그 구간이 예측에 중요하다.

핵심 규약 — 학습에 쓴 사람에게 설명을 돌리지 않는다
    피험자 s를 설명하려면 s가 test였던 fold의 모델을 써야 한다. 학습에 쓴
    모델로 뽑은 설명은 "모델이 외운 것"을 보는 것이라 근거가 되지 못한다.
    체크포인트에 함께 저장된 test_subjects로 짝을 맞추고, 어긋나면 죽는다.

사용법
    # 먼저 체크포인트를 만든다 (evaluate.py --save-ckpt)
    python occlusion.py --config configs/protocol.yaml --configs p16d128L1 --seed 42

    # 창 길이를 바꿔서
    python occlusion.py --config configs/protocol.yaml --win-ms 200,500,1000

산출물
    results/occlusion_{cfg}_s{seed}_w{win}ms.npz
        drop        (N, C+1, P)  윈도우 × 채널 × 위치별 확률 하락폭
                                 채널 축 마지막(C번)은 전 채널 동시 가림
        pos_start   (P,)         각 위치의 시작 샘플 인덱스
        subject/y/fold/task      (N,) 윈도우별 메타
        p_base      (N,)         가리기 전 정답 클래스 확률
    results/occlusion_{cfg}_s{seed}_summary.csv
        창 길이 × 채널 × 군(환자/정상)별 평균 하락폭

X2(대역 파워 검정)와의 관계 — 창 길이를 고를 때 반드시 볼 것
    X2는 4~6Hz 대역 파워를 잰다. 4Hz는 주기가 250ms이므로 그보다 짧은 구간
    에서는 그 대역의 파워를 추정할 수 없다. 1000Hz 표집 기준으로

        200 샘플(0.2초) = 0.8~1.2주기 → X2에 쓸 수 없다. X3 그림 전용
        500 샘플(0.5초) = 2~3주기     → 경계선
       1000 샘플(1.0초) = 4~6주기     → 쓸 만하다

    짧은 창은 중요도 해상도가 높아 히트맵(X3)에 좋고, 긴 창은 주파수 분석
    (X2)에 필요하다. 두 용도가 다르므로 창 길이도 따로 고른다.
    이 스크립트는 창별 원본(drop)을 그대로 남기므로, X2는 재실행 없이
    이 npz에서 선택 구간을 이어붙여 계산하면 된다.
"""
import sys
# Windows 콘솔(cp949)에서 일부 문자가 인코딩되지 않아 실행이 죽는 일을 막는다.
# 표시가 깨질지언정 실험이 중단되지는 않게 한다.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass
import argparse
import csv
import glob
import os

import numpy as np
import torch

from dataset import load_npz
from evaluate import build_model
from model import PatchTSTClassifier  # noqa: F401  (체크포인트 로드 시 필요)
from protocol import (add_common_args, apply_config_file, check_protocol,
                      filter_subjects)

RESULT_DIR = "results"
FS = 1000            # 표집 주파수 (Hz). 2초 창 = 2000 샘플
MIN_MS_FOR_BAND = 500  # 4~6Hz 추정에 필요한 최소 구간 길이 (2주기 이상)


# ──────────────────────────────────────────────────────────────────
# 가리기
# ──────────────────────────────────────────────────────────────────
def occlude(x, start, length, mask_type):
    """x의 [start, start+length) 구간을 가린 사본을 돌려준다.

    x: (B, C, T) — 이미 채널이 선택된 상태로 들어온다.

    0으로 채우지 않는 이유
        0은 그 자체가 이상 신호가 되어 모델이 "여기 뭔가 이상하다"에 반응한다.
        그러면 원래 그 구간이 중요해서 확률이 떨어진 것인지, 이상한 값을
        넣어서 떨어진 것인지 구별할 수 없다.
    """
    out = x.clone()
    end = start + length

    if mask_type == "mean":
        # 채널별 시간축 평균으로 채운다 (윈도우마다 따로 계산)
        fill = x.mean(dim=-1, keepdim=True)           # (B, C, 1)
        out[:, :, start:end] = fill

    elif mask_type == "interp":
        # 구간 양 끝값을 잇는 직선으로 채운다 (주변값 보간)
        left = x[:, :, start - 1] if start > 0 else x[:, :, end]
        right = x[:, :, end] if end < x.shape[-1] else x[:, :, start - 1]
        ramp = torch.linspace(0, 1, length, device=x.device, dtype=x.dtype)
        out[:, :, start:end] = (left.unsqueeze(-1) * (1 - ramp)
                                + right.unsqueeze(-1) * ramp)
    else:
        raise ValueError(f"모르는 mask_type: {mask_type}")

    return out


@torch.no_grad()
def occlusion_drops(model, X, y, positions, length, mask_type, device,
                    batch_size=64):
    """윈도우 × 채널 × 위치별 정답 클래스 확률 하락폭.

    채널 축은 C+1개다. 0..C-1은 해당 채널만 가린 경우, 마지막 C번은 전
    채널을 동시에 가린 경우. 전자는 "어느 채널의 어느 구간인가"(X3)에,
    후자는 시간 구간 자체의 중요도(X2·TimeSeg 비교)에 쓴다.
    """
    n, n_ch, _ = X.shape
    n_pos = len(positions)
    drop = np.zeros((n, n_ch + 1, n_pos), dtype=np.float32)
    p_base = np.zeros(n, dtype=np.float32)

    model.eval()
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        b = hi - lo
        xb = torch.from_numpy(X[lo:hi]).float().to(device)
        yb = torch.from_numpy(y[lo:hi]).long().to(device)

        base = model(xb).softmax(-1).gather(1, yb.unsqueeze(1)).squeeze(1)
        p_base[lo:hi] = base.cpu().numpy()

        for pi, start in enumerate(positions):
            # 한 위치의 (C+1)가지 가림 조건을 배치 축으로 쌓아 한 번에 통과시킨다.
            # 조건마다 따로 순전파하면 커널 실행 비용이 조건 수만큼 붙는데,
            # 모델이 작아서 그 오버헤드가 실제 연산보다 크다.
            filled = occlude(xb, start, length, mask_type)      # 전 채널 가림본
            variants = xb.unsqueeze(0).repeat(n_ch + 1, 1, 1, 1)  # (C+1,B,C,T)
            for c in range(n_ch):
                variants[c, :, c, start:start + length] = \
                    filled[:, c, start:start + length]          # 채널 c만 가림
            variants[n_ch] = filled                             # 전 채널 가림

            out = model(variants.reshape(-1, n_ch, X.shape[2]))
            p = out.softmax(-1).gather(
                1, yb.repeat(n_ch + 1).unsqueeze(1)).squeeze(1).reshape(n_ch + 1, b)
            drop[lo:hi, :, pi] = (base.unsqueeze(0) - p).T.cpu().numpy()

    return drop, p_base


# ──────────────────────────────────────────────────────────────────
# fold 짝 맞추기
# ──────────────────────────────────────────────────────────────────
def load_ckpts(cfg_name, seed):
    """fold별 체크포인트를 모아 (fold, 경로, 내용)으로 돌려준다."""
    pat = os.path.join(RESULT_DIR, f"ckpt_{cfg_name}_f*_s{seed}*.pt")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise SystemExit(
            f"체크포인트가 없습니다: {pat}\n"
            f"먼저 아래를 돌려 모델을 남기세요.\n"
            f"  python evaluate.py --config configs/protocol.yaml "
            f"--suite depth --configs {cfg_name} --seed {seed} --save-ckpt")

    out = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        out.append((ck["fold"], p, ck))
    out.sort(key=lambda t: t[0])
    return out


def main():
    parser = argparse.ArgumentParser(
        description="occlusion 구간 중요도 (X1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    g = parser.add_argument_group("occlusion")
    g.add_argument("--configs", default="p16d128L1",
                   help="체크포인트를 만든 설정 이름")
    g.add_argument("--win-ms", default="200,500,1000",
                   help="가릴 구간 길이(ms). 쉼표로 여러 개")
    g.add_argument("--stride-frac", type=float, default=0.5,
                   help="이동 간격을 창 길이의 몇 배로 할지")
    g.add_argument("--mask-type", default="mean", choices=["mean", "interp"],
                   help="가린 구간을 무엇으로 채울지. 0은 쓰지 않는다")
    g.add_argument("--occ-batch-size", type=int, default=64)
    g.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = apply_config_file(parser.parse_args())
    check_protocol(args)

    X, y, subject_id, task = load_npz(args.data_path)
    X, y, subject_id, task = filter_subjects(X, y, subject_id, task,
                                             args.subjects_file)
    n_ch, seq_len = X.shape[1], X.shape[2]
    device = torch.device(args.device)
    print(f"데이터 {args.data_path}  X{X.shape}  피험자 {len(set(subject_id))}명")
    print(f"장치 {device}  마스킹 {args.mask_type}\n")

    ckpts = load_ckpts(args.configs, args.seed)
    print(f"체크포인트 {len(ckpts)}개: fold {[f for f, _, _ in ckpts]}")

    # 모든 fold의 test 피험자를 합치면 전체가 되어야 한다. 겹치면 같은
    # 피험자를 두 모델로 설명하게 되므로 집계가 이상해진다.
    seen = []
    for _, _, ck in ckpts:
        seen.extend(ck["test_subjects"])
    if len(seen) != len(set(seen)):
        raise SystemExit("fold 간 test 피험자가 겹칩니다. 체크포인트를 확인하세요.")
    missing = set(subject_id.tolist()) - set(seen)
    if missing:
        print(f"  주의: 어느 fold의 test에도 없는 피험자 {len(missing)}명은 제외됩니다")

    # 마스킹 방식을 파일명에 남긴다. 기본값(mean)은 접미사 없이 두어
    # 이미 나온 결과·문서와 이름이 어긋나지 않게 한다.
    mtag = "" if args.mask_type == "mean" else f"_{args.mask_type}"

    summary_rows = []
    for win_ms in [int(s) for s in args.win_ms.split(",")]:
        length = int(round(win_ms * FS / 1000))
        if length >= seq_len:
            print(f"[{win_ms}ms] 창({length})이 신호({seq_len})보다 길어 건너뜁니다")
            continue
        step = max(1, int(round(length * args.stride_frac)))
        positions = list(range(0, seq_len - length + 1, step))

        note = "" if win_ms >= MIN_MS_FOR_BAND else "  ← X2(4~6Hz) 부적합, X3 전용"
        print(f"\n[{win_ms}ms] 창 {length}샘플 · 이동 {step} · 위치 {len(positions)}개{note}")

        drop = np.zeros((len(X), n_ch + 1, len(positions)), dtype=np.float32)
        p_base = np.zeros(len(X), dtype=np.float32)
        fold_of = np.full(len(X), -1, dtype=np.int8)

        for fold, path, ck in ckpts:
            # 이 fold의 모델이 학습에 쓰지 않은 피험자만 고른다
            te_subj = set(ck["test_subjects"])
            idx = np.where(np.isin(subject_id, list(te_subj)))[0]
            assert set(subject_id[idx].tolist()) <= te_subj, \
                "학습에 쓴 피험자가 섞였습니다"

            model = build_model(ck["config"], ck["seq_len"],
                                ck["num_channels"], int(y.max() + 1), args)
            model.load_state_dict(ck["state_dict"])
            model.to(device)

            d, pb = occlusion_drops(model, X[idx], y[idx], positions, length,
                                    args.mask_type, device,
                                    args.occ_batch_size)
            drop[idx], p_base[idx], fold_of[idx] = d, pb, fold
            print(f"  fold {fold}  피험자 {len(te_subj):2d}명 · 윈도우 {len(idx):5d}"
                  f"  평균 하락 {d[:, n_ch].mean():+.4f}")

        keep = fold_of >= 0
        out = os.path.join(
            RESULT_DIR,
            f"occlusion_{args.configs}_s{args.seed}_w{win_ms}ms{mtag}.npz")
        np.savez_compressed(
            out, drop=drop[keep], p_base=p_base[keep],
            pos_start=np.array(positions, dtype=np.int32),
            win_len=length, fs=FS, mask_type=args.mask_type,
            subject=subject_id[keep], y=y[keep], fold=fold_of[keep],
            task=(task[keep] if task is not None else np.array([""] * keep.sum())))
        print(f"  → {out}")

        # 군별·채널별 요약. 환자군에서만 커야 "떨림을 봤다"가 성립한다
        for grp, name in [(0, "정상"), (1, "환자")]:
            m = keep & (y == grp)
            for c in range(n_ch + 1):
                ch = "전채널" if c == n_ch else f"ch{c}"
                summary_rows.append(dict(
                    win_ms=win_ms, mask=args.mask_type, group=name, channel=ch,
                    n_windows=int(m.sum()),
                    mean_drop=round(float(drop[m, c].mean()), 5),
                    max_drop=round(float(drop[m, c].max(axis=1).mean()), 5),
                    usable_for_x2=win_ms >= MIN_MS_FOR_BAND))

    csv_path = os.path.join(
        RESULT_DIR,
        f"occlusion_{args.configs}_s{args.seed}_summary{mtag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n요약 → {csv_path}")

    print(f"\n{'창':>7}{'군':>6}{'채널':>8}{'평균하락':>10}{'최대하락':>10}")
    for r in summary_rows:
        print(f"{r['win_ms']:>5}ms{r['group']:>6}{r['channel']:>8}"
              f"{r['mean_drop']:>10.4f}{r['max_drop']:>10.4f}")


if __name__ == "__main__":
    main()
