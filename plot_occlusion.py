"""
6채널 파형 위에 중요 구간을 칠한 대표 그림 (X3) — 논문 대표 그림 2.

왜 나선 그림이 아니라 파형인가
    이 데이터는 태블릿 좌표가 아니라 스마트펜 센서값이라 궤적이 없다.
    원본의 *Spiral/*.jpg는 완성된 그림이지 시간축 궤적이 아니라 대응시킬 수
    없다. 대신 6채널 파형 heatmap은 "어느 채널의 어느 구간인가"를 동시에
    보여주므로 정보량은 오히려 많다.

흑백 인쇄 대비
    학술지·예선 자료는 흑백으로 인쇄될 수 있다. 색상만으로 구분되는 그림은
    그때 읽히지 않는다. 그래서 같은 그림을 회색조로 한 벌 더 저장하고,
    음영 농도만으로도 중요 구간이 구별되는지 눈으로 확인할 수 있게 한다.

사용법
    python plot_occlusion.py --configs p16d128L1 --seed 42 --win-ms 200

산출물
    results/fig_occlusion_{cfg}_s{seed}_w{win}ms.png        (컬러)
    results/fig_occlusion_{cfg}_s{seed}_w{win}ms_gray.png   (회색조)
    results/fig_occlusion_profile_{cfg}_s{seed}_w{win}ms.png (군별 평균 프로파일)
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
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from dataset import load_npz

RESULT_DIR = "results"
CH_NAMES = ["ch0", "ch1", "ch2", "ch3", "ch4", "ch5"]


def importance_curve(drop_row, pos, seq_len, win_len):
    """위치별 하락폭을 시간축 곡선으로 편다.

    구간이 겹치므로 (stride < win_len) 각 시점의 중요도는 그 시점을 덮는
    모든 구간의 평균으로 둔다. 겹치는 횟수가 위치마다 달라서, 합이 아니라
    평균이어야 가장자리가 인위적으로 낮아지지 않는다.
    """
    acc = np.zeros(seq_len)
    cnt = np.zeros(seq_len)
    for v, p in zip(drop_row, pos):
        acc[p:p + win_len] += v
        cnt[p:p + win_len] += 1
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)


def pick_examples(subj, yy, drop, n_ch, n_pat=3, n_hea=2):
    """대표 윈도우를 고른다 — 군별로 '중요도가 뚜렷한' 피험자에서 하나씩.

    가장 극단적인 윈도우를 고르면 체리피킹으로 보인다. 그래서 피험자별로
    전채널 중요도의 중앙값을 구하고, 그 중앙값 기준 상위 피험자를 고른 뒤
    그 사람 안에서는 중앙값에 가까운 윈도우를 쓴다.
    """
    out = []
    for grp, k in [(1, n_pat), (0, n_hea)]:
        subj_score = {}
        for s in np.unique(subj[yy == grp]):
            m = (subj == s) & (yy == grp)
            subj_score[s] = np.median(drop[m, n_ch, :].max(axis=1))
        top = sorted(subj_score, key=subj_score.get, reverse=True)[:k]
        for s in top:
            idx = np.where((subj == s) & (yy == grp))[0]
            score = drop[idx, n_ch, :].max(axis=1)
            pick = idx[np.argsort(score)[len(score) // 2]]   # 중앙값 윈도우
            out.append((pick, s, grp))
    return out


def draw(X, drop, pos, win_len, examples, n_ch, path, gray=False):
    cmap = "Greys" if gray else "Reds"
    line_c = "0.15" if gray else "#1f2a44"
    n = len(examples)
    fig, axes = plt.subplots(n_ch, n, figsize=(4.2 * n, 1.15 * n_ch),
                             sharex=True, squeeze=False)
    seq_len = X.shape[2]
    t = np.arange(seq_len) / 1000.0

    for col, (i, s, grp) in enumerate(examples):
        # 한 예시 안에서는 채널 간 중요도 눈금을 공유한다.
        # 채널마다 따로 정규화하면 "이 채널이 더 중요하다"가 사라진다.
        curves = np.stack([importance_curve(drop[i, c], pos, seq_len, win_len)
                           for c in range(n_ch)])
        vmax = max(curves.max(), 1e-9)
        if gray:
            vmax *= 1.35   # 최대 음영을 약 74% 회색에서 멈춘다

        for c in range(n_ch):
            ax = axes[c][col]
            ax.imshow(curves[c][None, :], aspect="auto", cmap=cmap,
                      vmin=0, vmax=vmax, alpha=0.85,
                      extent=[t[0], t[-1], -1, 1])
            sig = X[i, c]
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            # 파형에 흰 외곽선을 둘러 진한 음영 위에서도 보이게 한다.
            # 흑백 인쇄에서 선과 음영이 모두 검정이면 파형이 묻힌다.
            ax.plot(t, np.clip(sig / 3.5, -0.95, 0.95), lw=0.6, color=line_c,
                    path_effects=[pe.withStroke(linewidth=1.6,
                                                foreground="white",
                                                alpha=0.85)])
            ax.set_ylim(-1, 1)
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(CH_NAMES[c], rotation=0, ha="right", va="center",
                              fontsize=9)
            if c == 0:
                ax.set_title(f"{'환자' if grp else '정상'} · {s}", fontsize=10)
            if c == n_ch - 1:
                ax.set_xlabel("시간 (초)", fontsize=9)

    fig.suptitle("occlusion 중요도 — 진할수록 가렸을 때 확률이 크게 떨어진 구간",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  → {path}")


def draw_profile(drop, pos, win_len, yy, n_ch, seq_len, path):
    """군별·채널별 평균 중요도 프로파일. 대표 그림의 근거가 된다."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
    t = np.arange(seq_len) / 1000.0
    for ax, grp, name in [(axes[0], 1, "환자군"), (axes[1], 0, "정상군")]:
        m = yy == grp
        for c in range(n_ch):
            cur = importance_curve(drop[m, c, :].mean(axis=0), pos,
                                   seq_len, win_len)
            ax.plot(t, cur, lw=1.3, label=CH_NAMES[c])
        ax.set_title(f"{name} 평균 (n={int(m.sum())} 윈도우)", fontsize=10)
        ax.set_ylabel("확률 하락폭")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(ncol=6, fontsize=8, loc="upper right")
    axes[1].set_xlabel("시간 (초)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  → {path}")


def main():
    ap = argparse.ArgumentParser(description="occlusion 대표 그림 (X3)")
    ap.add_argument("--configs", default="p16d128L1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--win-ms", type=int, default=200,
                    help="그림에는 해상도가 높은 짧은 창이 낫다")
    ap.add_argument("--data-path", default="data/windows_2s.npz")
    ap.add_argument("--mask-type", default="mean", choices=["mean", "interp"],
                    help="어느 마스킹으로 만든 occlusion 결과를 읽을지")
    ap.add_argument("--n-patient", type=int, default=3)
    ap.add_argument("--n-healthy", type=int, default=2)
    ap.add_argument("--importance", default="abs", choices=["signed", "abs"],
                    help="그림의 기본은 크기(abs). 부호 있는 하락폭은 마스킹의 "
                         "클래스 편향 방향을 타서, 한쪽 군의 그림이 통째로 "
                         "비어 보이는 일이 생긴다")
    args = ap.parse_args()

    for fam in ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]:
        try:
            matplotlib.rcParams["font.family"] = fam
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    npz = os.path.join(
        RESULT_DIR, f"occlusion_{args.configs}_s{args.seed}_w{args.win_ms}ms"
        f"{'' if args.mask_type == 'mean' else '_' + args.mask_type}.npz")
    if not os.path.exists(npz):
        raise SystemExit(f"없음: {npz}  (occlusion.py를 먼저 돌리세요)")

    d = np.load(npz, allow_pickle=True)
    drop, pos = d["drop"], d["pos_start"]
    if args.importance == "abs":
        drop = np.abs(drop)
    win_len, subj, yy = int(d["win_len"]), d["subject"], d["y"]
    n_ch = drop.shape[1] - 1

    X, y, subject_id, task = load_npz(args.data_path)
    if len(X) != len(drop):
        keep = np.isin(subject_id, np.unique(subj))
        X = X[keep]
    assert len(X) == len(drop), f"윈도우 수 불일치 {len(X)} vs {len(drop)}"

    print(f"[{args.win_ms}ms] 윈도우 {len(drop)} · 채널 {n_ch} · 위치 {len(pos)}")
    ex = pick_examples(subj, yy, drop, n_ch, args.n_patient, args.n_healthy)
    print(f"  대표: {[(s, '환자' if g else '정상') for _, s, g in ex]}")

    base = os.path.join(
        RESULT_DIR,
        f"fig_occlusion_{args.configs}_s{args.seed}_w{args.win_ms}ms"
        f"{'_abs' if args.importance == 'abs' else ''}")
    draw(X, drop, pos, win_len, ex, n_ch, base + ".png", gray=False)
    draw(X, drop, pos, win_len, ex, n_ch, base + "_gray.png", gray=True)
    draw_profile(drop, pos, win_len, yy, n_ch, X.shape[2],
                 os.path.join(RESULT_DIR,
                              f"fig_occlusion_profile_{args.configs}_"
                              f"s{args.seed}_w{args.win_ms}ms"
                              f"{'_abs' if args.importance == 'abs' else ''}.png"))


if __name__ == "__main__":
    main()
