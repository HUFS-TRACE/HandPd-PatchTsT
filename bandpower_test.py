"""
선택 구간의 떨림 대역 파워 검정 (X2) — ⑤단계의 결론.

모델이 중요하다고 짚은 구간과 그렇지 않은 구간의 4~6Hz 대역 파워를 비교한다.
환자군에서 선택 구간의 떨림 성분이 유의하게 크면, "모델이 떨림을 본다"가
모델 동작 수준에서 증명된다.

이 검정의 강점 — 정답 라벨이 필요 없다
    구간의 대역 파워는 신호 자체에서 계산된다. 라벨을 안 쓰므로 "모델이
    맞혔는가"와 독립적으로 "모델이 무엇을 봤는가"를 잴 수 있다.

설계에서 반드시 지킨 것 네 가지

1) 상대 파워를 주 지표로 쓴다
   절대 파워만 보면 "그냥 신호가 큰 구간을 봤다"와 구별되지 않는다. 진폭이
   큰 구간은 모든 대역의 파워가 함께 커지기 때문이다. 상대 파워
   (4~6Hz 파워 / 전체 파워)는 진폭에 불변이므로 "떨림 대역에 에너지가
   몰렸는가"만 남는다. 절대 파워도 함께 보고하되 해석은 상대 파워로 한다.

2) 채널별로 분리한다
   나이 교란이 채널마다 다르게 작용한다. 채널을 뭉치면 한 채널의 효과가
   전체로 번져 보인다.

3) 환자군·정상군을 분리한다
   정상군에서도 똑같이 나오면 "떨림을 봤다"가 아니라 "에너지가 큰 구간을
   봤다"일 수 있다. 환자군에서 크고 정상군에서 약해야 ⑤가 성립한다.

4) 다중비교를 보정한다
   채널 6개 × 군 2개 = 12번 검정한다. 보정 없이는 우연히 하나쯤 유의해진다.
   Holm-Bonferroni로 보정한 값을 함께 낸다.

주파수 해상도의 한계 — 논문 한계 절에 적을 것
    구간 길이가 N 샘플이면 주파수 해상도는 fs/N이다. 1000Hz 표집에서
        500 샘플(0.5초) → 2Hz 해상도. 4~6Hz에 빈이 2개뿐
       1000 샘플(1.0초) → 1Hz 해상도. 빈 3개
    그래서 이 검정은 1000ms 창을 주 분석으로 쓴다. 200ms(0.2초)는 4Hz의
    한 주기도 담지 못하므로 아예 대상에서 뺀다.

사용법
    python bandpower_test.py --configs p16d128L1 --seed 42
    python bandpower_test.py --win-ms 1000 --band 4,6

산출물
    results/bandpower_{cfg}_s{seed}_w{win}ms.csv       채널×군별 검정 결과
    results/bandpower_{cfg}_s{seed}_w{win}ms_subj.csv  피험자별 원본 값
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
import os

import numpy as np
from scipy import stats

from dataset import load_npz

RESULT_DIR = "results"
MIN_MS_FOR_BAND = 500     # 이보다 짧은 창은 4~6Hz를 추정할 수 없다


# ──────────────────────────────────────────────────────────────────
# 대역 파워
# ──────────────────────────────────────────────────────────────────
def band_power(seg, fs, lo, hi):
    """구간의 절대·상대 대역 파워.

    seg: (..., N) 마지막 축이 시간.
    반환: (절대 파워, 상대 파워) — 상대 = 대역 파워 / 전체 파워

    평균을 빼고(detrend) 창함수를 씌운다. 평균을 안 빼면 DC 성분이 새어
    저주파 대역을 부풀린다.
    """
    seg = seg - seg.mean(axis=-1, keepdims=True)
    n = seg.shape[-1]
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(seg * win, axis=-1)) ** 2
    freq = np.fft.rfftfreq(n, d=1.0 / fs)

    in_band = (freq >= lo) & (freq <= hi)
    # 0Hz는 전체 파워에서도 뺀다 (직류 성분은 떨림과 무관)
    total_mask = freq > 0

    p_band = spec[..., in_band].sum(axis=-1)
    p_total = spec[..., total_mask].sum(axis=-1)
    rel = np.divide(p_band, p_total, out=np.zeros_like(p_band),
                    where=p_total > 0)
    return p_band, rel


# ──────────────────────────────────────────────────────────────────
# 선택 / 비선택 구간 나누기
# ──────────────────────────────────────────────────────────────────
def segment_statistic(seg, fs, lo, hi, kind):
    """구간에서 잴 양. (주 지표, 보조 지표) 두 개를 돌려준다.

    왜 대역 파워만으로는 부족한가 — 원종윤 E1·E2 결과 (2026-09-06)
        4~6Hz 떨림 대역 파워는 6채널을 다 써도 창 AUC .6125로, 딥러닝(.8629)
        에 크게 못 미친다. 단일 축압력은 .4330으로 우연 이하다. 반면 단일
        피처 최고는 **그립력의 가속도 표준편차 .7274**이고, 상위 8개 중
        떨림 대역은 하나도 없다. 즉 이 과제의 저차원 축은 떨림 대역이
        아니라 **움직임의 변동성과 저주파 성분**이다.

        따라서 ⑤단계가 "모델이 ④의 저차원 축을 쓴다"를 보이려면, 떨림
        대역이 아니라 그 변동성을 재야 한다. 이 함수가 그 선택지를 준다.

    척도 문제
        SD 계열은 절대량이라 진폭이 큰 구간에서 모두 커진다. 그래서
        "모델이 그냥 큰 구간을 봤다"와 구별되지 않는다. 상대 파워가 그
        문제를 피하듯, SD 계열에도 진폭으로 나눈 비율 지표를 함께 낸다.
    """
    seg = seg - seg.mean(axis=-1, keepdims=True)
    eps = 1e-12

    if kind == "relpower":                       # 대역 상대 파워 (기존 동작)
        return band_power(seg, fs, lo, hi)

    amp = seg.std(axis=-1)                       # 진폭 변동성
    if kind == "amp_sd":
        return amp, amp

    if kind == "vel_sd":                         # 1차 차분 = 속도
        v = np.diff(seg, n=1, axis=-1).std(axis=-1)
        return v, v / (amp + eps)

    if kind == "accel_sd":                       # 2차 차분 = 가속도
        a = np.diff(seg, n=2, axis=-1).std(axis=-1)
        return a, a / (amp + eps)

    raise ValueError(f"모르는 statistic: {kind}")


def split_positions(drop_vec, frac=1 / 3):
    """중요도 상위/하위 위치를 고른다.

    drop_vec: (P,) 위치별 확률 하락폭.
    위치가 적을 때(1000ms 창은 3개)도 최소 1개씩은 잡히게 한다.
    """
    p = len(drop_vec)
    k = max(1, int(round(p * frac)))
    order = np.argsort(drop_vec)          # 오름차순
    return order[-k:], order[:k]          # (선택=상위, 비선택=하위)


def min_attainable_p(n):
    """표본 n에서 Wilcoxon 양측검정이 낼 수 있는 가장 작은 p.

    왜 필요한가 — 부호순위 검정은 이산적이라 표본이 적으면 효과가 아무리
    커도 도달할 수 없는 p 하한이 있다. n=7이면 하한이 0.0156이고, 채널 6 ×
    군 2 = 12회 보정을 거치면 0.1875가 되어 **유의가 원천적으로 불가능**하다.
    이 경우 "유의하지 않음"을 "효과가 없음"으로 읽으면 안 된다.
    """
    if n < 3:
        return 1.0
    d = np.arange(1, n + 1, dtype=float)     # 완전히 한 방향인 최대 효과
    return float(stats.wilcoxon(d, np.zeros(n)).pvalue)


def holm(pvals):
    """Holm-Bonferroni 보정. 입력 순서 그대로 보정된 p를 돌려준다."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)       # 단조 증가 보장
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser(
        description="선택 구간의 4~6Hz 대역 파워 검정 (X2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--configs", default="p16d128L1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-path", default="data/windows_2s.npz")
    ap.add_argument("--win-ms", default="500,1000",
                    help="쓸 occlusion 창. 500 미만은 4~6Hz 추정 불가라 거부한다")
    ap.add_argument("--mask-type", default="mean", choices=["mean", "interp"],
                    help="어느 마스킹으로 만든 occlusion 결과를 읽을지")
    ap.add_argument("--band", default="4,6", help="떨림 대역 (Hz)")
    ap.add_argument("--top-frac", type=float, default=1 / 3,
                    help="상위/하위 몇 분위를 선택/비선택으로 볼지")
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="순열 대조 반복 수. 위치 자체의 효과를 배제한다")
    ap.add_argument("--statistic", default="relpower",
                    choices=["relpower", "vel_sd", "accel_sd", "amp_sd"],
                    help="구간에서 잴 양. relpower=대역 상대파워(기존), "
                         "vel_sd/accel_sd=속도/가속도 표준편차, amp_sd=진폭. "
                         "E1·E2가 지목한 저차원 축은 떨림 대역이 아니라 변동성이다")
    ap.add_argument("--importance", default="signed",
                    choices=["signed", "abs"],
                    help="중요도 정의. signed=정답 확률 하락폭(관례), "
                         "abs=변화 크기. 마스킹이 특정 클래스 쪽으로 미는 "
                         "편향이 있을 때 signed는 모델마다 의미가 뒤집힌다")
    ap.add_argument("--task", default=None,
                    help="한 과제만 볼 때 (circle / diadochokinesis / meander / "
                         "spiral). E1이 circle에서 떨림 대역 AUC .4084로 우연 "
                         "이하임을 보였으므로 과제별 분리 보고가 필요하다")
    ap.add_argument("--channel-mode", default="per-channel",
                    choices=["per-channel", "joint"],
                    help="중요도를 채널별로 볼지, 전채널 동시 가림 값으로 볼지")
    args = ap.parse_args()

    lo, hi = [float(s) for s in args.band.split(",")]
    X, y, subject_id, task = load_npz(args.data_path)
    print(f"데이터 {args.data_path}  X{X.shape}")
    what = (f"대역 상대파워 {lo:g}~{hi:g}Hz" if args.statistic == "relpower"
            else {"vel_sd": "속도 표준편차", "accel_sd": "가속도 표준편차",
                  "amp_sd": "진폭 표준편차"}[args.statistic])
    print(f"구간 통계: {what} · 상위/하위 {args.top_frac:.2f} 분위\n")

    for win_ms in [int(s) for s in args.win_ms.split(",")]:
        if win_ms < MIN_MS_FOR_BAND:
            print(f"[{win_ms}ms] 건너뜀 — {lo}Hz는 주기가 "
                  f"{1000 / lo:.0f}ms라 이 창에서는 추정할 수 없다")
            continue

        npz = os.path.join(
            RESULT_DIR,
            f"occlusion_{args.configs}_s{args.seed}_w{win_ms}ms"
            f"{'' if args.mask_type == 'mean' else '_' + args.mask_type}.npz")
        if not os.path.exists(npz):
            print(f"[{win_ms}ms] 없음: {npz}  (occlusion.py를 먼저 돌리세요)")
            continue

        d = np.load(npz, allow_pickle=True)
        drop = d["drop"]                  # (N, C+1, P)
        pos = d["pos_start"]
        L = int(d["win_len"])
        fs = int(d["fs"])
        subj = d["subject"]
        yy = d["y"]
        n_ch = drop.shape[1] - 1
        df = fs / L
        print(f"[{win_ms}ms] 창 {L}샘플 · 위치 {len(pos)}개 · "
              f"주파수 해상도 {df:.1f}Hz · 대역 내 빈 "
              f"{int(np.sum((np.fft.rfftfreq(L, 1 / fs) >= lo) & (np.fft.rfftfreq(L, 1 / fs) <= hi)))}개")

        # 이 npz의 윈도우 순서에 맞춰 원신호를 정렬한다.
        # occlusion.py는 원본 인덱스 순서를 유지하므로 fold로 걸러진 부분만
        # 빠져 있다. subject+순서로 다시 맞추는 대신, 원본에서 같은 마스크를
        # 재현해 쓴다 (npz의 subject 배열과 원본 subject_id를 대조).
        if len(drop) == len(X):
            Xs = X
        else:
            # 어느 fold에도 없던 피험자가 빠진 경우
            keep = np.isin(subject_id, np.unique(subj))
            Xs = X[keep]
            assert len(Xs) == len(drop), \
                f"윈도우 수 불일치: 신호 {len(Xs)} vs occlusion {len(drop)}"

        # 위치별 대역 파워를 미리 계산한다 (윈도우 × 채널 × 위치).
        #
        # 청크로 나누는 이유 — 한 번에 하면 중간 배열이 수 GB가 된다.
        # 13,474윈도우 × 6채널 × 7위치 × 500샘플을 float32로만 담아도 1.1GB이고,
        # numpy의 rfft는 배정밀도로 계산해 complex128을 돌려주므로 스펙트럼이
        # 그 두 배를 더 쓴다. 창을 500ms로 잡으면 2GB를 넘겨 죽는다.
        n_win = len(Xs)
        abs_p = np.zeros((n_win, Xs.shape[1], len(pos)), dtype=np.float32)
        rel_p = np.zeros_like(abs_p)
        step_n = max(1, int(2e8 // (Xs.shape[1] * len(pos) * L)))   # 청크 크기
        for c0 in range(0, n_win, step_n):
            c1 = min(c0 + step_n, n_win)
            segs = np.stack([Xs[c0:c1, :, p:p + L] for p in pos], axis=2)
            chunk_abs, chunk_rel = segment_statistic(segs, fs, lo, hi,
                                                     args.statistic)
            abs_p[c0:c1], rel_p[c0:c1] = chunk_abs, chunk_rel
            del segs, chunk_abs, chunk_rel

        # ── 과제 필터 ──
        # 4개 과제(circle·diadochokinesis·meander·spiral)를 뭉쳐서 보면,
        # 과제마다 판별 정보가 다른 경우 서로 상쇄되거나 한 과제의 효과가
        # 전체로 번져 보인다. E1에서 circle의 떨림 대역 AUC가 .4084로 우연
        # 이하였으므로, 이 분리는 선택이 아니라 필요다.
        if args.task:
            tsel = d["task"] == args.task
            if tsel.sum() == 0:
                print(f"  과제 '{args.task}' 없음 - 건너뜀"); continue
            drop, subj, yy = drop[tsel], subj[tsel], yy[tsel]
            Xs = Xs[tsel]
            abs_p, rel_p = abs_p[tsel], rel_p[tsel]
            print(f"  과제 {args.task}: 윈도우 {int(tsel.sum())} "
                  f"(환자 {int((yy==1).sum())} / 정상 {int((yy==0).sum())}), "
                  f"피험자 {len(set(subj.tolist()))}명")

        rows, subj_rows = [], []
        for grp, gname in [(1, "환자"), (0, "정상")]:
            gm = yy == grp
            for c in range(n_ch):
                imp = drop[:, c, :] if args.channel_mode == "per-channel" \
                    else drop[:, n_ch, :]
                if args.importance == "abs":
                    # 마스킹이 한쪽 클래스로 미는 편향이 있으면, 부호 있는
                    # 하락폭은 "이 구간이 얼마나 중요한가"가 아니라 "어느
                    # 방향으로 미는가"를 재게 된다. 크기만 보면 그 편향과
                    # 무관하게 "예측을 얼마나 바꾸는가"가 남는다.
                    imp = np.abs(imp)

                # 피험자별로 모으고, 피험자 단위로 대응 검정한다.
                # 윈도우 단위로 검정하면 한 사람이 수십 표본을 내서
                # 독립성이 깨진다 (같은 사람의 윈도우는 서로 닮았다).
                #
                # 전부 벡터화한다. 윈도우가 1만 개가 넘고 순열을 1000회
                # 돌리므로, 파이썬 루프로는 채널 하나에 수 분이 걸린다.
                gi = np.where(gm)[0]
                P = len(pos)
                k = max(1, int(round(P * args.top_frac)))
                codes, inv = np.unique(subj[gi], return_inverse=True)
                cnt = np.bincount(inv, minlength=len(codes)).astype(float)

                def subj_mean(vals):
                    """윈도우 값을 피험자 평균으로 접는다."""
                    return np.bincount(inv, weights=vals, minlength=len(codes)) / cnt

                rp = rel_p[gi, c, :]                 # (Nw, P)
                apw = abs_p[gi, c, :]
                order = np.argsort(imp[gi], axis=1)  # 중요도 오름차순
                hi_i, lo_i = order[:, -k:], order[:, :k]
                take = np.take_along_axis
                sel_rel = subj_mean(take(rp, hi_i, 1).mean(1))
                non_rel = subj_mean(take(rp, lo_i, 1).mean(1))
                sel_abs = subj_mean(take(apw, hi_i, 1).mean(1))
                non_abs = subj_mean(take(apw, lo_i, 1).mean(1))

                for si, s in enumerate(codes):
                    subj_rows.append(dict(
                        win_ms=win_ms, group=gname, channel=f"ch{c}", subject=s,
                        n_windows=int(cnt[si]),
                        sel_rel=round(float(sel_rel[si]), 6),
                        non_rel=round(float(non_rel[si]), 6),
                        sel_abs=round(float(sel_abs[si]), 4),
                        non_abs=round(float(non_abs[si]), 4)))

                # ── 순열 대조 (귀무 분포) ──
                # 교란 하나를 배제하기 위한 것이다. 위치마다 신호에서의
                # 시간대가 다르므로, 가장자리 구간과 중앙 구간의 스펙트럼이
                # 애초에 다를 수 있다. 그렇다면 "상위 1/3 vs 하위 1/3"이라는
                # 분할은 중요도와 무관하게도 대역 파워 차이를 만든다.
                # 중요도 순위를 윈도우 안에서 무작위로 섞어 같은 통계를
                # 다시 계산하면, 그 차이가 중요도에서 온 것인지 위치에서 온
                # 것인지 갈린다.
                rng = np.random.default_rng(0)
                perm_diffs = np.empty(args.n_perm)
                for pi_ in range(args.n_perm):
                    sh = np.argsort(rng.random((len(gi), P)), axis=1)
                    a = take(rp, sh[:, :k], 1).mean(1)
                    b = take(rp, sh[:, -k:], 1).mean(1)
                    perm_diffs[pi_] = subj_mean(a).mean() - subj_mean(b).mean()

                diff = sel_rel - non_rel
                # 관측 차이가 귀무 분포에서 얼마나 극단적인가
                obs = diff.mean()
                perm_p = (np.sum(np.abs(perm_diffs) >= abs(obs)) + 1) / \
                         (len(perm_diffs) + 1)
                perm_sd = perm_diffs.std(ddof=1) if len(perm_diffs) > 1 else np.nan
                perm_z = (obs - perm_diffs.mean()) / perm_sd if perm_sd else np.nan
                # 대응표본 — 같은 사람 안에서 선택 vs 비선택을 비교한다.
                # Wilcoxon: 정규성을 가정하지 않는다 (피험자 수가 적다)
                if len(diff) >= 6 and np.any(diff != 0):
                    w_stat, w_p = stats.wilcoxon(sel_rel, non_rel)
                    t_stat, t_p = stats.ttest_rel(sel_rel, non_rel)
                else:
                    w_stat = w_p = t_stat = t_p = np.nan
                # 효과크기 (Cohen's d, 대응표본)
                dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

                # 대역 성분이 사실상 없는 채널 표시.
                # 마이크 채널의 4~6Hz 상대 파워는 1e-6 수준이다(다른 채널의
                # 10만분의 1). dz는 척도에 불변이라 이런 채널에서도 유의가
                # 뜨지만, 그 차이는 물리적으로 해석할 수 없다. 검정 가족에서
                # 빼면 결과를 보고 가족을 바꾸는 것이 되므로 그대로 두고
                # 표시만 한다.
                negligible = (args.statistic == "relpower" and
                              float(max(sel_rel.mean(), non_rel.mean())) < 1e-4)

                rows.append(dict(
                    win_ms=win_ms, band=f"{lo:g}-{hi:g}Hz", group=gname,
                    channel=f"ch{c}", n_subjects=len(diff),
                    band_negligible=negligible,
                    sel_rel=round(float(sel_rel.mean()), 6),
                    non_rel=round(float(non_rel.mean()), 6),
                    delta_rel=round(float(diff.mean()), 6),
                    sel_abs=round(float(sel_abs.mean()), 4),
                    non_abs=round(float(non_abs.mean()), 4),
                    cohen_dz=round(float(dz), 4),
                    perm_mean=round(float(perm_diffs.mean()), 6),
                    perm_z=round(float(perm_z), 3) if np.isfinite(perm_z) else "",
                    perm_p=round(float(perm_p), 4),
                    wilcoxon_p=float(w_p), ttest_p=float(t_p)))

        # 다중비교 보정 — 채널 6 × 군 2 = 12회
        ps = [r["wilcoxon_p"] for r in rows]
        for r, adj in zip(rows, holm(ps)):
            r["wilcoxon_p"] = round(r["wilcoxon_p"], 6) if np.isfinite(r["wilcoxon_p"]) else ""
            r["ttest_p"] = round(r["ttest_p"], 6) if np.isfinite(r["ttest_p"]) else ""
            r["p_holm"] = round(float(adj), 6) if np.isfinite(adj) else ""
            r["sig_holm"] = "*" if np.isfinite(adj) and adj < 0.05 else ""

        # 파일명에 조건을 전부 남긴다. 대역을 빼먹었다가 대역 훑기가
        # 기본 대역(4~6Hz) 결과를 덮어쓴 적이 있다. 기본 조건
        # (4~6Hz · mean)은 접미사를 붙이지 않아 기존 이름과 어긋나지 않게 한다.
        btag = "" if (lo, hi) == (4.0, 6.0) else f"_b{lo:g}-{hi:g}"
        if args.task:
            btag += f"_{args.task[:4]}"
        if args.statistic != "relpower":
            btag = f"_{args.statistic}"   # 통계가 바뀌면 대역은 의미가 없다
        mtag = "" if args.mask_type == "mean" else f"_{args.mask_type}"
        out = os.path.join(
            RESULT_DIR,
            f"bandpower_{args.configs}_s{args.seed}_w{win_ms}ms"
            f"{'_abs' if args.importance == 'abs' else ''}{btag}{mtag}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        out2 = out.replace(".csv", "_subj.csv")
        with open(out2, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(subj_rows[0].keys()))
            w.writeheader(); w.writerows(subj_rows)

        print(f"\n{'군':>5}{'채널':>7}{'선택':>10}{'비선택':>10}{'차이':>10}"
              f"{'dz':>8}{'p(Holm)':>10}")
        neg = [r['channel'] for r in rows if r.get('band_negligible')]
        if neg:
            print(f"\n  [주의] 대역 성분이 사실상 없는 채널: "
                  f"{sorted(set(neg))} - 상대 파워가 1e-4 미만이라 "
                  f"유의성이 떠도 해석하지 않는다")
        for r in rows:
            print(f"{r['group']:>5}{r['channel']:>7}{r['sel_rel']:>10.4f}"
                  f"{r['non_rel']:>10.4f}{r['delta_rel']:>+10.4f}"
                  f"{r['cohen_dz']:>8.2f}{str(r['p_holm']):>10}{r['sig_holm']:<1}"
                  f"{str(r['perm_z']):>9}{r['perm_p']:>8.3f}")
        print(f"→ {out}\n  → {out2}\n")

        # ── 검정력 사전 점검 ──
        # 표본이 적으면 효과 크기와 무관하게 유의가 불가능하다. 그 경우
        # "유의하지 않음"을 "효과 없음"으로 읽으면 안 되므로 먼저 경고한다.
        n_tests = len(rows)
        underpowered = []
        for gname in ("환자", "정상"):
            ns = [r["n_subjects"] for r in rows if r["group"] == gname]
            if not ns:
                continue
            n = ns[0]
            floor = min_attainable_p(n) * n_tests
            if floor >= 0.05:
                underpowered.append((gname, n, floor))
        for gname, n, floor in underpowered:
            print(f"\n  [주의] {gname}군 n={n} — Wilcoxon 최소 달성 가능 p에 "
                  f"{n_tests}회 보정을 적용하면 {floor:.3f}. "
                  f"**유의가 원천적으로 불가능**하다.\n"
                  f"    이 군의 결론은 p가 아니라 효과 크기(dz)와 "
                  f"군 간 비교로 내려야 한다.")

        # 해석 도우미 — 환자군에서만 커야 ⑤가 성립한다
        pat = [r for r in rows if r["group"] == "환자" and r["sig_holm"]]
        hea = [r for r in rows if r["group"] == "정상" and r["sig_holm"]]
        dz_pat = np.nanmean([r["cohen_dz"] for r in rows if r["group"] == "환자"])
        dz_hea = np.nanmean([r["cohen_dz"] for r in rows if r["group"] == "정상"])
        print(f"\n  유의(Holm<0.05): 환자 {len(pat)}/{n_ch}채널, "
              f"정상 {len(hea)}/{n_ch}채널")
        print(f"  평균 효과크기 dz: 환자 {dz_pat:+.2f} / 정상 {dz_hea:+.2f}")

        # 순열 대조 요약 — Wilcoxon이 검정력에 막혀도 이쪽은 살아 있다.
        # 순열 p의 하한은 1/(n_perm+1)이라 표본 수가 아니라 반복 수로 정해진다.
        pp = [r for r in rows if r["group"] == "환자" and r["perm_p"] < 0.05]
        ph = [r for r in rows if r["group"] == "정상" and r["perm_p"] < 0.05]
        print(f"  순열 대조(p<0.05): 환자 {len(pp)}/{n_ch}채널, "
              f"정상 {len(ph)}/{n_ch}채널   (하한 {1/(args.n_perm+1):.4f})")

        if underpowered:
            print("  → Wilcoxon은 검정력에 막혔다. **순열 대조와 효과 크기로 읽을 것.**")
            if pp and not ph:
                print("     순열 기준으로는 환자군에서만 유의하다. ⑤ 성립 방향.")
        elif pat and not hea:
            print("  → 환자군에서만 유의. ⑤ 성립 방향.")
        elif pat and hea:
            print("  → 양군 모두 유의. '떨림'이 아니라 '에너지가 큰 구간'을 "
                  "봤을 가능성. 상대 파워를 썼으므로 진폭 효과는 아니지만, "
                  "군 간 효과크기 차이를 함께 보고할 것.")
        else:
            print("  → 환자군에서 유의하지 않음. ⑤ 불성립. "
                  "구간 선택 방식이나 대역을 재검토할 것.")


if __name__ == "__main__":
    main()
