"""공유 인코더 실험 집계 — 단독 vs 공유를 모달별·seed별로 맞대어 판정한다.

읽는 파일 (전부 train_shared.py 산출물이라 학습 조건이 동일하다)
    results/ext_hw_s{42,1,7}.csv       필기 단독
    results/ext_voice_s{42,1,7}.csv    음성 단독   (s42는 voice_baseline_d64L6_s42.csv)
    results/ext_2modal_s{42,1,7}.csv   2모달 공유  (s42는 shared_2modal_d64L6_s42.csv)

⚠️ 판정 기준에 관한 메모
    seed가 3개뿐이라 t 검정의 검정력이 매우 낮다. 그래서 t값 하나로 결론내지 않고
    아래 셋을 함께 본다.
      · 부호 일관성 — seed 3개에서 모두 같은 방향인가
      · 효과 크기 vs **공유 모델 자체의 seed 변동폭** (단독의 변동폭이 아니다)
      · 지표 간 일관성 — AUC와 피험자 정확도가 같은 말을 하는가

    특히 두 번째가 중요하다. 단독의 seed 변동폭이 작다는 것만으로 개선을 주장하면
    공유 쪽 불안정성을 놓친다 (실제로 음성에서 단독 0.001 vs 공유 0.017이었다).

실행: python analyze_shared.py
"""
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

R = Path(__file__).resolve().parent / "results"
SEEDS = (42, 1, 7)

# 같은 실행인데 파일명이 다른 것들(초기 실행분)을 함께 잡는다
PATHS = {
    ("hw", "solo"):     lambda s: [f"ext_hw_s{s}.csv"],
    ("voice", "solo"):  lambda s: [f"ext_voice_s{s}.csv", "voice_baseline_d64L6_s42.csv"]
                                  if s == 42 else [f"ext_voice_s{s}.csv"],
    ("hw", "share"):    lambda s: [f"ext_2modal_s{s}.csv", "shared_2modal_d64L6_s42.csv"]
                                  if s == 42 else [f"ext_2modal_s{s}.csv"],
    ("voice", "share"): lambda s: [f"ext_2modal_s{s}.csv", "shared_2modal_d64L6_s42.csv"]
                                  if s == 42 else [f"ext_2modal_s{s}.csv"],
}
METRICS = [("roc_auc", "ROC-AUC"), ("subject_acc", "피험자 정확도"),
           ("window_acc", "윈도우 정확도")]


def read(fname, modality):
    """CSV에서 해당 모달 행을 찾아 dict로. 없으면 None."""
    p = R / fname
    if not p.exists():
        return None
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    cols = lines[0].split(",")
    for line in lines[1:]:
        row = dict(zip(cols, line.split(",")))
        if row.get("modality") != modality:
            continue
        out = {}
        for k, v in row.items():          # 숫자로 읽히는 것만 변환 (pair_mode='longest' 등 제외)
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        return out
    return None


def collect(modality, kind):
    """{seed: row}"""
    out = {}
    for s in SEEDS:
        for fname in PATHS[(modality, kind)](s):
            r = read(fname, modality)
            if r:
                out[s] = r
                break
    return out


def report(modality):
    solo, share = collect(modality, "solo"), collect(modality, "share")
    seeds = [s for s in SEEDS if s in solo and s in share]
    print("\n" + "=" * 72)
    print(f"  {modality.upper()}   단독 seed {sorted(solo)} · 공유 seed {sorted(share)}"
          f"  → 맞대볼 수 있는 seed {seeds}")
    print("=" * 72)
    if not seeds:
        print("  비교 가능한 seed가 없습니다.")
        return

    for key, label in METRICS:
        a = np.array([solo[s][key] for s in seeds])
        b = np.array([share[s][key] for s in seeds])
        d = b - a
        print(f"\n  [{label}]")
        print(f"    {'seed':>6}{'단독':>10}{'공유':>10}{'차이':>10}")
        for s, x, y in zip(seeds, a, b):
            print(f"    {s:>6}{x:>10.4f}{y:>10.4f}{y - x:>+10.4f}")
        print(f"    {'평균':>6}{a.mean():>10.4f}{b.mean():>10.4f}{d.mean():>+10.4f}")

        # 판정 재료 셋
        signs = "일관(+)" if (d > 0).all() else "일관(−)" if (d < 0).all() else \
                f"엇갈림 ({int((d > 0).sum())}+/{int((d < 0).sum())}−)"
        solo_sp = a.max() - a.min()
        share_sp = b.max() - b.min()
        print(f"    부호 {signs} · seed 변동폭 단독 {solo_sp:.4f} / **공유 {share_sp:.4f}**")

        if len(seeds) >= 2:
            se = d.std(ddof=1) / np.sqrt(len(d))
            t = d.mean() / se if se > 0 else float("inf")
            verdict = ("효과 있음" if abs(t) >= 2 else
                       "판단 보류" if abs(t) >= 1 else "노이즈에 묻힘")
            print(f"    seed 간 t = {t:+.2f} (n={len(d)}) ⇒ {verdict}")
            if d.mean() != 0 and share_sp >= abs(d.mean()):
                print(f"    ⚠️ 공유 변동폭({share_sp:.4f})이 평균 개선폭"
                      f"({abs(d.mean()):.4f}) 이상 — 개선을 주장할 수 없다")


print("=" * 72)
print("공유 인코더 실험 집계 — 단독 vs 공유 (d64/L6, 5-fold, train_shared.py 동일 조건)")
print("=" * 72)
for m in ("voice", "hw"):
    report(m)

print("\n" + "=" * 72)
print("읽는 법")
print("=" * 72)
print("""  · 부호 일관성 + t ≥ 2 + 공유 변동폭 < 개선폭  → 셋 다 만족해야 '효과 있음'
  · seed 3개는 t 검정 검정력이 낮다. t < 2 라도 '효과 없음'이 아니라 '판정 불가'다.
  · 피험자 정확도는 필기에서 1명 = 8.3%p(fold당 12명)라 지표로 쓰기 어렵다.
    음성은 fold당 47명이라 1명 = 2.1%p로 상대적으로 안정적이다.""")
