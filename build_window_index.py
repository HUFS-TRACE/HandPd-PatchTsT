"""windows_2s.npz 에 run_id · window_index 를 사후에 붙인다 (P0).

왜
    R2(임시연)와 시간 축 분석은 "각 창이 어느 기록의 몇 번째인가"가 필요한데
    npz 에 그 정보가 없다. 시연이 요청한 windows_2s_index.npz 를 만든다.

⚠️ 기존 npz 를 다시 만들지 않는다
    재생성하면 지금까지의 결과가 전부 무효가 된다. 원본에서 후보 창을 만들어
    **내용으로 대조**해 인덱스만 별도 파일로 저장한다. 행 순서는 npz 그대로다.

전처리 규칙 (New-Hand-PD/preprocess.py 에서 확인)
    ① 헤더(#</meta>)까지 건너뛰고 본문만 읽는다
    ② 앞 2행 drop — 센서 초기화 글리치. 이것이 재현 시 0.6% 차이의 원인이었다
    ③ 2000샘플 창 · 1000 스트라이드 (50% overlap)
    ④ 1초 미만 파일은 제외 (실제로는 0개)

    파일 순회 순서가 glob 에 의존하므로 순서를 가정하지 않고 값으로 맞춘다.

실행
    python build_window_index.py
    python build_window_index.py --win 2000 --hop 1000 --out data/windows_2s_index.npz
"""
import argparse
import glob
import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GLITCH_ROWS = 2
SIGNAL_DIRS = ["data/HealthySignal", "data/PatientSignal"]

# Object 코드 → 과제명. 환자군 나선은 c_spiral 로 오타나 있다 (원본 그대로).
TASK_MAP = {
    "a_circ_p": "circle", "b_circ_a": "circle",
    "c_spira": "spiral", "c_spiral": "spiral",
    "d_mea": "meander",
    "e_diador": "diadochokinesis", "f_diadol": "diadochokinesis",
}


def parse_file(path):
    """(meta, body[T,6]). 앞 GLITCH_ROWS 행 제거 완료."""
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    meta_end = next(i for i, l in enumerate(lines) if l.strip() == "#</meta>")
    meta = {}
    for l in lines[:meta_end]:
        m = re.match(r"#<(\w+)>(.*)</\w+>", l.strip())
        if m:
            meta[m.group(1)] = m.group(2)
    rows = []
    for l in lines[meta_end + 1:]:
        l = l.strip()
        if l:
            rows.append([float(v) for v in l.split()])
    body = np.asarray(rows, dtype=np.float32)[GLITCH_ROWS:]
    return meta, body


def fingerprint(win):
    """창 하나의 지문 — 전체 내용 해시.

    처음에는 값 몇 개만 뽑아 썼는데 충돌이 1,439건 났다. 신호에 평탄한 구간이
    많아(펜을 뗀 동안) 서로 다른 창의 앞머리가 같은 값이었다. 그 결과 731 run 만
    복원되고(792여야 함) 연속성이 49 run 에서 깨졌다.

    전체를 해시하면 비트가 같은 창만 충돌한다. C-contiguous 로 맞춰야
    같은 내용이 같은 바이트가 된다.
    """
    return hashlib.md5(np.ascontiguousarray(win).tobytes()).digest()


def main():
    ap = argparse.ArgumentParser(description="창 인덱스 사후 부여")
    ap.add_argument("--npz", default="data/windows_2s.npz")
    ap.add_argument("--win", type=int, default=2000)
    ap.add_argument("--hop", type=int, default=1000)
    ap.add_argument("--out", default="data/windows_2s_index.npz")
    a = ap.parse_args()

    z = np.load(ROOT / a.npz, allow_pickle=True)
    X = z["X"]
    print(f"대상 {a.npz}  X{X.shape}")

    # ── 원본에서 후보 창을 만들고 지문 → (run_id, window_index) 사전 ──
    files = []
    for d in SIGNAL_DIRS:
        files += sorted(glob.glob(str(ROOT / d / "*.txt")))
    print(f"원본 파일 {len(files)}개 파싱 중...")

    # ⚠️ 전역 해시로 맞추면 안 된다. 펜을 뗀 평탄 구간이 많아 서로 다른 기록의
    #    창이 비트까지 같은 경우가 1,439건 있다. 그러면 나중 파일이 앞 파일을
    #    덮어써 61개 run 이 통째로 사라진다(731/792).
    #    같은 (피험자, 과제) 안에서만 맞추고, 남은 동률은 파일 순서대로 소비한다.
    from collections import defaultdict, deque
    cand = defaultdict(lambda: defaultdict(deque))   # (subj, task) → 해시 → 큐
    n_cand = 0
    for i, p in enumerate(files):
        if i % 200 == 0 and i:
            print(f"  {i}/{len(files)}")
        meta, body = parse_file(p)
        grp = "P" if "Patient" in p else "H"
        key = (f"{grp}_{meta.get('Person_ID_Number')}",
               TASK_MAP.get(meta.get("Object"), meta.get("Object")))
        run = os.path.basename(p)
        for wi, s in enumerate(range(0, len(body) - a.win + 1, a.hop)):
            cand[key][fingerprint(body[s:s + a.win].T)].append((run, wi))
            n_cand += 1
    print(f"후보 창 {n_cand:,}개 · (피험자,과제) 조합 {len(cand)}개")

    # ── npz 각 행을 같은 (피험자, 과제) 안에서 조회 ──
    sid_all, task_all = z["subject_id"], z["task"]
    run_id = np.empty(len(X), dtype=object)
    win_ix = np.full(len(X), -1, dtype=np.int32)
    miss = 0
    for i in range(len(X)):
        q = cand[(str(sid_all[i]), str(task_all[i]))].get(fingerprint(X[i]))
        if not q:
            miss += 1
        else:
            run_id[i], win_ix[i] = q.popleft()      # 한 번 쓰면 소비한다

    print(f"\n매칭 {len(X) - miss:,} / {len(X):,}  (실패 {miss})")
    if miss:
        print("⚠️ 매칭 실패가 있다. 전처리 규칙이 어긋났을 수 있다.")
        sys.exit(1)

    # ── 검증 ──
    runs = np.unique(run_id)
    print(f"복원된 run {len(runs)}개 (원본 파일 {len(files)}개)")
    per = [int((run_id == r).sum()) for r in runs]
    print(f"run 당 창  중앙 {int(np.median(per))}  범위 {min(per)}~{max(per)}")
    # window_index 가 run 마다 0부터 연속인지
    bad = [r for r in runs
           if not np.array_equal(np.sort(win_ix[run_id == r]),
                                 np.arange((run_id == r).sum()))]
    print(f"window_index 연속성 이상 run {len(bad)}개")

    # 같은 사람의 창이 여러 run 에 걸치는 구조 확인
    sid = z["subject_id"]
    rp = {}
    for s in np.unique(sid):
        rp[s] = len(np.unique(run_id[sid == s]))
    v = np.array(list(rp.values()))
    print(f"피험자당 run 수  중앙 {int(np.median(v))}  범위 {v.min()}~{v.max()}")

    np.savez_compressed(ROOT / a.out, run_id=run_id.astype("U32"),
                        window_index=win_ix)
    sz = os.path.getsize(ROOT / a.out) / 1e6
    print(f"\n저장: {a.out}  ({sz:.1f} MB)")
    print("행 순서는 windows_2s.npz 와 동일하다. 기존 npz 는 건드리지 않았다.")


if __name__ == "__main__":
    main()
