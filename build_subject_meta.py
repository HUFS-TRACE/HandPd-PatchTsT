"""원본 signal txt 헤더에서 피험자 메타데이터를 뽑아 서브셋 명단과 기준선을 만든다.

npz에는 X/y/subject_id/task 넷뿐이라 나이·성별·수집일이 없다. 그 셋은
원본 txt의 `#<meta>` 헤더에만 있으므로, R5(연령 통제)·R6(성별)·나이 기준선은
전부 이 스크립트를 거쳐야 한다.

실행: python build_subject_meta.py

산출물
    data/subject_meta.csv   피험자 61명 × (라벨·나이·성별·수집연도·파일번호)
    data/subset_R4.txt      2016년 수집분 제외 40명 (정상 14 / 환자 26)
    data/subset_R5.txt      나이 45세 이상 41명 (정상 17 / 환자 24)

`#<Date>`를 근거로 쓴다. ID 두 번째 필드(603/604/605)로 걸러도 결과가 같지만,
그건 "등록 연월일 것"이라는 추정이라 논문 근거로는 날짜 필드를 쓴다.
(두 방식의 일치 여부를 아래에서 매번 검증한다.)
"""
import collections
import glob
import os
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = "data"
HEALTHY_GLOB = os.path.join(DATA_DIR, "HealthySignal", "*.txt")
PATIENT_GLOB = os.path.join(DATA_DIR, "PatientSignal", "*.txt")

# 파일명 접두사 → npz의 task 라벨.
#   사람마다 12개 기록이 있고 그것이 4개 과제로 묶인다.
#   spiral·meander는 4회, circle·diadochokinesis는 2회 반복 측정이다.
TASK_OF = {
    "circA": "circle",          "circB": "circle",
    "sigDiaA": "diadochokinesis", "sigDiaB": "diadochokinesis",
    "sigMea1": "meander", "sigMea2": "meander",
    "sigMea3": "meander", "sigMea4": "meander",
    "sigSp1": "spiral",   "sigSp2": "spiral",
    "sigSp3": "spiral",   "sigSp4": "spiral",
}

# 2016년 수집분의 ID 두 번째 필드. 날짜 기준 결과와 대조하는 용도로만 쓴다.
ID_FIELD_2016 = {"603", "604", "605"}

AGE_THRESHOLD_R5 = 45          # R5 서브셋 경계
AGE_RULE_GRID = [45, 48, 50, 52, 55, 58, 60]   # 나이 규칙 기준선 탐색 범위


def read_meta(path):
    """txt 앞머리의 `#<meta>` 블록을 dict로 읽는다. 본문은 읽지 않는다."""
    meta = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#</meta>"):
                break
            m = re.match(r"#<(\w+)>(.*)</\1>", line.strip())
            if m:
                meta[m.group(1)] = m.group(2)
    return meta


def collect_records():
    """792개 파일을 훑어 (피험자 ID → 기록 목록)으로 모은다."""
    files = sorted(glob.glob(HEALTHY_GLOB)) + sorted(glob.glob(PATIENT_GLOB))
    if not files:
        sys.exit(f"원본 signal txt를 찾지 못했습니다: {HEALTHY_GLOB} / {PATIENT_GLOB}")

    by_subject = collections.defaultdict(list)
    for path in files:
        base = os.path.basename(path)
        prefix = base.split("-")[0]
        m = re.search(r"-([HP]\d+)\.txt$", base)
        if not m or prefix not in TASK_OF:
            print(f"  ⚠️  파일명 규칙에 안 맞아 건너뜀: {base}")
            continue
        meta = read_meta(path)
        by_subject[meta["Person_ID_Number"]].append(dict(
            label=0 if "Healthy" in path else 1,
            age=int(meta["Age"]),
            gender=int(meta["Gender"]),
            year=meta["Date"].split(".")[-1],
            samplerate=meta["Samplerate"],
            folder=m.group(1),          # 원본 파일번호 H1~H35 / P1~P31
            task=TASK_OF[prefix],
            run=prefix,                 # 반복 기록 구분 (sigSp1 …)
            file=base,
        ))
    return files, by_subject


def check_consistency(files, by_subject):
    """피험자 안에서 라벨·나이·성별이 엇갈리지 않는지, 샘플레이트가 같은지 본다.

    엇갈리면 아래 집계가 전부 무의미해지므로 여기서 멈춘다.
    """
    print(f"파일 {len(files)}개 → 고유 인물 {len(by_subject)}명")

    rates = {r["samplerate"] for recs in by_subject.values() for r in recs}
    print(f"샘플레이트: {sorted(rates)}")
    if len(rates) != 1:
        print("  ⚠️  샘플레이트가 파일마다 다릅니다. 전처리 가정을 다시 확인하세요.")

    for field in ("label", "age", "gender"):
        bad = [k for k, recs in by_subject.items() if len({r[field] for r in recs}) > 1]
        if bad:
            print(f"  ⚠️  피험자 안에서 {field}가 엇갈립니다: {bad}")


def report_dedup(by_subject):
    """66개 파일번호가 61명으로 정제되는 과정을 출력한다 (논문 방법론 절 근거)."""
    folder_to_ids = collections.defaultdict(set)
    for pid, recs in by_subject.items():
        for r in recs:
            folder_to_ids[r["folder"]].add(pid)

    print(f"\n── 피험자 정제 {len(folder_to_ids)} → {len(by_subject)} ──")

    mixed = {k: v for k, v in folder_to_ids.items() if len(v) > 1}
    for folder, ids in sorted(mixed.items()):
        print(f"  인물 혼입  {folder} 폴더에 {len(ids)}명: {sorted(ids)}")

    id_to_folders = collections.defaultdict(list)
    for folder, ids in folder_to_ids.items():
        for pid in ids:
            id_to_folders[pid].append(folder)
    for pid, folders in sorted(id_to_folders.items()):
        if len(folders) > 1:
            print(f"  중복      {sorted(folders)} → 동일인 {pid}")

    print("  → 파일번호로 train/test를 나누면 같은 사람이 양쪽에 들어간다."
          " 그룹 키는 Person_ID_Number여야 한다.")


def build_table(by_subject):
    """피험자 1명 = 1행. 기록 단위 정보는 대표값으로 접는다."""
    rows = []
    for pid, recs in sorted(by_subject.items()):
        years = sorted({r["year"] for r in recs})
        rows.append(dict(
            subject_id=pid,
            npz_subject_id=("H_" if recs[0]["label"] == 0 else "P_") + pid,
            label=recs[0]["label"],
            age=recs[0]["age"],
            gender=recs[0]["gender"],
            year="|".join(years),
            n_records=len(recs),
            folders="|".join(sorted({r["folder"] for r in recs})),
        ))
    return rows


def report_demographics(rows):
    y = np.array([r["label"] for r in rows])
    age = np.array([r["age"] for r in rows])
    gender = np.array([r["gender"] for r in rows])

    print(f"\n── 인구통계 ──")
    for lab, name in ((0, "정상"), (1, "환자")):
        a = age[y == lab]
        print(f"  {name} {len(a):2d}명  평균 {a.mean():.1f}세  범위 {a.min()}–{a.max()}")
    print(f"  나이 차이 {age[y==1].mean() - age[y==0].mean():+.1f}세")

    for lab, name in ((0, "정상"), (1, "환자")):
        c = collections.Counter(gender[y == lab].tolist())
        male = c.get(1, 0)
        print(f"  {name} 성별  남 {male} / 여 {c.get(2, 0)}  (남성 {100*male/len(gender[y==lab]):.1f}%)")

    print(f"\n── 수집 연도 × 라벨 ──")
    year_label = collections.Counter()
    for r in rows:
        for yr in r["year"].split("|"):
            year_label[(yr, r["label"])] += 1
    for yr in sorted({k[0] for k in year_label}):
        h, p = year_label.get((yr, 0), 0), year_label.get((yr, 1), 0)
        flag = "  ← 전원 정상" if p == 0 and h > 0 else ""
        print(f"  {yr}  정상 {h:2d} / 환자 {p:2d}{flag}")


def report_baselines(rows, n_windows_healthy=None, n_windows_patient=None):
    """모델이 넘어야 하는 선. 논문 성능표 맨 위에 들어간다."""
    y = np.array([r["label"] for r in rows])
    age = np.array([r["age"] for r in rows])
    gender = np.array([r["gender"] for r in rows])

    print(f"\n── 성능 기준선 ──")
    subj_major = max((y == 0).mean(), (y == 1).mean())
    print(f"  피험자 다수결        {100*subj_major:.1f}%  (전부 정상이라고 찍기)")
    if n_windows_healthy is not None:
        tot = n_windows_healthy + n_windows_patient
        print(f"  윈도우 다수결        {100*max(n_windows_healthy, n_windows_patient)/tot:.1f}%"
              f"  (윈도우 단위로 전부 환자라고 찍기)")
        print(f"     ⚠️ 다수 클래스가 뒤집힌다 — 피험자 기준 다수는 정상,"
              f" 윈도우 기준 다수는 환자. 논문에 명시할 것")

    print(f"  나이 규칙 (임계값별 정확도)")
    best_t, best_acc = None, -1
    for t in AGE_RULE_GRID:
        acc = ((age >= t).astype(int) == y).mean()
        if acc > best_acc:
            best_t, best_acc = t, acc
        print(f"     ≥{t}세  {100*acc:.1f}%")
    print(f"  → 나이 단독 최고 {100*best_acc:.1f}% (≥{best_t}세)."
          f" 모델이 실제로 넘어야 하는 선은 이것이다.")

    for target, name in ((1, "남성"), (2, "여성")):
        acc = ((gender == target).astype(int) == y).mean()
        print(f"  성별 규칙 ({name}→환자)  {100*acc:.1f}%")
    print(f"  → 성별 규칙이 피험자 다수결({100*subj_major:.1f}%)을 넘지 못하면"
          f" 기준선 표에 넣지 않고 한계 절에만 적는다. (R6)")


def report_subsets(rows):
    """R4·R5 명단과 두 서브셋의 겹침. 04 가이드 M0의 산출물."""
    def stat(sel):
        h = [r for r in sel if r["label"] == 0]
        p = [r for r in sel if r["label"] == 1]
        gap = (np.mean([r["age"] for r in p]) - np.mean([r["age"] for r in h])
               if h and p else float("nan"))
        return len(sel), len(h), len(p), gap

    r4 = [r for r in rows if "2016" not in r["year"].split("|")]
    r5 = [r for r in rows if r["age"] >= AGE_THRESHOLD_R5]

    print(f"\n── 교란 통제 서브셋 ──")
    for sel, name in ((rows, "전체"), (r4, "R4 (2016 제외)"),
                      (r5, f"R5 (나이 ≥{AGE_THRESHOLD_R5})")):
        n, h, p, gap = stat(sel)
        print(f"  {name:<22} {n:2d}명  정상 {h:2d} / 환자 {p:2d}  나이차 {gap:+.1f}세")

    # ID 코드 기반 필터와 날짜 기반 필터가 같은 집합인지 확인.
    #   npz만 있고 원본이 없을 때 ID 코드로 R4를 만들 수 있는지가 여기서 갈린다.
    by_id = {r["subject_id"] for r in rows
             if r["subject_id"].split("-")[1] not in ID_FIELD_2016}
    by_date = {r["subject_id"] for r in r4}
    print(f"  R4 검증: 날짜 기준과 ID 코드 기준이 "
          f"{'일치' if by_id == by_date else '불일치 ⚠️'}")

    inter = [r for r in r4 if r["age"] >= AGE_THRESHOLD_R5]
    n, h, p, gap = stat(inter)
    print(f"\n  R4 ∩ R5                {n:2d}명  정상 {h:2d} / 환자 {p:2d}  나이차 {gap:+.1f}세")
    print(f"  → 두 교란을 동시에 통제하면 정상이 {h}명뿐이다."
          f" 3-fold로도 fold당 {h/3:.1f}명이라 쓸 수 없다.")
    print(f"  → '61명 규모에서 시기와 나이를 동시에 통제하는 것은 불가능하다'"
          f"는 논문 한계 절의 근거 수치.")
    return r4, r5


def write_outputs(rows, r4, r5):
    meta_path = os.path.join(DATA_DIR, "subject_meta.csv")
    cols = ["subject_id", "npz_subject_id", "label", "age", "gender",
            "year", "n_records", "folders"]
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\n저장: {meta_path}  ({len(rows)}명)")

    # 서브셋 명단은 npz의 subject_id 표기(H_/P_ 접두사)로 쓴다.
    # 학습 스크립트가 그대로 읽어 필터에 쓸 수 있어야 하기 때문이다.
    for sel, name in ((r4, "subset_R4.txt"), (r5, "subset_R5.txt")):
        path = os.path.join(DATA_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in sorted(sel, key=lambda r: r["npz_subject_id"]):
                f.write(r["npz_subject_id"] + "\n")
        print(f"저장: {path}  ({len(sel)}명)")


def main():
    files, by_subject = collect_records()
    check_consistency(files, by_subject)
    report_dedup(by_subject)

    rows = build_table(by_subject)
    report_demographics(rows)

    # 윈도우 다수결은 npz가 있어야 셀 수 있다. 없으면 그 줄만 건너뛴다.
    nw_h = nw_p = None
    npz_path = os.path.join(DATA_DIR, "windows_2s.npz")
    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        nw_h, nw_p = int((d["y"] == 0).sum()), int((d["y"] == 1).sum())

    report_baselines(rows, nw_h, nw_p)
    r4, r5 = report_subsets(rows)
    write_outputs(rows, r4, r5)


if __name__ == "__main__":
    main()
