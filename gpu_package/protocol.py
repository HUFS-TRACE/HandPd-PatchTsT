"""고정 프로토콜 — 세 사람이 같은 조건으로 돌리기 위한 단일 출처.

`train.py`와 `evaluate.py`가 인자를 따로 정의하던 탓에 한쪽에만 있는 플래그가
있었다(`--stratified`, `--val-size`, `--auto-class-weight`, `--label-smoothing`).
같은 이름의 인자가 스크립트마다 다르게 동작하면 세 사람 결과를 나란히 놓을 수
없으므로, 인자 정의와 기본값을 여기 한 곳에 모은다.

사용법
    from protocol import add_common_args, apply_config_file, filter_subjects

설정 파일로 덮어쓰기
    python evaluate.py --config configs/protocol.yaml --suite depth
    (명령줄에서 직접 준 인자가 설정 파일보다 우선한다)
"""
import os
import sys

import numpy as np

# ──────────────────────────── 고정 프로토콜 ────────────────────────────
# 근거는 docs/protocol_rationale.md 및 가이드 08_공통규칙.md.
# 변경은 팀 결정을 거친다. 여기 값이 바뀌면 이전 결과와 비교가 깨진다.

DEFAULTS = dict(
    data_path="data/windows_2s.npz",   # 창 길이는 2초로 고정 (1s/4s는 쓰지 않는다)
    epochs=50,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    patience=10,
    seed=42,
    n_folds=5,
    val_size=0.1,
    stratified=False,
    auto_class_weight=False,
    label_smoothing=0.0,
    class_weight_healthy=2.0,
    class_weight_patient=1.0,
    dropout=0.2,
    head_dropout=0.2,
)

# 시드는 전원 이 세 개만 쓴다. 전체·서브셋·경량ML 구분 없이 동일하다.
#   시드가 fold 분할을 바꾸지 않으므로(GroupKFold는 shuffle 안 함) 시드를 늘려도
#   독립 단위는 fold 수 그대로다. 늘려서 얻는 건 학습 반복뿐이라, 사람마다
#   다른 시드를 쓸 위험을 없애는 쪽이 이득이 크다.
SEEDS = [42, 1, 7]
SEEDS_FULL = SEEDS       # 이전 이름 호환
SEEDS_SUBSET = SEEDS

# ──────────────────────────── 기준 모델 ────────────────────────────
# 하나의 "경량 대표"로 두 가지 역할을 겸하게 하면 ⑥단계의 결론이 왜곡된다.
# 깊이만 줄인 모델과 깊이·패치·폭을 모두 줄인 모델은 연산량이 15배 차이나므로
# 역할을 나눠 이름을 따로 붙인다.
#
#   REFERENCE      비교의 원점. TOST가 전부 `X vs 이것` 구조이고,
#                  R군이 검증하는 "불필요하다고 주장하는 깊은 모델"이다.
#   DEPTH_LIGHT    깊이 축 전용 경량 대표. 백본이 REFERENCE와 완전히 같고
#                  층 수만 1이라, 깊이의 효과만 분리해 보여준다.
#   STATIC_LIGHT   정적 축의 실제 도달점. 깊이·패치·폭을 함께 줄인 결과이며
#                  ⑥의 파레토 그림에서 정적 축을 대표하는 점이다.
#
# 3시드 실측 (windows_2s, 5-fold, 주 지표 = 윈도우 AUC):
#   p16d128L6   3,499 MFLOPs   AUC .8629   피험자 .8790
#   p16d128L1     588 MFLOPs   AUC .8664   피험자 .8846
#   d64L2          38 MFLOPs   AUC .8680   피험자 .8534
REFERENCE = "p16d128L6"
DEPTH_LIGHT = "p16d128L1"
STATIC_LIGHT = "d64L2"

# TOST 등가 마진 (윈도우 AUC). 실험 전에 확정·기록한다.
#   ① 시드 간 재현 오차(±0.011)의 약 2배
#   ② 61명 규모에서 임상적으로 무시 가능한 차이
# 전체 데이터 전용. 서브셋(R4·R5)은 표본이 작아 분산이 커서 재사용하지 않는다.
TOST_MARGIN_AUC = 0.02


def add_common_args(parser, *, include_model_args=False):
    """train.py와 evaluate.py가 공유하는 인자.

    include_model_args=True면 단일 모델 구조 인자(train.py용)도 붙인다.
    evaluate.py는 구조를 CONFIGS 목록에서 가져오므로 필요 없다.
    """
    d = DEFAULTS

    g = parser.add_argument_group("데이터")
    g.add_argument("--data-path", default=d["data_path"])
    g.add_argument("--subjects-file", default=None,
                   help="이 파일에 적힌 피험자만 사용한다 (한 줄에 하나). "
                        "R4·R5 서브셋 실험용. 예: data/subset_R4.txt")

    g = parser.add_argument_group("학습")
    g.add_argument("--epochs", type=int, default=d["epochs"])
    g.add_argument("--batch-size", type=int, default=d["batch_size"])
    g.add_argument("--lr", type=float, default=d["lr"])
    g.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    g.add_argument("--patience", type=int, default=d["patience"])
    g.add_argument("--seed", type=int, default=d["seed"])
    g.add_argument("--n-folds", type=int, default=d["n_folds"])
    g.add_argument("--val-size", type=float, default=d["val_size"],
                   help="train_val에서 val로 떼는 비율")
    g.add_argument("--stratified", action="store_true",
                   help="StratifiedGroupKFold 사용 (기본은 GroupKFold)")
    g.add_argument("--label-smoothing", type=float, default=d["label_smoothing"])
    g.add_argument("--dropout", type=float, default=d["dropout"])
    g.add_argument("--head-dropout", type=float, default=d["head_dropout"])

    g = parser.add_argument_group("클래스 가중 (정상:환자)")
    g.add_argument("--class-weight-healthy", type=float, default=d["class_weight_healthy"])
    g.add_argument("--class-weight-patient", type=float, default=d["class_weight_patient"])
    g.add_argument("--auto-class-weight", action="store_true",
                   help="fold의 train 분포에서 역빈도로 가중을 다시 계산한다. "
                        "서브셋은 클래스 비율이 뒤집히므로 이쪽이 주 실험이다.")

    if include_model_args:
        g = parser.add_argument_group("모델")
        g.add_argument("--patch-len", type=int, default=16)
        g.add_argument("--stride", type=int, default=8)
        g.add_argument("--d-model", type=int, default=128)
        g.add_argument("--n-heads", type=int, default=8)
        # 참조 모델과 같은 6층을 기본값으로 둔다.
        # 이전 기본값이 3이었는데, L3는 세 지표 모두에서 가장 낮아
        # 아무 인자 없이 실행하면 하필 가장 나쁜 설정이 돌아갔다.
        g.add_argument("--n-layers", type=int, default=6)
        g.add_argument("--d-ff", type=int, default=256)

    parser.add_argument("--config", default=None,
                        help="YAML/JSON 설정 파일. 명령줄 인자가 이보다 우선한다.")
    return parser


def apply_config_file(args, argv=None):
    """설정 파일 값을 args에 적용한다. 명령줄에서 직접 준 인자는 건드리지 않는다.

    argparse는 "기본값을 그대로 쓴 것"과 "기본값과 같은 값을 명시한 것"을
    구분하지 못하므로, sys.argv 문자열을 보고 명시 여부를 판단한다.
    """
    if not getattr(args, "config", None):
        return args

    import json
    path = args.config
    with open(path, encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            cfg = yaml.safe_load(f) or {}
        else:
            cfg = json.load(f)

    # 중첩 dict(data:/학습: 같은 그룹)도 평평하게 편다.
    flat = {}
    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            else:
                flat[k.replace("-", "_")] = v
    walk(cfg)

    argv = sys.argv[1:] if argv is None else argv
    given = {a.split("=")[0].lstrip("-").replace("-", "_")
             for a in argv if a.startswith("--")}

    applied = []
    for k, v in flat.items():
        if not hasattr(args, k):
            continue          # 이 스크립트와 무관한 항목은 조용히 넘긴다
        if k in given:
            continue          # 명령줄이 이긴다
        setattr(args, k, v)
        applied.append(f"{k}={v}")

    if applied:
        print(f"[config] {path} 적용: {', '.join(applied)}")
    return args


def filter_subjects(X, y, subject_id, task, subjects_file):
    """명단에 있는 피험자의 윈도우만 남긴다. R4·R5 서브셋 실험의 입구.

    명단에 없는 이름이 있으면 조용히 무시하지 않고 멈춘다. 오타 하나로
    서브셋이 달라지면 결과를 나란히 놓을 수 없기 때문이다.
    """
    if not subjects_file:
        return X, y, subject_id, task

    with open(subjects_file, encoding="utf-8") as f:
        want = {line.strip() for line in f if line.strip()}

    have = set(subject_id.tolist())
    missing = want - have
    if missing:
        sys.exit(f"명단에 있으나 데이터에 없는 피험자 {len(missing)}명: "
                 f"{sorted(missing)[:5]}{' …' if len(missing) > 5 else ''}\n"
                 f"  ({subjects_file} 의 표기가 npz의 subject_id와 같은지 확인하세요)")

    mask = np.isin(subject_id, list(want))
    n_before, n_after = len(y), int(mask.sum())
    Xf, yf, sf, tf = X[mask], y[mask], subject_id[mask], task[mask]

    print(f"[subset] {subjects_file}: 피험자 {len(have)} → {len(want)}명, "
          f"윈도우 {n_before} → {n_after}개")
    print(f"         정상 {(yf==0).sum()} / 환자 {(yf==1).sum()} 윈도우, "
          f"피험자 정상 {sum(1 for s in want if yf[sf==s][0]==0)} / "
          f"환자 {sum(1 for s in want if yf[sf==s][0]==1)}명")
    print(f"         [주의] 서브셋은 클래스 비율이 전체와 다르다. "
          f"--auto-class-weight 를 주 실험으로 삼고, 고정 가중으로도 한 번 돌려 "
          f"결론이 뒤집히지 않는지 확인할 것.")
    return Xf, yf, sf, tf


def get_folds(subject_id, y, args):
    """고정 프로토콜에 따른 fold 분할. **모든 실험이 이 함수를 거친다.**

    딥러닝(evaluate.py)과 경량 ML(E1·E2·E3)이 서로 다른 분할을 쓰면 성능을
    나란히 놓을 수 없다. sklearn 쪽에서도 이 함수로 인덱스를 받아 쓸 것.

        from protocol import get_folds
        for tr, va, te in get_folds(subject_id, y, args):
            clf.fit(F[tr], y[tr]); ...

    args 없이 쓰려면 SimpleNamespace(n_folds=5, val_size=0.1, stratified=False,
    seed=42)를 만들어 넘기면 된다.
    """
    from dataset import subject_kfold
    return subject_kfold(subject_id, y,
                         n_splits=args.n_folds,
                         val_size=args.val_size,
                         seed=args.seed,
                         stratified=args.stratified)


def describe(args):
    """실행 조건 한 줄 요약. 로그와 결과 파일 양쪽에 남겨 조건 불일치를 눈에 띄게 한다."""
    return (f"data={os.path.basename(args.data_path)} "
            f"subset={os.path.basename(args.subjects_file) if getattr(args, 'subjects_file', None) else '-'} "
            f"folds={args.n_folds} val={args.val_size} "
            f"strat={int(args.stratified)} seed={args.seed} "
            f"cw={'auto' if args.auto_class_weight else f'{args.class_weight_healthy}/{args.class_weight_patient}'} "
            f"ls={args.label_smoothing} ep={args.epochs} pat={args.patience} "
            f"bs={args.batch_size} lr={args.lr}")


def check_protocol(args):
    """고정 프로토콜에서 벗어난 항목을 실행 시작할 때 크게 알린다.

    조건이 어긋난 채로 몇 시간을 돌리고 나서야 알아차리는 일을 막는 장치다.
    벗어나는 것 자체는 막지 않는다 — 스윕은 일부러 벗어나야 하니까.
    """
    print(f"[protocol] {describe(args)}")

    # 실험마다 정당하게 달라지는 축은 경고하지 않는다.
    skip = {"seed", "n_folds", "data_path"}
    if getattr(args, "subjects_file", None) and args.auto_class_weight:
        # 서브셋은 클래스 비율이 뒤집히므로 가중 재계산이 주 실험이다 (08 §6-②)
        skip.add("auto_class_weight")

    off = []
    for k, v in DEFAULTS.items():
        if k in skip:
            continue
        if hasattr(args, k) and getattr(args, k) != v:
            off.append(f"{k}: {v} -> {getattr(args, k)}")
    if args.seed not in SEEDS:
        off.append(f"seed: {SEEDS} 밖의 값 {args.seed}")
    if args.n_folds not in (3, 5):
        off.append(f"n_folds: 3(서브셋)/5(전체)가 아닌 {args.n_folds}")

    if off:
        print("[protocol] 고정 프로토콜과 다른 항목:")
        for line in off:
            print(f"             - {line}")
        print("           의도한 것이면 그대로 진행하세요. "
              "아니면 --config configs/protocol.yaml 로 되돌립니다.")
    return args


def run_tag(args, extra=None):
    """결과 파일 이름에 붙일 실행 조건 태그. 조건이 다른 실행이 서로 덮어쓰지 않게 한다."""
    parts = []
    ds = os.path.basename(args.data_path).replace("windows", "").replace(".npz", "")
    if ds.strip("_"):
        parts.append(ds.strip("_"))
    if getattr(args, "subjects_file", None):
        parts.append(os.path.basename(args.subjects_file)
                     .replace("subset_", "").replace(".txt", ""))
    if args.lr != DEFAULTS["lr"]:
        parts.append(f"lr{args.lr:g}")
    if args.batch_size != DEFAULTS["batch_size"]:
        parts.append(f"bs{args.batch_size}")
    if args.n_folds != DEFAULTS["n_folds"]:
        parts.append(f"fold{args.n_folds}")
    if args.patience != DEFAULTS["patience"]:
        parts.append(f"pat{args.patience}")
    if args.stratified:
        parts.append("strat")
    if args.val_size != DEFAULTS["val_size"]:
        parts.append(f"val{args.val_size:g}")
    if args.auto_class_weight:
        parts.append("autocw")
    if args.label_smoothing != DEFAULTS["label_smoothing"]:
        parts.append(f"ls{args.label_smoothing:g}")
    if extra:
        parts = [p for p in extra if p] + parts
    if args.seed != DEFAULTS["seed"]:
        parts.append(f"s{args.seed}")
    return "_".join(parts)
