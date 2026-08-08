"""
공유 인코더용 데이터 로딩 — 모달별 npz를 각각 읽고, 학습 때 짝지어 흘린다.

⚠️ 데이터를 물리적으로 합치지 않는다
    필기  X[13474,  6, 2000]   6채널(펜 센서) · 2000 타임스텝
    음성  X[ 2008, 64,  126]  64채널(멜 밴드) · 126 프레임

    · 샘플 축으로 잇기   → 채널 6≠64, 길이 2000≠126이라 배열이 안 맞는다
    · 채널 축으로 잇기   → 같은 사람의 필기+음성이 0명이라 짝지을 수 없다
    · 리샘플해서 맞추기  → 펜 압력을 멜 밴드로 바꾸는 건 의미가 없다

    그래서 "합친 모델"은 데이터가 아니라 **모델 안에서만** 존재한다. 여기서는
    두 로더를 나란히 굴리기만 하고, 합쳐지는 지점은 shared_model.py의 인코더다.

fold는 모달마다 독립적으로 만든다(필기 61명 / 음성 116명). fold i끼리 짝지어
함께 학습하고, 평가는 각자의 test 피험자에 대해 각자의 head로 한다.
"""
import numpy as np
from torch.utils.data import DataLoader

from dataset import WindowDataset, subject_kfold


def load_modality(path):
    """npz → (X, y, subject_id).

    dataset.load_npz와 달리 task를 요구하지 않는다. 음성 npz에는 task 필드가
    없기 때문이다(필기는 과제 12종이 있지만 음성은 발성 과제 구분이 다르다).

    X는 float32로 캐스팅해 둔다. fold 인덱싱(X[tr])이 어차피 복사본을 만드는데,
    원본이 float64면 그 복사본도 float64라 메모리를 두 배로 쓴다.
    """
    d = np.load(path, allow_pickle=True)
    for k in ("X", "y", "subject_id"):
        if k not in d:
            raise KeyError(f"{path}에 '{k}'가 없습니다. (있는 키: {list(d.keys())})")
    X = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["y"]).astype(np.int64)
    return X, y, np.asarray(d["subject_id"])


def summarize(name, X, y, subject_id):
    n_subj = len(set(subject_id.tolist()))
    per = np.array([np.sum(subject_id == s) for s in np.unique(subject_id)])
    return (f"{name:<8} X{tuple(X.shape)}  정상 {(y == 0).sum()} / 환자 {(y == 1).sum()}"
            f"  피험자 {n_subj}명"
            f"  (1인당 윈도우 최소 {per.min()} / 중앙 {int(np.median(per))} / 최대 {per.max()})")


def make_loaders(X, y, fold, batch_size):
    """(train, val, test) DataLoader. train만 shuffle."""
    tr, va, te = fold
    return (
        DataLoader(WindowDataset(X[tr], y[tr]), batch_size=batch_size, shuffle=True),
        DataLoader(WindowDataset(X[va], y[va]), batch_size=batch_size),
        DataLoader(WindowDataset(X[te], y[te]), batch_size=batch_size),
    )


def class_weights(y_train, n_classes=2):
    """역빈도 가중치. 분모의 n_classes 덕에 가중치 평균이 1.0이 되어 손실
    스케일이 유지되므로 learning rate를 다시 잡을 필요가 없다.

    모달마다 불균형 정도가 다르므로(필기 1:1.77 · 음성 1:1.48) 반드시 따로 준다.
    한쪽 비율로 양쪽을 덮으면 다른 모달의 손실이 왜곡된다.
    """
    cnt = np.bincount(y_train, minlength=n_classes)
    return cnt.sum() / (n_classes * np.maximum(cnt, 1))


def _cycle(loader):
    """로더를 무한 반복. shuffle=True면 한 바퀴 돌 때마다 다시 섞인다."""
    while True:
        for batch in loader:
            yield batch


class PairedBatches:
    """모달별 로더를 한 스텝에 하나씩 짝지어 내보낸다.

    길이가 크게 다르다 — 필기 13,474 윈도우 vs 음성 2,008 윈도우(약 6.7배).
    두 정책이 있다.

      longest  (기본) 가장 긴 로더 기준으로 한 epoch을 돌고 짧은 쪽은 순환시킨다.
               큰 모달의 데이터를 버리지 않는다. 대신 작은 모달이 한 epoch에
               약 6.7회 반복 노출되므로 과적합이 빨리 온다.
      shortest zip과 같다. 짧은 쪽에 맞춰 끊는다. 필기 윈도우의 85%가 매 epoch
               버려지므로 필기 단독 기준선과의 비교가 성립하지 않는다.

    ⭐ 옵티마이저 스텝 수에 관한 메모
       한 스텝에서 두 모달의 손실을 **합산해 backward를 한 번만** 호출하므로,
       epoch당 옵티마이저 스텝 수는 longest 정책에서 필기 단독 학습과 같다.
       "공유 모델은 인코더 업데이트가 2배라 비교가 불공정하다"는 교란이
       이 구현에서는 생기지 않는다. (모달을 번갈아 backward 하면 2배가 된다)

       다만 스텝당 인코더가 보는 **토큰 수**는 여전히 다르다. batch 64 기준
       필기는 64×6=384 수열 × 39토큰 = 14,976, 음성은 64×64=4,096 수열 ×
       9토큰 = 36,864로 음성이 약 2.5배다. 배치 크기를 모달별로 따로 두면
       맞출 수 있으나, 그러면 "배치 크기" 축이 새로 생기므로 일단 두지 않았다.
    """

    def __init__(self, loaders, mode="longest"):
        if mode not in ("longest", "shortest"):
            raise ValueError(f"mode는 longest 또는 shortest (받은 값: {mode})")
        self.loaders = loaders
        self.mode    = mode
        lens = [len(l) for l in loaders.values()]
        self.n_steps = max(lens) if mode == "longest" else min(lens)

    def __len__(self):
        return self.n_steps

    def __iter__(self):
        iters = {m: _cycle(l) for m, l in self.loaders.items()}
        for _ in range(self.n_steps):
            yield {m: next(it) for m, it in iters.items()}


# ───────────────────────── 스모크 테스트용 합성 데이터 ─────────────────────────

def synthetic(n_subjects, n_channels, seq_len, win_per_subject=8, seed=0):
    """파이프라인 점검용 가짜 시계열. 실제 데이터 없이 학습이 도는지 확인한다.

    주의 — RevIN이 윈도우마다 평균 0·분산 1로 정규화하므로 "환자는 진폭이 크다"
    같은 신호는 정규화에 지워져 학습되지 않는다. 그래서 **주파수**로 라벨을
    심는다(정상 저주파 / 환자 고주파). 정규화 후에도 남는 성질이라야 스모크
    테스트가 의미 있다.
    """
    rng = np.random.default_rng(seed)

    # 라벨을 피험자 번호 순서와 어긋나게 섞는다.
    #   s % 2 로 주면 GroupKFold(층화 없음)가 홀/짝 피험자를 그대로 fold로 갈라
    #   train이 전부 정상 · test가 전부 환자가 된다. 그러면 학습에서 test 클래스를
    #   한 번도 못 봐 정확도가 0으로 나오고, 파이프라인 점검이 무의미해진다.
    #   (실제로 겪음 — README §7 "GroupKFold는 층화를 하지 않는다"의 사례)
    labels = np.array([i % 2 for i in range(n_subjects)])
    rng.shuffle(labels)

    X, y, sid = [], [], []
    for s in range(n_subjects):
        label = int(labels[s])                         # 절반씩 정상/환자
        freq  = 3.0 if label == 0 else 11.0            # 라벨을 주파수에 심는다
        t = np.linspace(0, 1, seq_len, dtype=np.float32)
        for _ in range(win_per_subject):
            phase = rng.uniform(0, 2 * np.pi, size=(n_channels, 1)).astype(np.float32)
            wave  = np.sin(2 * np.pi * freq * t[None, :] + phase)
            X.append(wave + rng.normal(0, 0.4, size=(n_channels, seq_len)).astype(np.float32))
            y.append(label)
            sid.append(f"S{s:03d}")
    return (np.stack(X).astype(np.float32),
            np.array(y, dtype=np.int64),
            np.array(sid))


def build_folds(y, subject_id, n_splits, seed, val_size=0.1, stratified=False):
    """피험자 단위 fold. dataset.subject_kfold를 그대로 쓴다(누수 검증 포함)."""
    return subject_kfold(subject_id, y, n_splits=n_splits,
                         val_size=val_size, seed=seed, stratified=stratified)
