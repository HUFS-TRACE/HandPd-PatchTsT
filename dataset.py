import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     StratifiedGroupKFold)


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"], d["subject_id"], d["task"]


def subject_kfold(subject_id, y, n_splits=5, val_size=0.1, seed=42,
                  stratified=False):
    """피험자 단위 K-Fold. 같은 사람이 train/val/test에 걸치지 않는다.

    stratified=False (기본): GroupKFold — 기존 동작 그대로 유지한다.
    stratified=True: StratifiedGroupKFold — fold마다 정상/환자 비율도 맞춘다.
        GroupKFold는 층화를 하지 않아 fold당 피험자가 적을 때(예: 10-fold면
        6명) 클래스 비율이 심하게 깨진 fold가 생길 수 있다.

    기본값을 유지하는 이유: 플래그를 켜지 않으면 이전 실험 결과가 그대로
    재현되어야 새 방식과의 비교가 성립한다.
    """
    splitter = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                     random_state=seed)
                if stratified else GroupKFold(n_splits=n_splits))
    folds = []

    for train_val_idx, test_idx in splitter.split(
            np.arange(len(y)), y, groups=subject_id):

        # train_val에서 val을 추가로 분리
        gss = GroupShuffleSplit(
            n_splits=1, test_size=val_size, random_state=seed
        )
        train_idx_rel, val_idx_rel = next(
            gss.split(
                train_val_idx,
                y[train_val_idx],
                groups=subject_id[train_val_idx]
            )
        )

        train_idx = train_val_idx[train_idx_rel]
        val_idx   = train_val_idx[val_idx_rel]

        # 데이터 누출 검증
        assert set(subject_id[train_idx]) & set(subject_id[val_idx])  == set()
        assert set(subject_id[train_idx]) & set(subject_id[test_idx]) == set()
        assert set(subject_id[val_idx])   & set(subject_id[test_idx]) == set()

        folds.append((train_idx, val_idx, test_idx))

    return folds
