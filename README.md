# 기본값으로 실행
python train.py --data-path data/windows.npz

# 하이퍼파라미터 바꾸려면
python train.py --data-path data/windows.npz --epochs 100 --d-model 64 --dropout 0.1

# 클래스 가중치 조정
python train.py --data-path data/windows.npz --class-weight-healthy 3.0