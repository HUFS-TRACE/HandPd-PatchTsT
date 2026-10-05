GPU 실행 안내 — HandPD 파킨슨 스크리닝 실험
============================================

[ Windows 에서 (원격 데스크톱 포함) ]

  1. 이 zip을 아무 폴더에나 풉니다
  2. windows_2s.npz 를 data\ 아래에 넣습니다
       gpu_package\data\windows_2s.npz
  3. 그 폴더에서 PowerShell을 엽니다
       탐색기 주소창에  powershell  이라고 치고 Enter
  4. 아래를 붙여넣습니다

       powershell -ExecutionPolicy Bypass -File run_on_gpu.ps1 -Only 1

     -Only 1  = 우선순위 1(패치 길이 통제 실험)만. 약 40분
     빼면 전체를 다 합니다 (약 3시간)

  VRAM 6GB(GTX 1660 등)에서 out of memory 가 나면 배치를 줄입니다
       powershell -ExecutionPolicy Bypass -File run_on_gpu.ps1 -Only 1 -OccBatch 16

  원격 접속이 끊겨도 그 컴퓨터에서 계속 돕니다. 다시 접속해 창을 보면 됩니다.
  중간에 멈췄더라도 다시 실행하면 끝난 것은 건너뛰고 이어서 합니다.


[ 리눅스 / WSL 에서 ]

  bash run_on_gpu.sh


[ 먼저 확인할 것 ]

  torch가 CUDA로 설치돼 있어야 합니다.

       python -c "import torch; print(torch.cuda.is_available())"

  True 가 나와야 합니다. False 면:
       pip install torch --index-url https://download.pytorch.org/whl/cu121


[ 끝나면 ]

  results_to_return 폴더가 생깁니다. 그 폴더만 압축해서 주시면 됩니다.
  (100MB 이하)

  문제가 생기면 results_to_returnun_log.txt 를 함께 주세요.


[ 데이터에 대해 ]

  [중요] 반드시 팀이 쓰는 그 npz 파일이어야 합니다
  md5 = e8c887ef62098cb7384fbf0e74ef4708
  확인:  md5sum data/windows_2s.npz

  원본 NewHandPD 신호에서 윈도우를 새로 만들면 안 됩니다. 팀의 전처리
  코드가 남아 있지 않아 윈도우 분할이 비트 단위로 재현되지 않습니다
  (0.6% 차이). 다른 npz로 돌리면 fold 분할이 어긋나 결과를 팀 것과
  나란히 놓을 수 없습니다.
  스크립트가 md5를 확인하고, 다르면 실행을 멈춥니다.

필요 환경
  NVIDIA GPU (VRAM 8GB 이상이면 충분. T4 / RTX 3060 이상)
  python 3.9 이상, pip
  torch는 미리 설치돼 있어야 합니다 (CUDA 빌드)
  확인:  python -c "import torch; print(torch.cuda.is_available())"
  없으면: pip install torch --index-url https://download.pytorch.org/whl/cu121

시간
  우선순위 1만  약 1시간   <- 이것만 해주셔도 큰 도움입니다
  전체         약 3시간
  중간에 끊겨도 됩니다. 다시 실행하면 남은 것만 이어서 합니다.

돌려주실 것
  results_to_return/ 폴더 (100MB 이하)

데이터에 대해
  NewHandPD 공개 데이터셋을 전처리한 것입니다. 연구 목적 공개 자료라
  실행에 문제 없습니다. 다만 개인 신호이므로 실행 후 지워주시면 감사합니다.

문제가 생기면
  results_to_return/run_log.txt 를 같이 보내주세요. 원인이 거기 남습니다.
