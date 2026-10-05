# =====================================================================
#  GPU 실행 스크립트 (Windows PowerShell 판)
# =====================================================================
#
#  실행법
#    이 폴더에서 PowerShell을 열고
#
#        powershell -ExecutionPolicy Bypass -File run_on_gpu.ps1
#
#    끝나면 results_to_return 폴더가 생깁니다. 그것만 압축해 돌려주세요.
#
#  준비
#    data\windows_2s.npz 를 넣어두세요 (md5 e8c887ef62098cb7384fbf0e74ef4708)
#
#  원격 접속이 끊겨도
#    작업은 그 컴퓨터에서 계속 돕니다. 다시 접속해서 창을 보면 됩니다.
#    혹시 중단됐더라도 다시 실행하면 **이미 끝난 fold는 건너뛰고 이어서** 합니다.
#
#  VRAM 6GB(GTX 1660 등)에서
#    -OccBatch 를 줄이면 메모리를 덜 씁니다. 기본 32면 대개 안전합니다.
#    그래도 out of memory 가 나면 16 이나 8로 낮춰 다시 실행하세요.
#        powershell -ExecutionPolicy Bypass -File run_on_gpu.ps1 -OccBatch 16
# =====================================================================

param(
    [int]$OccBatch = 32,      # occlusion 배치. VRAM이 작으면 낮춘다
    [int]$Only = 0,           # 1이면 우선순위 1만 하고 끝낸다
    [string]$Py = "python"    # 파이썬 실행 이름. Store 스텁이 python을 가로채면
                              #   -Py py  또는  -Py "py -3.11"  로 지정한다
)

# $Py 가 "py -3.11" 처럼 인자를 포함할 수 있으므로 분리해 둔다
$PyExe  = ($Py -split ' ')[0]
$PyArgs = @($Py -split ' ' | Select-Object -Skip 1)
function Py { & $PyExe @PyArgs @args }

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$OUT = "results_to_return"
New-Item -ItemType Directory -Force -Path results, $OUT | Out-Null
$LOG = Join-Path $OUT "run_log.txt"

function Say($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $m
    Write-Host $line
    Add-Content -Path $LOG -Value $line -Encoding UTF8
}

# ── 환경 점검 ────────────────────────────────────────────────────────
Say "환경 점검"
Py -c @"
import sys, torch
print('  python', sys.version.split()[0])
print('  torch ', torch.__version__)
ok = torch.cuda.is_available()
if ok:
    p = torch.cuda.get_device_properties(0)
    print('  GPU   ', p.name, f'VRAM {p.total_memory/1024**3:.1f} GB')
else:
    print('  GPU    없음 - CPU로는 매우 오래 걸립니다')
"@ 2>&1 | Tee-Object -Append -FilePath $LOG

if (-not (Test-Path "data\windows_2s.npz")) {
    Say "!! data\windows_2s.npz 가 없습니다. 팀에서 받은 npz를 data\ 아래 두고 다시 실행하세요."
    exit 1
}

# 데이터가 팀이 쓰는 그 파일인지 확인한다.
#   원본에서 윈도우를 새로 만들면 안 된다 — 전처리 코드가 남아 있지 않아
#   비트 단위로 재현되지 않는다(0.6% 차이). 다른 npz면 fold 분할이 어긋나
#   결과를 팀 것과 나란히 놓을 수 없다.
$dataOk = Py -c @"
import sys, hashlib, numpy as np
EXPECT='e8c887ef62098cb7384fbf0e74ef4708'
h=hashlib.md5(open('data/windows_2s.npz','rb').read()).hexdigest()
print('  데이터 md5', h)
if h!=EXPECT:
    print('  !! 기대값과 다릅니다:', EXPECT)
    print('  !! 팀이 보낸 그 파일이어야 합니다. 원본에서 새로 만든 것은 안 됩니다.')
    sys.exit(1)
d=np.load('data/windows_2s.npz',allow_pickle=True)
print('  shape', {k:d[k].shape for k in d.files})
print('OK')
"@ 2>&1
$dataOk | Tee-Object -Append -FilePath $LOG | Out-Host
if ($LASTEXITCODE -ne 0) { Say "!! 데이터 확인 실패 - 중단합니다."; exit 1 }

Py -m pip install -q numpy scipy scikit-learn matplotlib pyyaml 2>&1 | Select-Object -Last 2

$CFG = "--config", "configs/protocol.yaml"

function HasCkpt($cfg, $seed) {
    (Get-ChildItem "results\ckpt_${cfg}_f4_s${seed}_*" -ErrorAction SilentlyContinue).Count -gt 0
}

function Train($suite, $cfg, $seed) {
    if (HasCkpt $cfg $seed) { Say "  건너뜀 (이미 있음): $cfg seed $seed"; return }
    Say "  학습: $cfg seed $seed"
    Py -u evaluate.py @CFG --suite $suite --configs $cfg --seed $seed `
        --save-ckpt --save-probs *>> $LOG
    if ($LASTEXITCODE -ne 0) { Say "  !! 학습 실패: $cfg seed $seed" }
}

function Xai($cfg, $seed) {
    Say "  occlusion: $cfg seed $seed  (배치 $OccBatch)"
    Py -u occlusion.py @CFG --configs $cfg --seed $seed `
        --win-ms 500,1000 --occ-batch-size $OccBatch *>> $LOG
    if ($LASTEXITCODE -ne 0) {
        Say "  !! occlusion 실패 - VRAM 부족이면 -OccBatch 16 으로 다시 실행하세요"
        return
    }
    foreach ($B in @("1,3","2,4","4,6","6,8","8,12","12,20")) {
        Py -u bandpower_test.py --configs $cfg --seed $seed --win-ms 1000 `
            --n-perm 1000 --importance abs --band $B *>> $LOG
    }
}

# ── 우선순위 1 : 패치 길이 통제 실험 ─────────────────────────────────
#   폭·깊이를 고정하고 패치 길이만 바꾼 두 짝을 학습해 비교한다.
#   이 팀 논문에서 가장 중요한 미해결 질문이다.
Say ""
Say "=== [1/3] 패치 길이 통제 실험 - 가장 중요 ==="
foreach ($C in @("d64L2p16", "p100d128L1")) {
    Train "patchctl" $C 42
    Xai $C 42
}
if ($Only -eq 1) { Say "우선순위 1만 실행하도록 지정됨 - 여기서 마칩니다" }

if ($Only -ne 1) {
    Say ""
    Say "=== [2/3] p16d128L1 시드 1,7 ==="
    foreach ($S in @(1, 7)) {
        Train "depth" "p16d128L1" $S
        Xai "p16d128L1" $S
    }

    Say ""
    Say "=== [3/3] 통제 실험 시드 확장 ==="
    foreach ($C in @("d64L2p16", "p100d128L1")) {
        foreach ($S in @(1, 7)) {
            Train "patchctl" $C $S
            Xai $C $S
        }
    }
}

# ── 돌려받을 것만 모으기 ─────────────────────────────────────────────
Say ""
Say "결과 정리"
Copy-Item results\bandpower_*.csv          $OUT -Force -ErrorAction SilentlyContinue
Copy-Item results\occlusion_*_summary*.csv $OUT -Force -ErrorAction SilentlyContinue
Copy-Item results\pareto_*.csv             $OUT -Force -ErrorAction SilentlyContinue
Copy-Item results\folds_*.csv              $OUT -Force -ErrorAction SilentlyContinue
Copy-Item results\ckpt_*.pt                $OUT -Force -ErrorAction SilentlyContinue
$n = (Get-ChildItem $OUT).Count
$mb = [math]::Round(((Get-ChildItem $OUT | Measure-Object Length -Sum).Sum / 1MB), 1)
Say "완료. $OUT 폴더($n 개 파일, $mb MB)를 압축해 돌려주세요."
