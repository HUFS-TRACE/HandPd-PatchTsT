# Protocol / regularization sweep, sequential.
#   ASCII only: PowerShell 5.1 reads .ps1 as ANSI unless the file has a BOM.
#
#   Sequential, not parallel: the earlier 2-way parallel run lost >half the
#   results because both processes wrote the same temp checkpoint. The
#   checkpoint name now includes the PID, but parallel gave only ~7% speedup
#   (GPU compute is the bottleneck), so it is not worth the risk.
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\JongyoonWon\Documents\GitHub\HandPd-PatchTsT"

$base = "--suite size --configs d64L2 --data-path data/windows_2s.npz"

$jobs = @(
  @("strat",        "--stratified"),
  @("val0.2",       "--val-size 0.2"),
  @("val0.05",      "--val-size 0.05"),
  @("autocw",       "--auto-class-weight"),
  @("ls0.05",       "--label-smoothing 0.05"),
  @("ls0.1",        "--label-smoothing 0.1"),
  @("ls0.2",        "--label-smoothing 0.2"),
  @("strat+autocw", "--stratified --auto-class-weight"),
  @("strat+ls0.1",  "--stratified --label-smoothing 0.1"),
  @("all",          "--stratified --auto-class-weight --label-smoothing 0.1")
)

$total = $jobs.Count * 3
$sw = [Diagnostics.Stopwatch]::StartNew()
$i = 0
foreach ($j in $jobs) {
  foreach ($s in 42, 1, 7) {
    $i++
    Write-Output "[$i/$total] $($j[0]) seed=$s"
    $p = Start-Process -FilePath "python" `
      -ArgumentList "evaluate.py $base $($j[1]) --seed $s" `
      -NoNewWindow -PassThru -Wait `
      -RedirectStandardOutput "results\_run.log" `
      -RedirectStandardError  "results\_err.log"
    if ($p.ExitCode -ne 0) {
      Write-Output "  FAILED (exit $($p.ExitCode))"
      Get-Content "results\_err.log" -Tail 5
    }
  }
}
$sw.Stop()
Write-Output ("done in {0:N1} min" -f $sw.Elapsed.TotalMinutes)
Remove-Item results\_run.log, results\_err.log, results\_tmp_*.pt -ErrorAction SilentlyContinue
