# 构建共享 DLL 目录（方案 A：主程序与 ML 训练器共用 mkl 数学库）
# 用法: powershell -ExecutionPolicy Bypass -File scripts\build_common.ps1
# 产物: dist\PyQuantLab_common\（23 个 mkl_*.dll）

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Src = Join-Path $Root "dist\PyQuantLab\_internal"
$Dst = Join-Path $Root "dist\PyQuantLab_common"

if (-not (Test-Path $Src)) {
    Write-Host "[错误] 未找到 $Src（请先打包主程序）" -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path $Dst | Out-Null
$copied = 0
Get-ChildItem $Src -Filter "mkl_*.dll" | ForEach-Object {
    Copy-Item $_.FullName $Dst -Force
    $copied++
}
$size = (Get-ChildItem $Dst -File | Measure-Object -Property Length -Sum).Sum
Write-Host "[OK] 已复制 $copied 个 mkl DLL 到 $Dst（$([Math]::Round($size/1MB)) MB）"
if ($copied -ne 23) {
    Write-Host "[警告] 预期 23 个，实际 $copied 个" -ForegroundColor Yellow
}
