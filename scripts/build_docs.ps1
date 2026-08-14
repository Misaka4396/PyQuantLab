# 生成 pdoc API 文档（P4）
# 用法: powershell -ExecutionPolicy Bypass -File scripts\build_docs.ps1
# 产物: docs\api\（HTML，含 index.html）

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

& "E:\Anaconda3-2026\python.exe" -m pdoc -o docs/api engine etf ml data backtest cost_model cost_config core

$index = Join-Path $Root "docs\api\index.html"
if (Test-Path $index) {
    $n = (Get-ChildItem (Join-Path $Root "docs\api") -Recurse -Filter *.html | Measure-Object).Count
    Write-Host "[OK] API 文档已生成: docs\api\index.html（$n 个页面）"
} else {
    Write-Host "[错误] 文档生成失败" -ForegroundColor Red
    exit 1
}
