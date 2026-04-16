# AI 客服系统 - 安全启动脚本 (PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI 客服系统 - 安全启动脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否已设置 API Key
if (-not $env:DASHSCOPE_API_KEY) {
    Write-Host "[警告] 未检测到 DASHSCOPE_API_KEY 环境变量" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "请先设置环境变量：" -ForegroundColor White
    Write-Host ""
    Write-Host "  1. 临时设置（当前窗口有效）：" -ForegroundColor Gray
    Write-Host '     $env:DASHSCOPE_API_KEY="sk-your-api-key"' -ForegroundColor Green
    Write-Host ""
    Write-Host "  2. 永久设置（推荐）：" -ForegroundColor Gray
    Write-Host '     [System.Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "sk-your-api-key", "User")' -ForegroundColor Green
    Write-Host "     （需要重新打开 PowerShell）" -ForegroundColor Gray
    Write-Host ""
    Write-Host "获取 API Key: https://dashscope.console.aliyun.com/" -ForegroundColor Cyan
    Write-Host ""
    Read-Host "按回车键退出"
    exit 1
}

# 隐藏显示 API Key（只显示前后几位）
$apiKey = $env:DASHSCOPE_API_KEY
if ($apiKey.Length -gt 12) {
    $maskedKey = $apiKey.Substring(0, 8) + "..." + $apiKey.Substring($apiKey.Length - 4)
} else {
    $maskedKey = "***"
}

Write-Host "[成功] 检测到 API Key: $maskedKey" -ForegroundColor Green
Write-Host ""
Write-Host "正在启动服务..." -ForegroundColor Cyan
Write-Host ""

python ai_customer_service/api.py

Read-Host "`n按回车键退出"
