@echo off
chcp 65001 >nul
echo ========================================
echo   AI 客服系统 - 安全启动脚本
echo ========================================
echo.

REM 检查是否已设置 API Key
if "%DASHSCOPE_API_KEY%"=="" (
    echo [警告] 未检测到 DASHSCOPE_API_KEY 环境变量
    echo.
    echo 请先设置环境变量：
    echo   1. 临时设置（当前窗口有效）：
    echo      set DASHSCOPE_API_KEY=sk-your-api-key
    echo.
    echo   2. 永久设置（推荐）：
    echo      setx DASHSCOPE_API_KEY "sk-your-api-key"
    echo      （需要重新打开命令行窗口）
    echo.
    echo 获取 API Key: https://dashscope.console.aliyun.com/
    echo.
    pause
    exit /b 1
)

echo [成功] 检测到 API Key: %DASHSCOPE_API_KEY:~0,8%...%DASHSCOPE_API_KEY:~-4%
echo.
echo 正在启动服务...
echo.

python ai_customer_service/api.py

pause
