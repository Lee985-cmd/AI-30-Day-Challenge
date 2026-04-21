@echo off
chcp 65001 >nul
echo ========================================
echo   智能投研助手 - 快速启动
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

echo [1/4] 检查虚拟环境...
if not exist "venv" (
    echo 创建虚拟环境...
    python -m venv venv
)

echo [2/4] 激活虚拟环境...
call venv\Scripts\activate.bat

echo [3/4] 安装依赖...
pip install -r requirements.txt -q

echo [4/4] 检查 LOCAL_LLM_URL 配置...
if "%LOCAL_LLM_URL%"=="" (
    echo.
    echo [警告] 未检测到 LOCAL_LLM_URL 环境变量
    echo 请设置环境变量后再运行
    echo.
    echo PowerShell 设置命令（管理员权限）:
    echo [System.Environment]::SetEnvironmentVariable("LOCAL_LLM_URL", "sk-your-api", "User")
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   环境准备完成！
echo ========================================
echo.
echo 请选择运行模式：
echo 1. 运行测试脚本
echo 2. 进入 Python 交互模式
echo 3. 退出
echo.

set /p choice="请输入选项 (1-3): "

if "%choice%"=="1" (
    echo.
    echo 正在运行测试脚本...
    python test_agent.py
) else if "%choice%"=="2" (
    echo.
    echo 进入 Python 交互模式...
    python
) else (
    echo 退出程序
)

pause
