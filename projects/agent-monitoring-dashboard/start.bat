@echo off
echo ========================================
echo Agent监控Dashboard - 快速启动
echo ========================================
echo.

echo [1/2] 检查依赖...
pip list | findstr "fastapi" >nul
if errorlevel 1 (
    echo 安装依赖...
    pip install -r requirements.txt
)

echo.
echo [2/2] 启动示例应用...
echo 访问 http://localhost:8000/docs
echo Prometheus指标: http://localhost:8000/metrics
python example_app.py

pause
