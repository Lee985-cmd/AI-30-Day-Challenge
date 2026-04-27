@echo off
chcp 65001 >nul
echo ========================================
echo   🤖 多模态智能客服系统
echo ========================================
echo.

cd /d "%~dp0"

echo 正在启动 Streamlit...
echo.
echo 浏览器将自动打开 http://localhost:8501
echo 按 Ctrl+C 停止服务
echo.

streamlit run streamlit_app.py

pause
