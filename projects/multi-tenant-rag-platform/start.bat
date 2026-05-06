@echo off
echo ========================================
echo 多租户RAG平台 - 快速启动
echo ========================================
echo.

echo [1/3] 检查依赖...
pip list | findstr "streamlit" >nul
if errorlevel 1 (
    echo 安装Streamlit...
    pip install streamlit
)

echo.
echo [2/3] 运行核心演示...
python core_demo.py

echo.
echo [3/3] 启动管理后台...
echo 访问 http://localhost:8501
streamlit run admin_dashboard.py

pause
