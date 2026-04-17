@echo off
chcp 65001 >nul
echo ========================================
echo   AI 数据分析 Agent - 快速启动
echo ========================================
echo.

echo [1/4] 检查 Python 环境...
python --version
if errorlevel 1 (
    echo ❌ 未找到 Python，请先安装 Python 3.9+
    pause
    exit /b 1
)
echo ✅ Python 环境正常
echo.

echo [2/4] 安装依赖...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo ❌ 依赖安装失败
    pause
    exit /b 1
)
echo ✅ 依赖安装完成
echo.

echo [3/4] 生成示例数据...
python generate_sample_data.py
if errorlevel 1 (
    echo ❌ 数据生成失败
    pause
    exit /b 1
)
echo ✅ 示例数据生成完成
echo.

echo [4/4] 启动服务...
echo.
echo ⚠️  请确保已设置 DASHSCOPE_API_KEY 环境变量
echo.
echo 提示：如果没有设置，请按 Ctrl+C 停止，然后运行：
echo [System.Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "sk-your-api-key", "User")
echo.
echo 正在启动 API 服务...
start "API Service" cmd /k "python data_agent\api.py"
timeout /t 3 /nobreak >nul
echo 正在启动 Web 界面...
start "Web Interface" cmd /k "streamlit run data_agent\web_app.py"
echo.
echo ========================================
echo   ✅ 服务启动成功！
echo   API 文档: http://localhost:8000/docs
echo   Web 界面: http://localhost:8501
echo ========================================
echo.
pause
