@echo off
chcp 65001 >nul
echo ========================================
echo   Markdown 转公众号格式工具
echo ========================================
echo.

if "%~1"=="" (
    echo ❌ 用法: convert.bat ^<markdown文件路径^>
    echo.
    echo 示例:
    echo   convert.bat promotion-articles\公众号首篇.md
    echo   convert.bat "promotion-articles\公众号文章-AI 客服系统实战.md"
    echo.
    pause
    exit /b 1
)

echo 📄 正在转换: %~1
echo.

python tools\md_to_wechat.py %~1

if %errorlevel% equ 0 (
    echo.
    echo ✅ 转换成功！
    echo.
    echo 💡 下一步操作：
    echo    1. 用浏览器打开生成的 .html 文件
    echo    2. 全选复制 (Ctrl+A, Ctrl+C)
    echo    3. 粘贴到公众号编辑器
) else (
    echo.
    echo ❌ 转换失败，请检查错误信息
)

echo.
pause
