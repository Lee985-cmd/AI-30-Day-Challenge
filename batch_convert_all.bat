@echo off
chcp 65001 >nul
echo ========================================
echo   批量转换所有 Markdown 文件
echo ========================================
echo.

set /p theme="请选择主题 (default/light/dark/colorful/minimal): "
if "%theme%"=="" set theme=default

echo.
echo 🎨 使用主题: %theme%
echo.
echo 开始批量转换...
echo.

set count=0
for %%f in (promotion-articles\*.md) do (
    echo [!count!] 正在转换: %%f
    python tools\md_to_wechat_enhanced.py "%%f" -t %theme%
    set /a count+=1
)

echo.
echo ========================================
echo ✅ 批量转换完成！共处理 %count% 个文件
echo ========================================
echo.
pause
