@echo off
chcp 65001 >nul
title 亲子共调节系统
echo ============================================
echo   亲子共调节系统 - 一键启动
echo ============================================
echo.

:: 启动项目服务
echo [1/2] 启动项目服务...
start "Coregulation Server" cmd /k "cd /d "%~dp0" && .venv\Scripts\python.exe -m coregulation_poc web-live --host 127.0.0.1 --port 8766 --enable-closed-loop --enable-voice --window-seconds 10 --assessment-interval-seconds 10 --max-assessments 180"

:: 等待项目启动
echo 等待项目启动（8秒）...
timeout /t 8 /nobreak >nul

:: 启动 Cloudflare 隧道
echo [2/2] 启动 Cloudflare 隧道...
start "Cloudflare Tunnel" cmd /k "cd /d "%~dp0" && "%TEMP%\cloudflared.exe" tunnel --url http://127.0.0.1:8766"

echo.
echo ============================================
echo   两个窗口已启动，请勿关闭！
echo   隧道地址在 "Cloudflare Tunnel" 窗口里找
echo   形如 https://xxxxx.trycloudflare.com
echo.
echo   家庭端：打开隧道地址
echo   研究端：打开隧道地址/research
echo ============================================
echo.
pause
