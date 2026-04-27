@echo off
chcp 65001 > nul
echo ============================================================
echo  产业链报告智能体 V3
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/2] 安装依赖...
python -m pip install -r backend\requirements.txt -q

echo [2/2] 启动服务...
echo.
echo  访问地址: http://localhost:5000
echo  按 Ctrl+C 停止服务
echo.
python -m backend.app

pause
