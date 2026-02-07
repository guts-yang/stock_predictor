@echo off

rem 股票预测系统启动脚本
rem 支持K线预测功能

cls
echo 股票预测系统启动中...
echo 支持基线模型和K线模型预测
echo ---------------------------------------

rem 创建必要的目录结构
if not exist "models" mkdir "models"
if not exist "data" mkdir "data"
if not exist "logs" mkdir "logs"
if not exist "plots" mkdir "plots"  rem 新增目录：用于存储K线图表
echo 必要目录已创建/确认

rem 设置Python环境变量
set PYTHONIOENCODING=utf-8

rem 启动应用
echo 正在启动Flask应用...
python app.py

rem 捕获退出状态
if %errorlevel% neq 0 (
    echo 应用异常退出！
    pause
    exit /b %errorlevel%
)