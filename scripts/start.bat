@echo off
REM VectorSphere 启动脚本
REM 使用方法: start.bat [mode] [port] [etcd_endpoints] [service_name]

setlocal enabledelayedexpansion

REM 设置默认参数
set MODE=%1
set PORT=%2
set ETCD=%3
set SERVICE=%4

if "%MODE%"=="" set MODE=standalone
if "%PORT%"=="" set PORT=8080
if "%ETCD%"=="" set ETCD=localhost:2379
if "%SERVICE%"=="" set SERVICE=VectorSphere

echo ========================================
echo VectorSphere 分布式向量数据库
echo ========================================
echo 运行模式: %MODE%
echo 服务端口: %PORT%
echo etcd 端点: %ETCD%
echo 服务名称: %SERVICE%
echo ========================================

REM 检查 etcd 是否运行
echo 检查 etcd 连接...
ping -n 1 localhost >nul 2>&1
if errorlevel 1 (
    echo 警告: 无法连接到 localhost，请确保 etcd 正在运行
    echo 启动 etcd 命令: etcd --listen-client-urls http://localhost:2379 --advertise-client-urls http://localhost:2379
    pause
    exit /b 1
)

REM 切换到项目根目录
cd /d "%~dp0.."

REM 检查配置文件
if not exist "conf\app.yaml" (
    echo 错误: 配置文件 conf\app.yaml 不存在
    pause
    exit /b 1
)

REM 创建必要的目录
if not exist "data" mkdir data
if not exist "logs" mkdir logs

echo 启动 VectorSphere 服务...
echo.

REM 根据模式启动服务
if "%MODE%"=="master" (
    echo 启动主节点...
    go run src\main.go --mode=master --port=%PORT% --etcd=%ETCD% --service=%SERVICE% --config=conf\app.yaml
) else if "%MODE%"=="slave" (
    echo 启动从节点...
    go run src\main.go --mode=slave --port=%PORT% --etcd=%ETCD% --service=%SERVICE% --config=conf\app.yaml
) else if "%MODE%"=="standalone" (
    echo 启动独立节点...
    go run src\main.go --mode=standalone --port=%PORT% --etcd=%ETCD% --service=%SERVICE% --config=conf\app.yaml
) else (
    echo 错误: 未知的运行模式 '%MODE%'
    echo 支持的模式: master, slave, standalone
    pause
    exit /b 1
)

if errorlevel 1 (
    echo.
    echo 服务启动失败，错误代码: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo 服务已停止
pause