@echo off
setlocal enabledelayedexpansion

REM 设置默认值
set ROLE=both
set CONFIG_PATH=D:\code\VectorSphere\conf\idc\simple\service.yaml
set LOG_LEVEL=info

REM 解析命令行参数
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="-role" (
    set ROLE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-config" (
    set CONFIG_PATH=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-log-level" (
    set LOG_LEVEL=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-h" (
    goto :show_help
)
if "%~1"=="--help" (
    goto :show_help
)
shift
goto :parse_args
:end_parse_args

REM 验证角色参数
if not "%ROLE%"=="master" if not "%ROLE%"=="slave" if not "%ROLE%"=="both" (
    echo 错误: 无效的角色参数 "%ROLE%"
    echo 有效的角色选项: master, slave, both
    exit /b 1
)

REM 验证日志级别参数
if not "%LOG_LEVEL%"=="debug" if not "%LOG_LEVEL%"=="info" if not "%LOG_LEVEL%"=="warn" if not "%LOG_LEVEL%"=="error" (
    echo 错误: 无效的日志级别参数 "%LOG_LEVEL%"
    echo 有效的日志级别选项: debug, info, warn, error
    exit /b 1
)

REM 检查配置文件是否存在
if not exist "%CONFIG_PATH%" (
    echo 错误: 配置文件不存在: %CONFIG_PATH%
    exit /b 1
)

REM 显示启动信息
echo 正在启动 VectorSphere 服务...
echo 角色: %ROLE%
echo 配置文件: %CONFIG_PATH%
echo 日志级别: %LOG_LEVEL%
echo.

REM 启动服务
cd /d D:\code\VectorSphere
go run src\main.go -role=%ROLE% -config="%CONFIG_PATH%" -log-level=%LOG_LEVEL%

goto :eof

:show_help
echo VectorSphere 启动脚本
echo 用法: start.bat [选项]
echo.
echo 选项:
echo   -role ^<role^>        服务角色: master, slave, both (默认: both)
echo   -config ^<path^>      配置文件路径 (默认: D:\code\VectorSphere\conf\idc\simple\service.yaml)
echo   -log-level ^<level^>  日志级别: debug, info, warn, error (默认: info)
echo   -h, --help           显示此帮助信息
echo.
echo 示例:
echo   start.bat -role=master -log-level=debug
echo   start.bat -role=slave -config="D:\code\VectorSphere\conf\custom\service.yaml"
exit /b 0