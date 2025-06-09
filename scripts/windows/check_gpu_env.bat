@echo off
setlocal enabledelayedexpansion

echo ===== GPU环境检测 =====

:: 设置路径
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set FAISS_PATH=C:\faiss

echo 操作系统: Windows
echo CUDA路径: %CUDA_PATH%
echo FAISS路径: %FAISS_PATH%

:: 检测CUDA
set CUDA_AVAILABLE=0
if exist "%CUDA_PATH%\include\cuda_runtime.h" (
    echo CUDA环境检测: 可用
    set CUDA_AVAILABLE=1
) else (
    echo CUDA环境检测: 不可用
)

:: 检测FAISS
set FAISS_AVAILABLE=0
if exist "%FAISS_PATH%\include\faiss\Index.h" (
    echo FAISS环境检测: 可用
    set FAISS_AVAILABLE=1
) else (
    echo FAISS环境检测: 不可用
)

:: 输出结果
if %CUDA_AVAILABLE%==1 if %FAISS_AVAILABLE%==1 (
    echo GPU加速: 可用
    echo BUILD_TAGS=-tags gpu
    exit /b 0
) else (
    echo GPU加速: 不可用 (将使用CPU回退实现)
    echo BUILD_TAGS=
    exit /b 1
)