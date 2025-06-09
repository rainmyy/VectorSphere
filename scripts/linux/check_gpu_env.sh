#!/bin/bash

# 检测CUDA环境
check_cuda() {
    if [ -d "$CUDA_PATH" ] && [ -f "$CUDA_PATH/include/cuda_runtime.h" ]; then
        echo "CUDA环境检测: 可用"
        return 0
    else
        echo "CUDA环境检测: 不可用"
        return 1
    fi
}

# 检测FAISS环境
check_faiss() {
    if [ -d "$FAISS_PATH" ] && [ -f "$FAISS_PATH/include/faiss/Index.h" ]; then
        echo "FAISS环境检测: 可用"
        return 0
    else
        echo "FAISS环境检测: 不可用"
        return 1
    fi
}

# 主函数
main() {
    echo "===== GPU环境检测 ====="

    # 检测操作系统
    if [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win"* ]]; then
        export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
        export FAISS_PATH="C:/faiss"
    else
        export CUDA_PATH="/usr/local/cuda"
        export FAISS_PATH="/usr/local/faiss"
    fi

    echo "操作系统: $OSTYPE"
    echo "CUDA路径: $CUDA_PATH"
    echo "FAISS路径: $FAISS_PATH"

    # 执行检测
    cuda_available=0
    faiss_available=0

    check_cuda
    if [ $? -eq 0 ]; then
        cuda_available=1
    fi

    check_faiss
    if [ $? -eq 0 ]; then
        faiss_available=1
    fi

    # 输出结果
    if [ $cuda_available -eq 1 ] && [ $faiss_available -eq 1 ]; then
        echo "GPU加速: 可用"
        echo "BUILD_TAGS=-tags gpu"
        exit 0
    else
        echo "GPU加速: 不可用 (将使用CPU回退实现)"
        echo "BUILD_TAGS="
        exit 1
    fi
}

# 执行主函数
main