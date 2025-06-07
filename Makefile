# 跨平台Makefile - 支持Windows和Linux环境

# 检测操作系统
ifeq ($(OS),Windows_NT)
    # Windows 环境变量设置
    DETECTED_OS := Windows
    CUDA_PATH := C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8
    FAISS_PATH := C:/faiss
    LIB_EXT := .dll
    EXE_EXT := .exe
    PATH_SEP := \\
    RM_CMD := del
    SET_ENV := set
else
    # Linux 环境变量设置
    DETECTED_OS := Linux
    CUDA_PATH := /usr/local/cuda
    FAISS_PATH := /usr/local/faiss
    LIB_EXT := .so
    EXE_EXT :=
    PATH_SEP := /
    RM_CMD := rm -f
    SET_ENV := export
endif

# 通用编译器设置
CXX := g++
CXXFLAGS := -shared -fPIC -O3
INCLUDE_FLAGS := -I$(CUDA_PATH)/include -I$(FAISS_PATH)/include

# 平台特定的库路径和链接设置
ifeq ($(DETECTED_OS),Windows)
    LIB_FLAGS := -L$(CUDA_PATH)/lib/x64 -L$(FAISS_PATH)/lib
    LINK_FLAGS := -lcudart -lcuda -lfaiss -lfaiss_gpu
    WRAPPER_OUTPUT := src/library/algorithm/libfaiss_gpu_wrapper$(LIB_EXT)
else
    LIB_FLAGS := -L$(CUDA_PATH)/lib64 -L$(FAISS_PATH)/lib
    LINK_FLAGS := -lcudart -lcuda -lfaiss -lfaiss_gpu -Wl,-rpath,$(CUDA_PATH)/lib64 -Wl,-rpath,$(FAISS_PATH)/lib
    WRAPPER_OUTPUT := src/library/algorithm/libfaiss_gpu_wrapper$(LIB_EXT)
endif

# 显示检测到的操作系统
info:
	@echo "检测到的操作系统: $(DETECTED_OS)"
	@echo "CUDA路径: $(CUDA_PATH)"
	@echo "FAISS路径: $(FAISS_PATH)"
	@echo "库文件扩展名: $(LIB_EXT)"
	@echo "可执行文件扩展名: $(EXE_EXT)"

# 编译 C++ 包装器
faiss_wrapper:
	@echo "正在为 $(DETECTED_OS) 编译 FAISS GPU 包装器..."
	@echo "g++ command: $(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $(LIB_FLAGS) $(LINK_FLAGS) -o $(WRAPPER_OUTPUT) src/library/algorithm/faiss_gpu_wrapper.cpp"
	$(CXX) $(CXXFLAGS) \
		$(INCLUDE_FLAGS) \
		$(LIB_FLAGS) \
		$(LINK_FLAGS) \
		-o $(WRAPPER_OUTPUT) \
		src/library/algorithm/faiss_gpu_wrapper.cpp
	@echo "FAISS GPU 包装器编译完成: $(WRAPPER_OUTPUT)"

# 编译 Go 程序
ifeq ($(DETECTED_OS),Windows)
build: faiss_wrapper
	@echo "正在为 Windows 编译 Go 程序..."
	$(SET_ENV) CGO_ENABLED=1 && go build -o VectorSphere$(EXE_EXT) .
	@echo "编译完成: VectorSphere$(EXE_EXT)"
else
build: faiss_wrapper
	@echo "正在为 Linux 编译 Go 程序..."
	@echo "LD_LIBRARY_PATH: $(CUDA_PATH)/lib64:$(FAISS_PATH)/lib:$$LD_LIBRARY_PATH"
	$(SET_ENV) CGO_ENABLED=1 && \
	$(SET_ENV) LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(FAISS_PATH)/lib:$$LD_LIBRARY_PATH && \
	go build -o VectorSphere$(EXE_EXT) .
	@echo "编译完成: VectorSphere$(EXE_EXT)"
endif

# 安装依赖 (仅Linux)
install-deps-linux:
	@echo "安装 Linux 依赖..."
	sudo apt-get update
	sudo apt-get install -y build-essential cmake git
	# 注意: CUDA 和 FAISS 需要手动安装
	@echo "请手动安装 CUDA Toolkit 和 FAISS-GPU"

# 清理
ifeq ($(DETECTED_OS),Windows)
clean:
	@echo "清理 Windows 构建文件..."
	-$(RM_CMD) src(PATH_SEP)library$(PATH_SEP)algorithm$(PATH_SEP)libfaiss_gpu_wrapper$(LIB_EXT)
	-$(RM_CMD) VectorSphere$(EXE_EXT)
	@echo "清理完成"
else
clean:
	@echo "清理 Linux 构建文件..."
	-$(RM_CMD) src(PATH_SEP)library$(PATH_SEP)algorithm$(PATH_SEP)libfaiss_gpu_wrapper$(LIB_EXT)
	-$(RM_CMD) VectorSphere$(EXE_EXT)
	@echo "清理完成"
endif

# 测试编译环境
test-env:
	@echo "测试编译环境..."
	@echo "操作系统: $(DETECTED_OS)"
	@which $(CXX) || echo "警告: 未找到 g++ 编译器"
	@test -d "$(CUDA_PATH)" && echo "CUDA路径存在: $(CUDA_PATH)" || echo "警告: CUDA路径不存在: $(CUDA_PATH)"
	@test -d "$(FAISS_PATH)" && echo "FAISS路径存在: $(FAISS_PATH)" || echo "警告: FAISS路径不存在: $(FAISS_PATH)"
	@go version || echo "警告: 未找到 Go 编译器"

# 帮助信息
help:
	@echo "可用的make目标:"
	@echo "  info          - 显示检测到的系统信息"
	@echo "  faiss_wrapper - 编译 FAISS GPU 包装器"
	@echo "  build         - 编译完整程序 (包括包装器)"
	@echo "  clean         - 清理构建文件"
	@echo "  test-env      - 测试编译环境"
	@echo "  install-deps-linux - 安装Linux依赖 (仅Linux)"
	@echo "  help          - 显示此帮助信息"

# 默认目标
.DEFAULT_GOAL := build

# 声明伪目标
.PHONY: info faiss_wrapper build clean test-env install-deps-linux help