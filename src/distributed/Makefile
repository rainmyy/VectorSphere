# VectorSphere 分布式系统 Makefile

# 变量定义
APP_NAME := vectorsphere
VERSION := 1.0.0
BUILD_TIME := $(shell date +%Y-%m-%d_%H:%M:%S)
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GO_VERSION := $(shell go version | awk '{print $$3}')

# 构建标志
LDFLAGS := -ldflags "-X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME) -X main.GitCommit=$(GIT_COMMIT) -X main.GoVersion=$(GO_VERSION)"

# 目录定义
BIN_DIR := ./bin
CONFIG_DIR := ./config
DATA_DIR := ./data
LOGS_DIR := ./logs
DOCKER_DIR := ./docker

# 默认目标
.PHONY: all
all: clean build

# 构建
.PHONY: build
build:
	@echo "Building $(APP_NAME)..."
	@mkdir -p $(BIN_DIR)
	go build $(LDFLAGS) -o $(BIN_DIR)/$(APP_NAME) main.go
	@echo "Build completed: $(BIN_DIR)/$(APP_NAME)"

# 构建所有平台
.PHONY: build-all
build-all: build-linux build-windows build-darwin

# Linux 构建
.PHONY: build-linux
build-linux:
	@echo "Building for Linux..."
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build $(LDFLAGS) -o $(BIN_DIR)/$(APP_NAME)-linux-amd64 main.go
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build $(LDFLAGS) -o $(BIN_DIR)/$(APP_NAME)-linux-arm64 main.go

# Windows 构建
.PHONY: build-windows
build-windows:
	@echo "Building for Windows..."
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build $(LDFLAGS) -o $(BIN_DIR)/$(APP_NAME)-windows-amd64.exe main.go

# macOS 构建
.PHONY: build-darwin
build-darwin:
	@echo "Building for macOS..."
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build $(LDFLAGS) -o $(BIN_DIR)/$(APP_NAME)-darwin-amd64 main.go
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build $(LDFLAGS) -o $(BIN_DIR)/$(APP_NAME)-darwin-arm64 main.go

# 安装依赖
.PHONY: deps
deps:
	@echo "Installing dependencies..."
	go mod download
	go mod tidy

# 运行测试
.PHONY: test
test:
	@echo "Running tests..."
	go test -v ./...

# 运行基准测试
.PHONY: bench
bench:
	@echo "Running benchmarks..."
	go test -bench=. -benchmem ./...

# 代码检查
.PHONY: lint
lint:
	@echo "Running linter..."
	golangci-lint run

# 格式化代码
.PHONY: fmt
fmt:
	@echo "Formatting code..."
	go fmt ./...
	goimports -w .

# 生成 protobuf
.PHONY: proto
proto:
	@echo "Generating protobuf files..."
	protoc --go_out=. --go-grpc_out=. ../server/proto/*.proto

# 清理
.PHONY: clean
clean:
	@echo "Cleaning..."
	rm -rf $(BIN_DIR)
	rm -rf $(DATA_DIR)
	rm -rf $(LOGS_DIR)

# 创建目录结构
.PHONY: init
init:
	@echo "Initializing project structure..."
	mkdir -p $(BIN_DIR)
	mkdir -p $(CONFIG_DIR)
	mkdir -p $(DATA_DIR)
	mkdir -p $(LOGS_DIR)
	mkdir -p $(DOCKER_DIR)
	@if [ ! -f $(CONFIG_DIR)/config.yaml ]; then \
		cp config/config.yaml $(CONFIG_DIR)/; \
	fi
	@echo "Project structure initialized"

# 运行（开发模式）
.PHONY: run
run: build
	@echo "Running $(APP_NAME) in development mode..."
	./$(BIN_DIR)/$(APP_NAME) -config $(CONFIG_DIR)/config.yaml -log debug

# 运行 master 节点
.PHONY: run-master
run-master: build
	@echo "Running $(APP_NAME) as master node..."
	./$(BIN_DIR)/$(APP_NAME) -config $(CONFIG_DIR)/config.yaml -type master -log info

# 运行 slave 节点
.PHONY: run-slave
run-slave: build
	@echo "Running $(APP_NAME) as slave node..."
	./$(BIN_DIR)/$(APP_NAME) -config $(CONFIG_DIR)/config.yaml -type slave -log info

# 启动 etcd（开发用）
.PHONY: etcd-start
etcd-start:
	@echo "Starting etcd for development..."
	mkdir -p ./etcd-data
	etcd --data-dir=./etcd-data --listen-client-urls=http://0.0.0.0:2379 --advertise-client-urls=http://localhost:2379 &
	@echo "etcd started on http://localhost:2379"

# 停止 etcd
.PHONY: etcd-stop
etcd-stop:
	@echo "Stopping etcd..."
	pkill etcd || true
	rm -rf ./etcd-data

# 启动集群（3节点）
.PHONY: cluster-start
cluster-start: build
	@echo "Starting 3-node cluster..."
	@mkdir -p $(DATA_DIR)/node1 $(DATA_DIR)/node2 $(DATA_DIR)/node3
	@mkdir -p $(LOGS_DIR)
	# 启动节点1（端口 8080, 9090）
	HTTP_PORT=8080 GRPC_PORT=9090 DATA_DIR=$(DATA_DIR)/node1 \
		./$(BIN_DIR)/$(APP_NAME) -config $(CONFIG_DIR)/config.yaml > $(LOGS_DIR)/node1.log 2>&1 &
	# 启动节点2（端口 8081, 9091）
	HTTP_PORT=8081 GRPC_PORT=9091 DATA_DIR=$(DATA_DIR)/node2 \
		./$(BIN_DIR)/$(APP_NAME) -config $(CONFIG_DIR)/config.yaml > $(LOGS_DIR)/node2.log 2>&1 &
	# 启动节点3（端口 8082, 9092）
	HTTP_PORT=8082 GRPC_PORT=9092 DATA_DIR=$(DATA_DIR)/node3 \
		./$(BIN_DIR)/$(APP_NAME) -config $(CONFIG_DIR)/config.yaml > $(LOGS_DIR)/node3.log 2>&1 &
	@echo "Cluster started. Check logs in $(LOGS_DIR)/"
	@echo "Node1: http://localhost:8080"
	@echo "Node2: http://localhost:8081"
	@echo "Node3: http://localhost:8082"

# 停止集群
.PHONY: cluster-stop
cluster-stop:
	@echo "Stopping cluster..."
	pkill $(APP_NAME) || true
	@echo "Cluster stopped"

# 查看集群状态
.PHONY: cluster-status
cluster-status:
	@echo "Checking cluster status..."
	@curl -s http://localhost:8080/api/v1/cluster/status | jq . || echo "Node1 not responding"
	@curl -s http://localhost:8081/api/v1/cluster/status | jq . || echo "Node2 not responding"
	@curl -s http://localhost:8082/api/v1/cluster/status | jq . || echo "Node3 not responding"

# Docker 构建
.PHONY: docker-build
docker-build:
	@echo "Building Docker image..."
	docker build -t $(APP_NAME):$(VERSION) -t $(APP_NAME):latest .

# Docker 运行
.PHONY: docker-run
docker-run:
	@echo "Running Docker container..."
	docker run -d --name $(APP_NAME) \
		-p 8080:8080 -p 9090:9090 \
		-v $(PWD)/config:/app/config \
		-v $(PWD)/data:/app/data \
		$(APP_NAME):latest

# Docker Compose 启动
.PHONY: docker-compose-up
docker-compose-up:
	@echo "Starting services with Docker Compose..."
	docker-compose up -d

# Docker Compose 停止
.PHONY: docker-compose-down
docker-compose-down:
	@echo "Stopping services with Docker Compose..."
	docker-compose down

# 性能测试
.PHONY: perf-test
perf-test: build
	@echo "Running performance tests..."
	@if [ ! -f $(BIN_DIR)/$(APP_NAME) ]; then echo "Binary not found. Run 'make build' first."; exit 1; fi
	# 启动测试实例
	./$(BIN_DIR)/$(APP_NAME) -config $(CONFIG_DIR)/config.yaml &
	APP_PID=$$!; \
	sleep 5; \
	echo "Running load test..."; \
	ab -n 1000 -c 10 http://localhost:8080/health; \
	kill $$APP_PID

# 生成文档
.PHONY: docs
docs:
	@echo "Generating documentation..."
	godoc -http=:6060 &
	@echo "Documentation server started at http://localhost:6060"

# 安装工具
.PHONY: install-tools
install-tools:
	@echo "Installing development tools..."
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	go install golang.org/x/tools/cmd/goimports@latest
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# 版本信息
.PHONY: version
version:
	@echo "$(APP_NAME) version $(VERSION)"
	@echo "Build time: $(BUILD_TIME)"
	@echo "Git commit: $(GIT_COMMIT)"
	@echo "Go version: $(GO_VERSION)"

# 打包发布
.PHONY: package
package: build-all
	@echo "Creating release packages..."
	mkdir -p ./release
	# Linux amd64
	tar -czf ./release/$(APP_NAME)-$(VERSION)-linux-amd64.tar.gz \
		-C $(BIN_DIR) $(APP_NAME)-linux-amd64 \
		-C ../config config.yaml \
		-C .. README.md
	# Linux arm64
	tar -czf ./release/$(APP_NAME)-$(VERSION)-linux-arm64.tar.gz \
		-C $(BIN_DIR) $(APP_NAME)-linux-arm64 \
		-C ../config config.yaml \
		-C .. README.md
	# Windows amd64
	zip -j ./release/$(APP_NAME)-$(VERSION)-windows-amd64.zip \
		$(BIN_DIR)/$(APP_NAME)-windows-amd64.exe \
		config/config.yaml README.md
	# macOS amd64
	tar -czf ./release/$(APP_NAME)-$(VERSION)-darwin-amd64.tar.gz \
		-C $(BIN_DIR) $(APP_NAME)-darwin-amd64 \
		-C ../config config.yaml \
		-C .. README.md
	# macOS arm64
	tar -czf ./release/$(APP_NAME)-$(VERSION)-darwin-arm64.tar.gz \
		-C $(BIN_DIR) $(APP_NAME)-darwin-arm64 \
		-C ../config config.yaml \
		-C .. README.md
	@echo "Release packages created in ./release/"

# 帮助信息
.PHONY: help
help:
	@echo "VectorSphere 分布式系统构建工具"
	@echo ""
	@echo "可用命令:"
	@echo "  build          构建应用程序"
	@echo "  build-all      构建所有平台版本"
	@echo "  build-linux    构建 Linux 版本"
	@echo "  build-windows  构建 Windows 版本"
	@echo "  build-darwin   构建 macOS 版本"
	@echo "  deps           安装依赖"
	@echo "  test           运行测试"
	@echo "  bench          运行基准测试"
	@echo "  lint           代码检查"
	@echo "  fmt            格式化代码"
	@echo "  proto          生成 protobuf 文件"
	@echo "  clean          清理构建文件"
	@echo "  init           初始化项目结构"
	@echo "  run            运行应用（开发模式）"
	@echo "  run-master     运行 master 节点"
	@echo "  run-slave      运行 slave 节点"
	@echo "  etcd-start     启动 etcd（开发用）"
	@echo "  etcd-stop      停止 etcd"
	@echo "  cluster-start  启动 3 节点集群"
	@echo "  cluster-stop   停止集群"
	@echo "  cluster-status 查看集群状态"
	@echo "  docker-build   构建 Docker 镜像"
	@echo "  docker-run     运行 Docker 容器"
	@echo "  perf-test      性能测试"
	@echo "  docs           生成文档"
	@echo "  install-tools  安装开发工具"
	@echo "  version        显示版本信息"
	@echo "  package        打包发布"
	@echo "  help           显示此帮助信息"