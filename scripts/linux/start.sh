#!/bin/bash

# VectorSphere 启动脚本
# 使用方法: ./start.sh [mode] [port] [etcd_endpoints] [service_name]

# 设置默认参数
MODE=${1:-standalone}
PORT=${2:-8080}
ETCD=${3:-localhost:2379}
SERVICE=${4:-VectorSphere}

echo "========================================"
echo "VectorSphere 分布式向量数据库"
echo "========================================"
echo "运行模式: $MODE"
echo "服务端口: $PORT"
echo "etcd 端点: $ETCD"
echo "服务名称: $SERVICE"
echo "========================================"

# 检查 etcd 是否运行
echo "检查 etcd 连接..."
if ! nc -z localhost 2379 2>/dev/null; then
    echo "警告: 无法连接到 etcd (localhost:2379)"
    echo "请确保 etcd 正在运行"
    echo "启动 etcd 命令: etcd --listen-client-urls http://localhost:2379 --advertise-client-urls http://localhost:2379"
    exit 1
fi

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 检查配置文件
if [ ! -f "conf/app.yaml" ]; then
    echo "错误: 配置文件 conf/app.yaml 不存在"
    exit 1
fi

# 创建必要的目录
mkdir -p data logs

echo "启动 VectorSphere 服务..."
echo

# 设置信号处理
trap 'echo "\n接收到中断信号，正在关闭服务..."; kill $PID 2>/dev/null; wait $PID 2>/dev/null; echo "服务已关闭"; exit 0' INT TERM

# 根据模式启动服务
case "$MODE" in
    "master")
        echo "启动主节点..."
        go run src/main.go --mode=master --port=$PORT --etcd=$ETCD --service=$SERVICE --config=conf/app.yaml &
        PID=$!
        ;;
    "slave")
        echo "启动从节点..."
        go run src/main.go --mode=slave --port=$PORT --etcd=$ETCD --service=$SERVICE --config=conf/app.yaml &
        PID=$!
        ;;
    "standalone")
        echo "启动独立节点..."
        go run src/main.go --mode=standalone --port=$PORT --etcd=$ETCD --service=$SERVICE --config=conf/app.yaml &
        PID=$!
        ;;
    *)
        echo "错误: 未知的运行模式 '$MODE'"
        echo "支持的模式: master, slave, standalone"
        exit 1
        ;;
esac

# 等待进程
wait $PID
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo
    echo "服务启动失败，错误代码: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo
echo "服务已停止"