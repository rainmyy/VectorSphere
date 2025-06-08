package main

import (
	"VectorSphere/src/index"
	"VectorSphere/src/library/log"
	"VectorSphere/src/library/tree"
	"VectorSphere/src/llm"
	"VectorSphere/src/server"
	"context"
	"flag"
	"fmt"
	"google.golang.org/grpc"
	"net"
	"strings"
	"time"
)

func main() {
	// 解析命令行参数
	mode := flag.String("mode", "standalone", "运行模式: master, slave 或 standalone")
	port := flag.Int("port", 8080, "服务端口")
	etcdEndpoints := flag.String("etcd", "localhost:2379", "etcd 端点，多个端点用逗号分隔")
	serviceName := flag.String("service", "VectorSphere", "服务名称")
	dataDir := flag.String("data", "./data", "数据目录")
	docNumEstimate := flag.Int("doc-num", 10000, "文档数量估计")
	dbType := flag.Int("db-type", 0, "数据库类型")
	flag.Parse()

	// 解析 etcd 端点
	endpointList := strings.Split(*etcdEndpoints, ",")
	var endpoints []server.EndPoint
	for _, ep := range endpointList {
		endpoints = append(endpoints, server.EndPoint{Ip: ep})
	}

	// 创建事务管理器、锁管理器和WAL管理器
	txMgr := tree.NewTransactionManager()
	lockMgr := tree.NewLockManager()
	wal, err := tree.NewWALManager(fmt.Sprintf("%s/wal.log", *dataDir))
	if err != nil {
		log.Fatal("创建WAL管理器失败: %v", err)
	}

	// 根据运行模式启动不同的服务
	switch *mode {
	case "master":
		// 启动主服务
		localhost := fmt.Sprintf("localhost:%d", *port)
		master, err := server.NewMasterService(context.Background(), endpoints, "VectorSphere", "localhost:50051", 8080, 5*time.Minute, 30*time.Second)
		if err != nil {
			log.Fatal("创建主服务失败: %v", err)
		}

		// 启动主服务
		err = master.Start(context.Background())
		if err != nil {
			log.Fatal("启动主服务失败: %v", err)
		}

		// 启动 gRPC 服务器
		lis, err := net.Listen("tcp", localhost)
		if err != nil {
			log.Fatal("监听端口失败: %v", err)
		}

		s := grpc.NewServer()
		log.Info("主服务启动，监听端口: %d", *port)
		if err := s.Serve(lis); err != nil {
			log.Fatal("启动 gRPC 服务器失败: %v", err)
		}

	case "slave":
		// 启动从服务
		slave, err := server.NewSlaveService(context.Background(), endpoints, *serviceName, *port)
		if err != nil {
			log.Fatal("创建从服务失败: %v", err)
		}

		// 初始化索引服务，添加缺少的参数
		err = slave.Init(*docNumEstimate, *dbType, *dataDir, txMgr, lockMgr, wal)
		if err != nil {
			log.Fatal("初始化索引服务失败: %v", err)
		}

		// 启动从服务
		err = slave.Start(context.Background())
		if err != nil {
			log.Fatal("启动从服务失败: %v", err)
		}

		// 启动 gRPC 服务器
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
		if err != nil {
			log.Fatal("监听端口失败: %v", err)
		}

		s := grpc.NewServer()
		server.RegisterIndexServiceServer(s, slave)

		log.Info("从服务启动，监听端口: %d", *port)
		if err := s.Serve(lis); err != nil {
			log.Fatal("启动 gRPC 服务器失败: %v", err)
		}

	case "standalone":
		// 启动独立服务
		indexServer := &server.IndexServer{}

		// 初始化索引服务
		err := indexServer.Init(*docNumEstimate, *dbType, *dataDir)
		if err != nil {
			log.Fatal("初始化索引服务失败: %v", err)
		}

		// 注册服务（如果需要）
		if len(endpoints) > 0 {
			err = indexServer.RegisterService(endpoints, *port, *serviceName)
			if err != nil {
				log.Warning("注册服务失败，将以本地模式运行: %v", err)
			}
		}

		// 启动 gRPC 服务器
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
		if err != nil {
			log.Fatal("监听端口失败: %v", err)
		}

		s := grpc.NewServer()
		server.RegisterIndexServiceServer(s, indexServer)

		log.Info("独立服务启动，监听端口: %d", *port)
		if err := s.Serve(lis); err != nil {
			log.Fatal("启动 gRPC 服务器失败: %v", err)
		}

	default:
		log.Fatal("未知的运行模式: %s", *mode)
	}
}

// 初始化AnalysisService
func initAnalysisService() (*llm.AnalysisService, error) {
	// 向量数据库路径和集群数量
	vectorDBPath := "./data/vector_db"
	numClusters := 10

	// LLM配置
	llmConfig := llm.LLMConfig{
		Type:        llm.LocalLLM,    // 或其他支持的LLM类型
		Model:       "gpt-3.5-turbo", // 或您使用的模型
		Temperature: 0.7,
		MaxTokens:   2000,
		Timeout:     30, // 秒
	}

	// BadgerDB路径用于会话存储
	badgerDBPath := "./data/sessions"

	// 创建跳表索引用于会话搜索加速
	skipListIndex := index.NewSkipListInvertedIndex(10000)

	// 初始化AnalysisService
	return llm.NewAnalysisService(vectorDBPath, numClusters, llmConfig, badgerDBPath, skipListIndex)
}

// etcd --listen-client-urls http://localhost:2379 --advertise-client-urls http://localhost:2379
// go run main.go --mode=master --port=8080 --etcd=localhost:2379 --service=VectorSphere
// go run main.go --mode=slave --port=8082 --etcd=localhost:2379 --service=VectorSphere
//go run main.go --mode=slave --port=8083 --etcd=localhost:2379 --service=VectorSphere
