package main

import (
	"VectorSphere/src/bootstrap"
	"VectorSphere/src/distributed"
	"VectorSphere/src/library/logger"
	"VectorSphere/src/server"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	// 解析命令行参数
	configPath := flag.String("config", "config.yaml", "配置文件路径")
	nodeType := flag.String("mode", "master", "节点类型 (master/slave)")
	port := flag.Int("port", 8080, "服务端口")
	httpPort := flag.Int("http-port", 8081, "HTTP服务端口")
	etcdEndpoints := flag.String("etcd", "localhost:2379", "etcd端点，多个端点用逗号分隔")
	serviceName := flag.String("service", "VectorSphere", "服务名称")
	flag.Parse()

	// 加载配置
	config, err := bootstrap.LoadConfig(*configPath)
	if err != nil {
		logger.Info("Failed to load config from file, using default and command line parameters")
		// 使用默认配置
		config = &bootstrap.AppConfig{
			ServiceName:           *serviceName,
			NodeID:                fmt.Sprintf("%s-%d", *nodeType, *port),
			NodeType:              *nodeType,
			Version:               "1.0.0",
			Region:                "default",
			Zone:                  "default",
			Tags:                  []string{"production"},
			Port:                  *port,
			HttpPort:              *httpPort,
			EtcdEndpoints:         parseEtcdEndpoints(*etcdEndpoints),
			EtcdTimeout:           5,
			ServiceTTL:            60,
			HeartbeatInterval:     time.Second * 30,
			HealthCheckInterval:   time.Second * 60,
			HealthCheckEndpoint:   "/health",
			LoadBalancerAlgorithm: "round_robin",
			MaxRetries:            3,
			RetryTimeout:          time.Second * 10,
			ConfigNamespace:       "vector_sphere",
			ConfigEnv:             "production",
		}
	} else {
		// 命令行参数覆盖配置文件
		if *nodeType != "" {
			config.NodeType = *nodeType
		}
		if *port != 0 {
			config.Port = *port
		}
		if *httpPort != 0 {
			config.HttpPort = *httpPort
		}
		if *etcdEndpoints != "" {
			config.EtcdEndpoints = parseEtcdEndpoints(*etcdEndpoints)
		}
		if *serviceName != "" {
			config.ServiceName = *serviceName
		}
	}

	// 创建上下文
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 初始化etcd客户端
	etcdClient, err := bootstrap.InitEtcdClient(config)
	if err != nil {
		logger.Fatal("Failed to initialize etcd client: %v", err)
	}
	defer etcdClient.Close()

	// 初始化服务注册
	serviceRegistry, err := bootstrap.InitServiceRegistry(etcdClient, config)
	if err != nil {
		logger.Fatal("Failed to initialize service registry: %v", err)
	}

	// 注册服务
	err = bootstrap.RegisterService(ctx, serviceRegistry, config)
	if err != nil {
		logger.Fatal("Failed to register service: %v", err)
	}

	// 初始化负载均衡器
	loadBalancer, err := bootstrap.InitLoadBalancer(etcdClient, serviceRegistry, config)
	if err != nil {
		logger.Fatal("Failed to initialize load balancer: %v", err)
	}

	// 初始化配置管理器
	configManager, err := bootstrap.InitConfigManager(etcdClient, config)
	if err != nil {
		logger.Fatal("Failed to initialize config manager: %v", err)
	}

	// 创建分布式管理器配置
	dmConfig := &distributed.DistributedConfig{
		ServiceName: config.ServiceName,
		NodeType:    getNodeType(config.NodeType),
		TimeOut:     config.EtcdTimeout,
		DefaultPort: config.Port,
		Heartbeat:   int(config.HeartbeatInterval.Seconds()),
		Etcd: distributed.EtcdConfig{
			Endpoints: config.EtcdEndpoints,
		},
		SchedulerWorkerCount:  10,
		HttpPort:              config.HttpPort,
		TaskTimeout:           30,
		HealthCheckInterval:   int(config.HealthCheckInterval.Seconds()),
		Endpoints:             make(map[string]server.EndPoint),
		DataDir:               "data",
		LoadBalancerConfig:    loadBalancer,
		EnhancedConfigManager: configManager,
	}

	// 创建分布式管理器
	dm, err := distributed.NewDistributedManager(dmConfig)
	if err != nil {
		logger.Fatal("Failed to create distributed manager: %v", err)
	}

	// 启动分布式管理器
	err = dm.Start()
	if err != nil {
		logger.Fatal("Failed to start distributed manager: %v", err)
	}

	// 创建API网关
	apiGateway := createAPIGateway(dm, config)

	// 启动API网关
	if apiGateway != nil {
		err = apiGateway.Start(ctx)
		if err != nil {
			logger.Fatal("Failed to start API gateway: %v", err)
		}
	}

	// 等待信号
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// 打印启动信息
	printStartupInfo(config)

	// 等待退出信号
	<-sigCh
	logger.Info("Received shutdown signal, gracefully shutting down...")

	// 优雅关闭
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	// 停止API网关
	if apiGateway != nil {
		if err := apiGateway.Stop(shutdownCtx); err != nil {
			logger.Error("Failed to stop API gateway: %v", err)
		}
	}

	// 停止分布式管理器
	if err := dm.Stop(); err != nil {
		logger.Error("Failed to stop distributed manager: %v", err)
	}

	logger.Info("Shutdown complete")
}

// parseEtcdEndpoints 解析etcd端点
func parseEtcdEndpoints(endpoints string) []string {
	// 简单实现，按逗号分隔
	return distributed.ParseEndpoints(endpoints)
}

// getNodeType 获取节点类型
func getNodeType(nodeTypeStr string) distributed.NodeType {
	if nodeTypeStr == "master" {
		return distributed.MasterNode
	}
	return distributed.SlaveNode
}

// createAPIGateway 创建API网关
func createAPIGateway(dm *distributed.DistributedManager, config *bootstrap.AppConfig) *distributed.APIGateway {
	// 只有主节点才创建API网关
	if config.NodeType != "master" {
		return nil
	}

	// 创建服务发现
	sd := distributed.NewServiceDiscovery(dm.GetEtcdClient(), config.ServiceName)

	// 创建通信服务
	commSvc := distributed.NewCommunicationService(dm.GetEtcdClient(), config.ServiceName)

	// 创建API网关
	return distributed.NewAPIGateway(dm, commSvc, sd, config.HttpPort)
}

// printStartupInfo 打印启动信息
func printStartupInfo(config *bootstrap.AppConfig) {
	nodeTypeStr := "从节点"
	if config.NodeType == "master" {
		nodeTypeStr = "主节点"
	}

	fmt.Printf("\n========================================\n")
	fmt.Printf("  VectorSphere %s 已启动\n", config.Version)
	fmt.Printf("  节点类型: %s\n", nodeTypeStr)
	fmt.Printf("  服务端口: %d\n", config.Port)
	if config.NodeType == "master" {
		fmt.Printf("  HTTP API端口: %d\n", config.HttpPort)
	}
	fmt.Printf("  etcd端点: %v\n", config.EtcdEndpoints)
	fmt.Printf("========================================\n\n")
}

// etcd --listen-client-urls http://localhost:2379 --advertise-client-urls http://localhost:2379
// go run main.go --mode=master --port=8080 --etcd=localhost:2379 --service=VectorSphere
// go run main.go --mode=slave --port=8082 --etcd=localhost:2379 --service=VectorSphere
//go run main.go --mode=slave --port=8083 --etcd=localhost:2379 --service=VectorSphere
