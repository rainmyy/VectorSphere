package main

import (
	"VectorSphere/src/bootstrap"
	"VectorSphere/src/distributed"
	"VectorSphere/src/enhanced"
	"VectorSphere/src/index"
	"VectorSphere/src/library/config"
	"VectorSphere/src/library/log"
	"VectorSphere/src/library/tree"
	"VectorSphere/src/llm"
	"VectorSphere/src/server"
	"context"
	"flag"
	"fmt"
	"google.golang.org/grpc"
	"net"
	"os"
	"os/signal"
	"strings"
	"syscall"
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
	configFile := flag.String("config", "conf/app.yaml", "配置文件路径")
	flag.Parse()

	// 加载应用配置
	appConfig, err := config.LoadConfig(*configFile)
	if err != nil {
		log.Fatal("加载配置文件失败: %v", err)
	}

	// 设置etcd端点
	appConfig.EtcdEndpoints = strings.Split(*etcdEndpoints, ",")

	// 创建应用上下文和增强组件
	appCtx, enhancedServices, err := initializeEnhancedServices(appConfig, *serviceName)
	if err != nil {
		log.Fatal("初始化增强服务失败: %v", err)
	}
	defer appCtx.Close()

	// 设置优雅关闭
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Info("接收到关闭信号，开始优雅关闭...")
		cancel()
		enhancedServices.Shutdown()
	}()

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
		master, err := server.NewMasterService(ctx, endpoints, "VectorSphere", "localhost:50051", 8080, 5*time.Minute, 30*time.Second)
		if err != nil {
			log.Fatal("创建主服务失败: %v", err)
		}

		// 启动增强服务组件
		if err := enhancedServices.Start(ctx); err != nil {
			log.Fatal("启动增强服务组件失败: %v", err)
		}

		// 启动主服务
		err = master.Start(ctx)
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
		slave, err := server.NewSlaveService(ctx, endpoints, *serviceName, *port)
		if err != nil {
			log.Fatal("创建从服务失败: %v", err)
		}

		// 启动增强服务组件
		if err := enhancedServices.Start(ctx); err != nil {
			log.Fatal("启动增强服务组件失败: %v", err)
		}

		// 初始化索引服务，添加缺少的参数
		err = slave.Init(*docNumEstimate, *dbType, *dataDir, txMgr, lockMgr, wal)
		if err != nil {
			log.Fatal("初始化索引服务失败: %v", err)
		}

		// 启动从服务
		err = slave.Start(ctx)
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

		// 启动增强服务组件
		if err := enhancedServices.Start(ctx); err != nil {
			log.Fatal("启动增强服务组件失败: %v", err)
		}

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

// EnhancedServices 增强服务组件集合
type EnhancedServices struct {
	ServiceRegistry *enhanced.EnhancedServiceRegistry
	ConfigManager   *enhanced.EnhancedConfigManager
	DistributedLock *enhanced.EnhancedDistributedLock
	LeaderElection  *enhanced.EnhancedLeaderElection
	ErrorHandler    *enhanced.EnhancedErrorHandler
	HealthChecker   *enhanced.EnhancedHealthChecker
	LoadBalancer    *enhanced.EnhancedLoadBalancer
	SecurityManager *enhanced.EnhancedSecurityManager
}

// Start 启动所有增强服务组件
func (es *EnhancedServices) Start(ctx context.Context) error {
	log.Info("启动增强服务组件...")

	// 启动服务注册与发现
	if err := es.ServiceRegistry.Start(ctx); err != nil {
		return fmt.Errorf("启动服务注册失败: %w", err)
	}
	log.Info("服务注册与发现组件启动成功")

	// 启动配置管理
	if err := es.ConfigManager.Start(ctx); err != nil {
		return fmt.Errorf("启动配置管理失败: %w", err)
	}
	log.Info("配置管理组件启动成功")

	// 启动分布式锁
	if err := es.DistributedLock.Start(ctx); err != nil {
		return fmt.Errorf("启动分布式锁失败: %w", err)
	}
	log.Info("分布式锁组件启动成功")

	// 启动领导者选举
	if err := es.LeaderElection.Start(); err != nil {
		return fmt.Errorf("启动领导者选举失败: %w", err)
	}
	log.Info("领导者选举组件启动成功")

	// 启动错误处理
	if err := es.ErrorHandler.Start(ctx); err != nil {
		return fmt.Errorf("启动错误处理失败: %w", err)
	}
	log.Info("错误处理组件启动成功")

	// 启动健康检查
	if err := es.HealthChecker.Start(); err != nil {
		return fmt.Errorf("启动健康检查失败: %w", err)
	}
	log.Info("健康检查组件启动成功")

	// 启动负载均衡
	if err := es.LoadBalancer.Start(); err != nil {
		return fmt.Errorf("启动负载均衡失败: %w", err)
	}
	log.Info("负载均衡组件启动成功")

	// 启动安全管理
	if err := es.SecurityManager.Start(); err != nil {
		return fmt.Errorf("启动安全管理失败: %w", err)
	}
	log.Info("安全管理组件启动成功")

	log.Info("所有增强服务组件启动完成")
	return nil
}

// Shutdown 关闭所有增强服务组件
func (es *EnhancedServices) Shutdown() {
	log.Info("开始关闭增强服务组件...")

	// 按相反顺序关闭组件
	if es.SecurityManager != nil {
		es.SecurityManager.Stop()
		log.Info("安全管理组件已关闭")
	}

	if es.LoadBalancer != nil {
		es.LoadBalancer.Stop()
		log.Info("负载均衡组件已关闭")
	}

	if es.HealthChecker != nil {
		es.HealthChecker.Stop()
		log.Info("健康检查组件已关闭")
	}

	if es.ErrorHandler != nil {
		es.ErrorHandler.Stop()
		log.Info("错误处理组件已关闭")
	}

	if es.LeaderElection != nil {
		es.LeaderElection.Stop()
		log.Info("领导者选举组件已关闭")
	}

	if es.DistributedLock != nil {
		es.DistributedLock.Stop()
		log.Info("分布式锁组件已关闭")
	}

	if es.ConfigManager != nil {
		es.ConfigManager.Stop()
		log.Info("配置管理组件已关闭")
	}

	if es.ServiceRegistry != nil {
		es.ServiceRegistry.Stop()
		log.Info("服务注册组件已关闭")
	}

	log.Info("所有增强服务组件已关闭")
}

// initializeEnhancedServices 初始化增强服务组件
func initializeEnhancedServices(appConfig *config.AppConfig, serviceName string) (*bootstrap.AppContext, *EnhancedServices, error) {
	// 创建应用上下文
	appCtx, err := bootstrap.NewAppContext(appConfig)
	if err != nil {
		return nil, nil, fmt.Errorf("创建应用上下文失败: %w", err)
	}

	// 创建增强服务组件
	enhancedServices := &EnhancedServices{}

	// 初始化服务注册与发现
	enhancedServices.ServiceRegistry = enhanced.NewEnhancedServiceRegistry(
		appCtx.EtcdClient,
		&enhanced.ServiceRegistryConfig{
			ServiceName:    serviceName,
			RegistryPrefix: appConfig.ServiceRegistryPath,
			TTL:            time.Duration(appConfig.ServiceTTL) * time.Second,
			RetryInterval:  5 * time.Second,
			MaxRetries:     3,
		},
	)
	if err != nil {
		return nil, nil, fmt.Errorf("创建服务注册组件失败: %w", err)
	}

	// 初始化配置管理
	enhancedServices.ConfigManager, err = enhanced.NewEnhancedConfigManager(
		appCtx.EtcdClient,
		&distributed.ConfigManagerConfig{
			ConfigPrefix:   appConfig.ConfigPathPrefix,
			BackupEnabled:  true,
			BackupInterval: 1 * time.Hour,
			MaxBackups:     10,
			EncryptionKey:  "your-encryption-key-32-bytes-long", // 应该从安全配置中读取
		},
	)
	if err != nil {
		return nil, nil, fmt.Errorf("创建配置管理组件失败: %w", err)
	}

	// 初始化分布式锁
	enhancedServices.DistributedLock, err = enhanced.NewEnhancedDistributedLock(
		appCtx.EtcdClient,
		&distributed.DistributedLockConfig{
			LockPrefix:     "/locks/",
			DefaultTimeout: 30 * time.Second,
			RetryInterval:  1 * time.Second,
			MaxRetries:     5,
		},
	)
	if err != nil {
		return nil, nil, fmt.Errorf("创建分布式锁组件失败: %w", err)
	}

	// 初始化领导者选举
	enhancedServices.LeaderElection, err = enhanced.NewEnhancedLeaderElection(
		appCtx.EtcdClient,
		&distributed.LeaderElectionConfig{
			ElectionPrefix: appConfig.ElectionPathPrefix,
			CandidateID:    fmt.Sprintf("%s-%d", serviceName, time.Now().Unix()),
			TTL:            30 * time.Second,
			Strategy:       distributed.PriorityStrategy,
			Priority:       1,
		},
	)
	if err != nil {
		return nil, nil, fmt.Errorf("创建领导者选举组件失败: %w", err)
	}

	// 初始化错误处理
	enhancedServices.ErrorHandler = enhanced.NewEnhancedErrorHandler(
		&distributed.ErrorHandlerConfig{
			MaxRetries:        3,
			BaseDelay:         1 * time.Second,
			MaxDelay:          30 * time.Second,
			BackoffMultiplier: 2.0,
			JitterEnabled:     true,
		},
	)
	if err != nil {
		return nil, nil, fmt.Errorf("创建错误处理组件失败: %w", err)
	}

	// 初始化健康检查
	enhancedServices.HealthChecker = enhanced.NewEnhancedHealthChecker(
		appCtx.EtcdClient,
		&distributed.HealthCheckerConfig{
			CheckInterval:      10 * time.Second,
			Timeout:            5 * time.Second,
			UnhealthyThreshold: 3,
			HealthyThreshold:   2,
			PredictionEnabled:  true,
			AlertEnabled:       true,
		},
	)
	if err != nil {
		return nil, nil, fmt.Errorf("创建健康检查组件失败: %w", err)
	}

	// 初始化负载均衡
	enhancedServices.LoadBalancer = enhanced.NewEnhancedLoadBalancer(
		appCtx.EtcdClient,
		&enhanced.LoadBalancerConfig{
			Algorithm:             enhanced.WeightedRoundRobin,
			HealthCheckEnabled:    true,
			SessionAffinity:       true,
			SlowStartEnabled:      true,
			SlowStartDuration:     5 * time.Minute,
			CircuitBreakerEnabled: true,
		},
	)
	if err != nil {
		return nil, nil, fmt.Errorf("创建负载均衡组件失败: %w", err)
	}

	// 初始化安全管理
	enhancedServices.SecurityManager = enhanced.NewEnhancedSecurityManager(
		appCtx.EtcdClient,
		&enhanced.SecurityConfig{
			TLSEnabled:        true,
			RBACEnabled:       true,
			AuditEnabled:      true,
			EncryptionEnabled: true,
			SessionTimeout:    24 * time.Hour,
			PasswordPolicy: &enhanced.PasswordPolicy{
				MinLength:     8,
				RequireUpper:  true,
				RequireLower:  true,
				RequireDigit:  true,
				RequireSymbol: true,
			},
		},
	)
	if err != nil {
		return nil, nil, fmt.Errorf("创建安全管理组件失败: %w", err)
	}

	log.Info("增强服务组件初始化完成")
	return appCtx, enhancedServices, nil
}

// etcd --listen-client-urls http://localhost:2379 --advertise-client-urls http://localhost:2379
// go run main.go --mode=master --port=8080 --etcd=localhost:2379 --service=VectorSphere
// go run main.go --mode=slave --port=8082 --etcd=localhost:2379 --service=VectorSphere
//go run main.go --mode=slave --port=8083 --etcd=localhost:2379 --service=VectorSphere
