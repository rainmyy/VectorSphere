package config

import "time"

// AppConfig 应用配置
type AppConfig struct {
	AppName             string             `yaml:"appName"`
	Version             string             `yaml:"version"`
	ListenAddress       string             `yaml:"listenAddress"`
	EtcdEndpoints       []string           `yaml:"etcdEndpoints"`
	EtcdDialTimeout     time.Duration      `yaml:"etcdDialTimeout"`
	EtcdUsername        string             `yaml:"etcdUsername"`
	EtcdPassword        string             `yaml:"etcdPassword"`
	EtcdTLS             *TLSConfig         `yaml:"etcdTLS"`
	ServiceRegistryPath string             `yaml:"serviceRegistryPath"` // e.g., /services/my-app/
	ServiceTTL          int64              `yaml:"serviceTTL"`          // 服务租约TTL (秒)
	ConfigPathPrefix    string             `yaml:"configPathPrefix"`    // 配置在etcd中的路径前缀, e.g., /config/my-app/
	LockPathPrefix      string             `yaml:"lockPathPrefix"`      // 分布式锁路径前缀
	ElectionPathPrefix  string             `yaml:"electionPathPrefix"`  // 领导者选举路径前缀
	RetryPolicy         *RetryPolicy       `yaml:"retryPolicy"`
	LoadBalancer        string             `yaml:"loadBalancer"` // e.g., "round_robin", "random"
	CircuitBreaker      *CBConfig          `yaml:"circuitBreaker"`
	RateLimiter         *RLConfig          `yaml:"rateLimiter"`
	ClientTLS           *TLSConfig         `yaml:"clientTLS"` // 客户端TLS配置
	HealthCheck         *HealthCheckConfig `yaml:"healthCheck"`
}
