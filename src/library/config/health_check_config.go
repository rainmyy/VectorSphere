package config

import "time"

// HealthCheckConfig 健康检查配置
type HealthCheckConfig struct {
	Enabled           bool          `yaml:"enabled"`           // 是否启用健康检查
	CheckInterval     time.Duration `yaml:"checkInterval"`     // 健康检查间隔
	Timeout           time.Duration `yaml:"timeout"`           // 健康检查超时时间
	FailureThreshold  int           `yaml:"failureThreshold"`  // 失败阈值
	SuccessThreshold  int           `yaml:"successThreshold"`  // 成功阈值
	HeartbeatInterval time.Duration `yaml:"heartbeatInterval"` // 心跳间隔
	HeartbeatTimeout  time.Duration `yaml:"heartbeatTimeout"`  // 心跳超时
	GracefulShutdown  time.Duration `yaml:"gracefulShutdown"`  // 优雅关闭时间
	RetryPolicy       *RetryPolicy  `yaml:"retryPolicy"`       // 重试策略
}
