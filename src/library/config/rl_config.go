package config

import "golang.org/x/time/rate"

// RLConfig 限流器配置
type RLConfig struct {
	Rate  rate.Limit `yaml:"rate"`  // 每秒允许的事件数
	Burst int        `yaml:"burst"` // 令牌桶的容量
}
