package config

import (
	"github.com/sony/gobreaker"
	"time"
)

// CBConfig 熔断器配置
type CBConfig struct {
	Name          string                                                      `yaml:"name"`        // 熔断器名称
	MaxRequests   uint32                                                      `yaml:"maxRequests"` // 半开状态下允许的请求数
	Interval      time.Duration                                               `yaml:"interval"`    // 计数器重置周期
	Timeout       time.Duration                                               `yaml:"timeout"`     // 从打开状态到半开状态的超时时间
	ReadyToTrip   func(counts gobreaker.Counts) bool                          `yaml:"-"`           // 自定义判断是否熔断的函数
	OnStateChange func(name string, from gobreaker.State, to gobreaker.State) `yaml:"-"`           // 状态变化回调
}
