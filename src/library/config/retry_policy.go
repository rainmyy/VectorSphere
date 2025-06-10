package config

import "time"

// RetryPolicy 重试策略配置
type RetryPolicy struct {
	MaxElapsedTime      time.Duration `yaml:"maxElapsedTime"`
	InitialInterval     time.Duration `yaml:"initialInterval"`
	MaxInterval         time.Duration `yaml:"maxInterval"`
	Multiplier          float64       `yaml:"multiplier"`
	RandomizationFactor float64       `yaml:"randomizationFactor"`
}
