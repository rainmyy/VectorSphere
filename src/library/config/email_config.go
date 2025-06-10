package config

import "time"

type EmailConfig struct {
	Host    string `yaml:"host"`
	Port    int    `yaml:"port"`
	Enabled bool   `yaml:"enabled"`

	Username string   `yaml:"username"` // SMTP用户名
	Password string   // SMTP 密码
	From     string   `yaml:"from"`   // 发件人地址
	To       []string `yaml:"to"`     // 收件人地址，多个收件人用逗号分隔
	UseTLS   bool     `yaml:"useTLS"` // 是否使用TLS

	AllowInsecure bool          `yaml:"allowInsecure"` // TLS失败时是否允许回退到非加密连接
	Timeout       time.Duration `yaml:"timeout"`       // 连接和发送操作的超时时间
	MaxRetries    int           `yaml:"maxRetries"`    // 最大重试次数
	RetryInterval time.Duration `yaml:"retryInterval"` // 重试间隔
}
