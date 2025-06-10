package config

// TLSConfig TLS配置
type TLSConfig struct {
	CertFile   string `yaml:"certFile"`
	KeyFile    string `yaml:"keyFile"`
	CAFile     string `yaml:"caFile"`
	ServerName string `yaml:"serverName"` // 用于验证服务端证书的 CN
}
