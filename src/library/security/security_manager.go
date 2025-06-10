package security

import (
	"VectorSphere/src/library/config"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"google.golang.org/grpc/credentials"
	"io"
	"io/ioutil"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"VectorSphere/src/library/log"
	"github.com/golang-jwt/jwt/v4"
)

// SecurityManager 安全管理器
type SecurityManager struct {
	tlsConfig   *tls.Config
	rbacEnabled bool
	userRoles   sync.Map // map[string][]string 用户角色映射
	permissions sync.Map // map[string][]string 角色权限映射
}

// EnhancedSecurityManager 增强的安全管理器
type EnhancedSecurityManager struct {
	*SecurityManager
	encryptionKey    []byte
	auditLogger      *AuditLogger
	networkPolicy    *NetworkPolicy
	jwtSecret        []byte
	tlsConfig        *tls.Config
	encryptionCipher cipher.AEAD
	mutex            sync.RWMutex
}

// AuditLogger 审计日志记录器
type AuditLogger struct {
	logFile   string
	mutex     sync.Mutex
	enabled   bool
	retention time.Duration
}

// NetworkPolicy 网络安全策略
type NetworkPolicy struct {
	allowedIPs    []net.IPNet
	blockedIPs    []net.IPNet
	rateLimit     map[string]*RateLimiter
	firewallRules []FirewallRule
	mutex         sync.RWMutex
}

// FirewallRule 防火墙规则
type FirewallRule struct {
	ID          string
	Source      string
	Destination string
	Port        int
	Protocol    string
	Action      string // ALLOW, DENY
	Priority    int
}

// RateLimiter 速率限制器
type RateLimiter struct {
	requests    int
	window      time.Duration
	lastReset   time.Time
	currentReqs int
	mutex       sync.Mutex
}

// AuditEvent 审计事件
type AuditEvent struct {
	Timestamp time.Time `json:"timestamp"`
	UserID    string    `json:"user_id"`
	Action    string    `json:"action"`
	Resource  string    `json:"resource"`
	Result    string    `json:"result"`
	IP        string    `json:"ip"`
	UserAgent string    `json:"user_agent"`
	Details   string    `json:"details"`
	RiskLevel string    `json:"risk_level"`
}

// EncryptionConfig 加密配置
type EncryptionConfig struct {
	Algorithm    string `yaml:"algorithm"`
	KeySize      int    `yaml:"keySize"`
	RotationDays int    `yaml:"rotationDays"`
}

// SecurityConfig 安全配置
type SecurityConfig struct {
	TLS        *config.TLSConfig `yaml:"tls"`
	RBAC       *RBACConfig       `yaml:"rbac"`
	Encryption *EncryptionConfig `yaml:"encryption"`
	Audit      *AuditConfig      `yaml:"audit"`
	Network    *NetworkConfig    `yaml:"network"`
	JWT        *JWTConfig        `yaml:"jwt"`
}

// RBACConfig RBAC配置
type RBACConfig struct {
	Enabled     bool                `yaml:"enabled"`
	DefaultRole string              `yaml:"defaultRole"`
	Roles       map[string][]string `yaml:"roles"`
	Policies    []RBACPolicy        `yaml:"policies"`
}

// RBACPolicy RBAC策略
type RBACPolicy struct {
	ID         string   `yaml:"id"`
	Subject    string   `yaml:"subject"`
	Resource   string   `yaml:"resource"`
	Action     string   `yaml:"action"`
	Effect     string   `yaml:"effect"` // ALLOW, DENY
	Conditions []string `yaml:"conditions"`
}

// AuditConfig 审计配置
type AuditConfig struct {
	Enabled   bool          `yaml:"enabled"`
	LogFile   string        `yaml:"logFile"`
	Retention time.Duration `yaml:"retention"`
	LogLevel  string        `yaml:"logLevel"`
	Encrypted bool          `yaml:"encrypted"`
}

// NetworkConfig 网络安全配置
type NetworkConfig struct {
	Firewall   *FirewallConfig  `yaml:"firewall"`
	RateLimit  *RateLimitConfig `yaml:"rateLimit"`
	AllowedIPs []string         `yaml:"allowedIPs"`
	BlockedIPs []string         `yaml:"blockedIPs"`
}

// FirewallConfig 防火墙配置
type FirewallConfig struct {
	Enabled bool           `yaml:"enabled"`
	Rules   []FirewallRule `yaml:"rules"`
}

// RateLimitConfig 速率限制配置
type RateLimitConfig struct {
	Enabled   bool          `yaml:"enabled"`
	Requests  int           `yaml:"requests"`
	Window    time.Duration `yaml:"window"`
	BurstSize int           `yaml:"burstSize"`
}

// JWTConfig JWT配置
type JWTConfig struct {
	Secret     string        `yaml:"secret"`
	Expiration time.Duration `yaml:"expiration"`
	Issuer     string        `yaml:"issuer"`
	Algorithm  string        `yaml:"algorithm"`
}

// NewEnhancedSecurityManager 创建增强的安全管理器
func NewEnhancedSecurityManager(config *SecurityConfig) (*EnhancedSecurityManager, error) {
	// 创建基础安全管理器
	baseSM := NewSecurityManager(nil, config.RBAC.Enabled)

	// 生成加密密钥
	encryptionKey := make([]byte, 32) // AES-256
	if _, err := rand.Read(encryptionKey); err != nil {
		return nil, fmt.Errorf("failed to generate encryption key: %w", err)
	}

	// 创建AES-GCM加密器
	block, err := aes.NewCipher(encryptionKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	// 创建审计日志记录器
	auditLogger := &AuditLogger{
		logFile:   config.Audit.LogFile,
		enabled:   config.Audit.Enabled,
		retention: config.Audit.Retention,
	}

	// 创建网络策略
	networkPolicy := &NetworkPolicy{
		rateLimit: make(map[string]*RateLimiter),
	}

	// 解析允许的IP
	for _, ipStr := range config.Network.AllowedIPs {
		if _, ipNet, err := net.ParseCIDR(ipStr); err == nil {
			networkPolicy.allowedIPs = append(networkPolicy.allowedIPs, *ipNet)
		}
	}

	// 解析阻止的IP
	for _, ipStr := range config.Network.BlockedIPs {
		if _, ipNet, err := net.ParseCIDR(ipStr); err == nil {
			networkPolicy.blockedIPs = append(networkPolicy.blockedIPs, *ipNet)
		}
	}

	// 创建TLS配置
	tlsConfig, err := createTLSConfig(config.TLS)
	if err != nil {
		return nil, fmt.Errorf("failed to create TLS config: %w", err)
	}

	esm := &EnhancedSecurityManager{
		SecurityManager:  baseSM,
		encryptionKey:    encryptionKey,
		auditLogger:      auditLogger,
		networkPolicy:    networkPolicy,
		jwtSecret:        []byte(config.JWT.Secret),
		tlsConfig:        tlsConfig,
		encryptionCipher: gcm,
	}

	// 初始化RBAC策略
	if err := esm.initRBACPolicies(config.RBAC); err != nil {
		return nil, fmt.Errorf("failed to init RBAC policies: %w", err)
	}

	return esm, nil
}

// createTLSConfig 创建TLS配置
func createTLSConfig(config *config.TLSConfig) (*tls.Config, error) {
	if config == nil {
		return nil, nil
	}

	cert, err := tls.LoadX509KeyPair(config.CertFile, config.KeyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load key pair: %w", err)
	}

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
		},
		PreferServerCipherSuites: true,
	}

	if config.CAFile != "" {
		caCert, err := os.ReadFile(config.CAFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read CA file: %w", err)
		}

		caCertPool := x509.NewCertPool()
		caCertPool.AppendCertsFromPEM(caCert)
		tlsConfig.ClientCAs = caCertPool
		tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
	}

	return tlsConfig, nil
}

// NewSecurityManager 创建安全管理器
func NewSecurityManager(tlsConfig *tls.Config, rbacEnabled bool) *SecurityManager {
	sm := &SecurityManager{
		tlsConfig:   tlsConfig,
		rbacEnabled: rbacEnabled,
	}

	// 初始化默认角色和权限
	sm.initDefaultRoles()

	return sm
}

// initDefaultRoles 初始化默认角色
func (sm *SecurityManager) initDefaultRoles() {
	// 管理员角色
	sm.permissions.Store("admin", []string{
		"service:register",
		"service:unregister",
		"config:read",
		"config:write",
		"lock:acquire",
		"lock:release",
		"election:participate",
	})

	// 服务角色
	sm.permissions.Store("service", []string{
		"service:register",
		"service:unregister",
		"config:read",
		"lock:acquire",
		"lock:release",
	})

	// 只读角色
	sm.permissions.Store("readonly", []string{
		"config:read",
	})
}

// AuthenticateAndAuthorize 认证和授权
func (sm *SecurityManager) AuthenticateAndAuthorize(ctx context.Context, token string, requiredPermission string) error {
	if !sm.rbacEnabled {
		return nil // RBAC未启用，跳过检查
	}

	// 解析token获取用户信息（这里简化处理）
	userID, err := sm.parseToken(token)
	if err != nil {
		return fmt.Errorf("authentication failed: %w", err)
	}

	// 获取用户角色
	rolesInterface, exists := sm.userRoles.Load(userID)
	if !exists {
		return fmt.Errorf("user %s not found", userID)
	}

	userRoles := rolesInterface.([]string)

	// 检查权限
	for _, role := range userRoles {
		if permissionsInterface, exists := sm.permissions.Load(role); exists {
			permissions := permissionsInterface.([]string)
			for _, permission := range permissions {
				if permission == requiredPermission {
					return nil // 有权限
				}
			}
		}
	}

	return fmt.Errorf("insufficient permissions for %s", requiredPermission)
}

// parseToken 解析token（简化实现）
func (sm *SecurityManager) parseToken(token string) (string, error) {
	// 这里应该实现JWT token解析或其他认证机制
	// 简化处理，直接返回token作为用户ID
	if token == "" {
		return "", fmt.Errorf("empty token")
	}
	return token, nil
}

// AddUser 添加用户
func (sm *SecurityManager) AddUser(userID string, roles []string) {
	sm.userRoles.Store(userID, roles)
}

// AddRole 添加角色
func (sm *SecurityManager) AddRole(role string, permissions []string) {
	sm.permissions.Store(role, permissions)
}

// GetTLSConfig 获取TLS配置
func (sm *SecurityManager) GetTLSConfig() *tls.Config {
	return sm.tlsConfig
}

// initRBACPolicies 初始化RBAC策略
func (esm *EnhancedSecurityManager) initRBACPolicies(config *RBACConfig) error {
	if !config.Enabled {
		return nil
	}

	// 添加角色和权限
	for role, permissions := range config.Roles {
		esm.AddRole(role, permissions)
	}

	return nil
}

// EncryptData 加密数据
func (esm *EnhancedSecurityManager) EncryptData(data []byte) ([]byte, error) {
	esm.mutex.RLock()
	defer esm.mutex.RUnlock()

	nonce := make([]byte, esm.encryptionCipher.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	ciphertext := esm.encryptionCipher.Seal(nonce, nonce, data, nil)
	return ciphertext, nil
}

// DecryptData 解密数据
func (esm *EnhancedSecurityManager) DecryptData(encryptedData []byte) ([]byte, error) {
	esm.mutex.RLock()
	defer esm.mutex.RUnlock()

	nonceSize := esm.encryptionCipher.NonceSize()
	if len(encryptedData) < nonceSize {
		return nil, fmt.Errorf("ciphertext too short")
	}

	nonce, ciphertext := encryptedData[:nonceSize], encryptedData[nonceSize:]
	plaintext, err := esm.encryptionCipher.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt: %w", err)
	}

	return plaintext, nil
}

// GenerateJWT 生成JWT令牌
func (esm *EnhancedSecurityManager) GenerateJWT(userID string, roles []string) (string, error) {
	claims := jwt.MapClaims{
		"user_id": userID,
		"roles":   roles,
		"exp":     time.Now().Add(24 * time.Hour).Unix(),
		"iat":     time.Now().Unix(),
		"iss":     "VectorSphere",
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString(esm.jwtSecret)
	if err != nil {
		return "", fmt.Errorf("failed to sign token: %w", err)
	}

	return tokenString, nil
}

// ValidateJWT 验证JWT令牌
func (esm *EnhancedSecurityManager) ValidateJWT(tokenString string) (*jwt.MapClaims, error) {
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return esm.jwtSecret, nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to parse token: %w", err)
	}

	if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
		return &claims, nil
	}

	return nil, fmt.Errorf("invalid token")
}

// CheckNetworkAccess 检查网络访问权限
func (esm *EnhancedSecurityManager) CheckNetworkAccess(clientIP string) error {
	esm.networkPolicy.mutex.RLock()
	defer esm.networkPolicy.mutex.RUnlock()

	ip := net.ParseIP(clientIP)
	if ip == nil {
		return fmt.Errorf("invalid IP address: %s", clientIP)
	}

	// 检查是否在阻止列表中
	for _, blockedNet := range esm.networkPolicy.blockedIPs {
		if blockedNet.Contains(ip) {
			return fmt.Errorf("IP %s is blocked", clientIP)
		}
	}

	// 如果有允许列表，检查是否在允许列表中
	if len(esm.networkPolicy.allowedIPs) > 0 {
		allowed := false
		for _, allowedNet := range esm.networkPolicy.allowedIPs {
			if allowedNet.Contains(ip) {
				allowed = true
				break
			}
		}
		if !allowed {
			return fmt.Errorf("IP %s is not in allowed list", clientIP)
		}
	}

	return nil
}

// CheckRateLimit 检查速率限制
func (esm *EnhancedSecurityManager) CheckRateLimit(clientIP string, requests int, window time.Duration) error {
	esm.networkPolicy.mutex.Lock()
	defer esm.networkPolicy.mutex.Unlock()

	limiter, exists := esm.networkPolicy.rateLimit[clientIP]
	if !exists {
		limiter = &RateLimiter{
			requests:    requests,
			window:      window,
			lastReset:   time.Now(),
			currentReqs: 0,
		}
		esm.networkPolicy.rateLimit[clientIP] = limiter
	}

	limiter.mutex.Lock()
	defer limiter.mutex.Unlock()

	now := time.Now()
	if now.Sub(limiter.lastReset) > limiter.window {
		limiter.currentReqs = 0
		limiter.lastReset = now
	}

	if limiter.currentReqs >= limiter.requests {
		return fmt.Errorf("rate limit exceeded for IP %s", clientIP)
	}

	limiter.currentReqs++
	return nil
}

// LogAuditEvent 记录审计事件
func (esm *EnhancedSecurityManager) LogAuditEvent(event *AuditEvent) error {
	if !esm.auditLogger.enabled {
		return nil
	}

	esm.auditLogger.mutex.Lock()
	defer esm.auditLogger.mutex.Unlock()

	eventJSON, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal audit event: %w", err)
	}

	// 如果启用了加密，加密审计日志
	var logData []byte
	if esm.auditLogger.enabled {
		encryptedData, err := esm.EncryptData(eventJSON)
		if err != nil {
			return fmt.Errorf("failed to encrypt audit log: %w", err)
		}
		logData = []byte(base64.StdEncoding.EncodeToString(encryptedData))
	} else {
		logData = eventJSON
	}

	// 写入审计日志文件
	log.Info("[AUDIT] %s", string(logData))

	return nil
}

// EnhancedAuthenticateAndAuthorize 增强的认证和授权
func (esm *EnhancedSecurityManager) EnhancedAuthenticateAndAuthorize(ctx context.Context, token, clientIP, userAgent, action, resource string) error {
	// 检查网络访问权限
	if err := esm.CheckNetworkAccess(clientIP); err != nil {
		esm.LogAuditEvent(&AuditEvent{
			Timestamp: time.Now(),
			Action:    action,
			Resource:  resource,
			Result:    "DENIED",
			IP:        clientIP,
			UserAgent: userAgent,
			Details:   fmt.Sprintf("Network access denied: %v", err),
			RiskLevel: "HIGH",
		})
		return fmt.Errorf("network access denied: %w", err)
	}

	// 检查速率限制
	if err := esm.CheckRateLimit(clientIP, 100, time.Minute); err != nil {
		esm.LogAuditEvent(&AuditEvent{
			Timestamp: time.Now(),
			Action:    action,
			Resource:  resource,
			Result:    "DENIED",
			IP:        clientIP,
			UserAgent: userAgent,
			Details:   fmt.Sprintf("Rate limit exceeded: %v", err),
			RiskLevel: "MEDIUM",
		})
		return fmt.Errorf("rate limit exceeded: %w", err)
	}

	// 验证JWT令牌
	claims, err := esm.ValidateJWT(strings.TrimPrefix(token, "Bearer "))
	if err != nil {
		esm.LogAuditEvent(&AuditEvent{
			Timestamp: time.Now(),
			Action:    action,
			Resource:  resource,
			Result:    "DENIED",
			IP:        clientIP,
			UserAgent: userAgent,
			Details:   fmt.Sprintf("JWT validation failed: %v", err),
			RiskLevel: "HIGH",
		})
		return fmt.Errorf("authentication failed: %w", err)
	}

	userID := (*claims)["user_id"].(string)
	roles := (*claims)["roles"].([]interface{})

	// 检查授权
	if err := esm.checkAuthorization(userID, roles, action, resource); err != nil {
		esm.LogAuditEvent(&AuditEvent{
			Timestamp: time.Now(),
			UserID:    userID,
			Action:    action,
			Resource:  resource,
			Result:    "DENIED",
			IP:        clientIP,
			UserAgent: userAgent,
			Details:   fmt.Sprintf("Authorization failed: %v", err),
			RiskLevel: "MEDIUM",
		})
		return fmt.Errorf("authorization failed: %w", err)
	}

	// 记录成功的访问
	esm.LogAuditEvent(&AuditEvent{
		Timestamp: time.Now(),
		UserID:    userID,
		Action:    action,
		Resource:  resource,
		Result:    "ALLOWED",
		IP:        clientIP,
		UserAgent: userAgent,
		Details:   "Access granted",
		RiskLevel: "LOW",
	})

	return nil
}

// checkAuthorization 检查授权
func (esm *EnhancedSecurityManager) checkAuthorization(userID string, roles []interface{}, action, resource string) error {
	if !esm.rbacEnabled {
		return nil
	}

	// 转换角色列表
	userRoles := make([]string, len(roles))
	for i, role := range roles {
		userRoles[i] = role.(string)
	}

	// 检查权限
	for _, role := range userRoles {
		if permissionsInterface, exists := esm.permissions.Load(role); exists {
			permissions := permissionsInterface.([]string)
			for _, permission := range permissions {
				if esm.matchPermission(permission, action, resource) {
					return nil
				}
			}
		}
	}

	return fmt.Errorf("insufficient permissions for %s on %s", action, resource)
}

// matchPermission 匹配权限
func (esm *EnhancedSecurityManager) matchPermission(permission, action, resource string) bool {
	// 简单的权限匹配逻辑，可以扩展为更复杂的模式匹配
	parts := strings.Split(permission, ":")
	if len(parts) != 2 {
		return false
	}

	resourcePart, actionPart := parts[0], parts[1]

	// 支持通配符
	if resourcePart == "*" || resourcePart == resource {
		if actionPart == "*" || actionPart == action {
			return true
		}
	}

	return false
}

// GetTLSConfig 获取TLS配置
func (esm *EnhancedSecurityManager) GetTLSConfig() *tls.Config {
	return esm.tlsConfig
}

// RotateEncryptionKey 轮换加密密钥
func (esm *EnhancedSecurityManager) RotateEncryptionKey() error {
	esm.mutex.Lock()
	defer esm.mutex.Unlock()

	// 生成新的加密密钥
	newKey := make([]byte, 32)
	if _, err := rand.Read(newKey); err != nil {
		return fmt.Errorf("failed to generate new encryption key: %w", err)
	}

	// 创建新的加密器
	block, err := aes.NewCipher(newKey)
	if err != nil {
		return fmt.Errorf("failed to create new cipher: %w", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return fmt.Errorf("failed to create new GCM: %w", err)
	}

	// 更新密钥和加密器
	esm.encryptionKey = newKey
	esm.encryptionCipher = gcm

	log.Info("Encryption key rotated successfully")
	return nil
}

// LoadClientTLS 加载客户端TLS配置
func LoadClientTLS(tlsConfig *config.TLSConfig) (credentials.TransportCredentials, error) {
	if tlsConfig == nil {
		return nil, fmt.Errorf("TLS config is nil")
	}

	// 如果只有CA文件，创建仅验证服务端的TLS配置
	if tlsConfig.CertFile == "" && tlsConfig.KeyFile == "" && tlsConfig.CAFile != "" {
		caCert, err := ioutil.ReadFile(tlsConfig.CAFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read CA file: %w", err)
		}
		caCertPool := x509.NewCertPool()
		if !caCertPool.AppendCertsFromPEM(caCert) {
			return nil, fmt.Errorf("failed to append CA certificate")
		}
		tlsConf := &tls.Config{
			RootCAs:    caCertPool,
			ServerName: tlsConfig.ServerName,
		}
		return credentials.NewTLS(tlsConf), nil
	}

	// 如果有客户端证书和私钥，创建双向TLS配置
	if tlsConfig.CertFile != "" && tlsConfig.KeyFile != "" {
		cert, err := tls.LoadX509KeyPair(tlsConfig.CertFile, tlsConfig.KeyFile)
		if err != nil {
			return nil, fmt.Errorf("failed to load client certificate: %w", err)
		}

		tlsConf := &tls.Config{
			Certificates: []tls.Certificate{cert},
			ServerName:   tlsConfig.ServerName,
		}

		// 如果有CA文件，添加到根证书池
		if tlsConfig.CAFile != "" {
			caCert, err := ioutil.ReadFile(tlsConfig.CAFile)
			if err != nil {
				return nil, fmt.Errorf("failed to read CA file: %w", err)
			}
			caCertPool := x509.NewCertPool()
			if !caCertPool.AppendCertsFromPEM(caCert) {
				return nil, fmt.Errorf("failed to append CA certificate")
			}
			tlsConf.RootCAs = caCertPool
		}

		return credentials.NewTLS(tlsConf), nil
	}

	return nil, fmt.Errorf("invalid TLS configuration: missing certificate files")
}
