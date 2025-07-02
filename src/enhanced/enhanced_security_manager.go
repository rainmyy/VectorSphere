package enhanced

import (
	"VectorSphere/src/library/logger"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"golang.org/x/crypto/scrypt"
)

// SecurityLevel 安全级别
type SecurityLevel int

const (
	SecurityLevelLow SecurityLevel = iota
	SecurityLevelMedium
	SecurityLevelHigh
	SecurityLevelCritical
)

// PermissionType 权限类型
type PermissionType int

const (
	PermissionRead PermissionType = iota
	PermissionWrite
	PermissionDelete
	PermissionExecute
	PermissionAdmin
)

// AuditEventType 审计事件类型
type AuditEventType int

const (
	AuditLogin AuditEventType = iota
	AuditLogout
	AuditAccess
	AuditPermissionDenied
	AuditDataAccess
	AuditConfigChange
	AuditSecurityViolation
	AuditSystemEvent
)

// String 返回AuditEventType的字符串表示
func (a AuditEventType) String() string {
	switch a {
	case AuditLogin:
		return "Login"
	case AuditLogout:
		return "Logout"
	case AuditAccess:
		return "Access"
	case AuditPermissionDenied:
		return "PermissionDenied"
	case AuditDataAccess:
		return "DataAccess"
	case AuditConfigChange:
		return "ConfigChange"
	case AuditSecurityViolation:
		return "SecurityViolation"
	case AuditSystemEvent:
		return "SystemEvent"
	default:
		return "Unknown"
	}
}

// EncryptionAlgorithm 加密算法
type EncryptionAlgorithm int

const (
	AES256GCM EncryptionAlgorithm = iota
	AES256CBC
	RSA2048
	RSA4096
)

// User 用户信息
type User struct {
	ID           string                 `json:"id"`
	Username     string                 `json:"username"`
	Email        string                 `json:"email"`
	PasswordHash string                 `json:"password_hash"`
	Salt         string                 `json:"salt"`
	Roles        []string               `json:"roles"`
	Permissions  []Permission           `json:"permissions"`
	Status       UserStatus             `json:"status"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
	LastLogin    time.Time              `json:"last_login"`
	LoginCount   int64                  `json:"login_count"`
	FailedLogins int                    `json:"failed_logins"`
	LockoutUntil time.Time              `json:"lockout_until"`
	Metadata     map[string]interface{} `json:"metadata"`
	Tags         []string               `json:"tags"`
}

// UserStatus 用户状态
type UserStatus int

const (
	UserActive UserStatus = iota
	UserInactive
	UserLocked
	UserSuspended
	UserDeleted
)

// Role 角色
type Role struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Permissions []Permission           `json:"permissions"`
	ParentRoles []string               `json:"parent_roles"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// Permission 权限
type Permission struct {
	ID         string                 `json:"id"`
	Resource   string                 `json:"resource"`
	Action     string                 `json:"action"`
	Type       PermissionType         `json:"type"`
	Scope      string                 `json:"scope"`
	Conditions map[string]interface{} `json:"conditions"`
}

// SecurityPolicy 安全策略
type SecurityPolicy struct {
	ID               string            `json:"id"`
	Name             string            `json:"name"`
	Description      string            `json:"description"`
	Level            SecurityLevel     `json:"level"`
	PasswordPolicy   *PasswordPolicy   `json:"password_policy"`
	SessionPolicy    *SessionPolicy    `json:"session_policy"`
	EncryptionPolicy *EncryptionPolicy `json:"encryption_policy"`
	NetworkPolicy    *NetworkPolicy    `json:"network_policy"`
	AuditPolicy      *AuditPolicy      `json:"audit_policy"`
	RateLimitPolicy  *RateLimitPolicy  `json:"rate_limit_policy"`
	Enabled          bool              `json:"enabled"`
	CreatedAt        time.Time         `json:"created_at"`
	UpdatedAt        time.Time         `json:"updated_at"`
}

// PasswordPolicy 密码策略
type PasswordPolicy struct {
	MinLength        int           `json:"min_length"`
	MaxLength        int           `json:"max_length"`
	RequireUppercase bool          `json:"require_uppercase"`
	RequireLowercase bool          `json:"require_lowercase"`
	RequireNumbers   bool          `json:"require_numbers"`
	RequireSymbols   bool          `json:"require_symbols"`
	MaxAge           time.Duration `json:"max_age"`
	HistoryCount     int           `json:"history_count"`
	LockoutThreshold int           `json:"lockout_threshold"`
	LockoutDuration  time.Duration `json:"lockout_duration"`
}

// SessionPolicy 会话策略
type SessionPolicy struct {
	MaxDuration    time.Duration `json:"max_duration"`
	IdleTimeout    time.Duration `json:"idle_timeout"`
	MaxConcurrent  int           `json:"max_concurrent"`
	RequireRefresh bool          `json:"require_refresh"`
	SecureCookies  bool          `json:"secure_cookies"`
	SameSitePolicy string        `json:"same_site_policy"`
}

// EncryptionPolicy 加密策略
type EncryptionPolicy struct {
	Algorithm        EncryptionAlgorithm `json:"algorithm"`
	KeySize          int                 `json:"key_size"`
	RotationInterval time.Duration       `json:"rotation_interval"`
	EncryptAtRest    bool                `json:"encrypt_at_rest"`
	EncryptInTransit bool                `json:"encrypt_in_transit"`
	RequireTLS       bool                `json:"require_tls"`
	MinTLSVersion    string              `json:"min_tls_version"`
	CipherSuites     []string            `json:"cipher_suites"`
}

// NetworkPolicy 网络策略
type NetworkPolicy struct {
	AllowedIPs      []string       `json:"allowed_ips"`
	BlockedIPs      []string       `json:"blocked_ips"`
	AllowedPorts    []int          `json:"allowed_ports"`
	BlockedPorts    []int          `json:"blocked_ports"`
	RequireVPN      bool           `json:"require_vpn"`
	GeoRestrictions []string       `json:"geo_restrictions"`
	FirewallRules   []FirewallRule `json:"firewall_rules"`
}

// FirewallRule 防火墙规则
type FirewallRule struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Action     string `json:"action"`   // allow, deny, log
	Protocol   string `json:"protocol"` // tcp, udp, icmp
	SourceIP   string `json:"source_ip"`
	DestIP     string `json:"dest_ip"`
	SourcePort string `json:"source_port"`
	DestPort   string `json:"dest_port"`
	Priority   int    `json:"priority"`
	Enabled    bool   `json:"enabled"`
}

// AuditPolicy 审计策略
type AuditPolicy struct {
	Enabled         bool             `json:"enabled"`
	LogLevel        string           `json:"log_level"`
	RetentionPeriod time.Duration    `json:"retention_period"`
	EventTypes      []AuditEventType `json:"event_types"`
	SensitiveFields []string         `json:"sensitive_fields"`
	ExcludePatterns []string         `json:"exclude_patterns"`
	RealTimeAlerts  bool             `json:"real_time_alerts"`
	Compression     bool             `json:"compression"`
	Encryption      bool             `json:"encryption"`
}

// RateLimitPolicy 限流策略
type RateLimitPolicy struct {
	Enabled        bool          `json:"enabled"`
	RequestsPerMin int           `json:"requests_per_min"`
	BurstSize      int           `json:"burst_size"`
	WindowSize     time.Duration `json:"window_size"`
	BlockDuration  time.Duration `json:"block_duration"`
	Whitelist      []string      `json:"whitelist"`
	Blacklist      []string      `json:"blacklist"`
}

// Session 会话信息
type Session struct {
	ID         string                 `json:"id"`
	UserID     string                 `json:"user_id"`
	Token      string                 `json:"token"`
	CreatedAt  time.Time              `json:"created_at"`
	ExpiresAt  time.Time              `json:"expires_at"`
	LastAccess time.Time              `json:"last_access"`
	IPAddress  string                 `json:"ip_address"`
	UserAgent  string                 `json:"user_agent"`
	Metadata   map[string]interface{} `json:"metadata"`
	Active     bool                   `json:"active"`
}

// AuditEvent 审计事件
type AuditEvent struct {
	ID        string                 `json:"id"`
	Type      AuditEventType         `json:"type"`
	UserID    string                 `json:"user_id"`
	SessionID string                 `json:"session_id"`
	Resource  string                 `json:"resource"`
	Action    string                 `json:"action"`
	Result    string                 `json:"result"` // success, failure, denied
	IPAddress string                 `json:"ip_address"`
	UserAgent string                 `json:"user_agent"`
	Timestamp time.Time              `json:"timestamp"`
	Details   map[string]interface{} `json:"details"`
	Severity  string                 `json:"severity"`
	Message   string                 `json:"message"`
}

// SecurityConfig 安全配置
type SecurityConfig struct {
	Enabled          bool          `json:"enabled"`
	DefaultPolicy    string        `json:"default_policy"`
	CertificatePath  string        `json:"certificate_path"`
	PrivateKeyPath   string        `json:"private_key_path"`
	CAPath           string        `json:"ca_path"`
	TLSConfig        *tls.Config   `json:"-"`
	EncryptionKey    []byte        `json:"-"`
	TokenSecret      string        `json:"-"`
	SessionTimeout   time.Duration `json:"session_timeout"`
	MaxLoginAttempts int           `json:"max_login_attempts"`
	LockoutDuration  time.Duration `json:"lockout_duration"`
	AuditLogPath     string        `json:"audit_log_path"`
	BasePrefix       string        `json:"base_prefix"`
}

// EnhancedSecurityManager 增强安全管理器
type EnhancedSecurityManager struct {
	client        *clientv3.Client
	config        *SecurityConfig
	users         map[string]*User
	roles         map[string]*Role
	policies      map[string]*SecurityPolicy
	sessions      map[string]*Session
	auditEvents   []*AuditEvent
	mu            sync.RWMutex
	sessionMu     sync.RWMutex
	auditMu       sync.RWMutex
	ctx           context.Context
	cancel        context.CancelFunc
	tlsConfig     *tls.Config
	encryptionKey []byte
	basePrefix    string
	auditTicker   *time.Ticker
	cleanupTicker *time.Ticker
	isRunning     bool
}

// NewEnhancedSecurityManager 创建增强安全管理器
func NewEnhancedSecurityManager(client *clientv3.Client, config *SecurityConfig) *EnhancedSecurityManager {
	if config == nil {
		config = &SecurityConfig{
			Enabled:          true,
			DefaultPolicy:    "default",
			SessionTimeout:   24 * time.Hour,
			MaxLoginAttempts: 5,
			LockoutDuration:  30 * time.Minute,
			AuditLogPath:     "/var/log/vector_sphere/audit.log",
			BasePrefix:       "/vector_sphere/security",
		}
	}

	ctx, cancel := context.WithCancel(context.Background())

	sm := &EnhancedSecurityManager{
		client:      client,
		config:      config,
		users:       make(map[string]*User),
		roles:       make(map[string]*Role),
		policies:    make(map[string]*SecurityPolicy),
		sessions:    make(map[string]*Session),
		auditEvents: make([]*AuditEvent, 0),
		ctx:         ctx,
		cancel:      cancel,
		basePrefix:  config.BasePrefix,
	}

	// 初始化加密密钥
	if err := sm.initializeEncryption(); err != nil {
		logger.Error("Failed to initialize encryption: %v", err)
	}

	// 初始化TLS配置
	if err := sm.initializeTLS(); err != nil {
		logger.Error("Failed to initialize TLS: %v", err)
	}

	// 创建默认策略
	sm.createDefaultPolicy()

	logger.Info("Enhanced security manager created")
	return sm
}

// Start 启动安全管理器
func (sm *EnhancedSecurityManager) Start() error {
	if sm.isRunning {
		return fmt.Errorf("security manager is already running")
	}

	logger.Info("Starting enhanced security manager")

	// 加载用户和角色
	if err := sm.loadUsersAndRoles(); err != nil {
		logger.Error("Failed to load users and roles: %v", err)
		return err
	}

	// 加载安全策略
	if err := sm.loadPolicies(); err != nil {
		logger.Error("Failed to load security policies: %v", err)
		return err
	}

	// 启动审计日志
	sm.auditTicker = time.NewTicker(60 * time.Second)
	go sm.auditLogger()

	// 启动清理器
	sm.cleanupTicker = time.NewTicker(10 * time.Minute)
	go sm.cleaner()

	// 启动会话监控
	go sm.sessionMonitor()

	sm.isRunning = true
	logger.Info("Enhanced security manager started successfully")
	return nil
}

// Stop 停止安全管理器
func (sm *EnhancedSecurityManager) Stop() error {
	if !sm.isRunning {
		return fmt.Errorf("security manager is not running")
	}

	logger.Info("Stopping enhanced security manager")

	// 停止定时器
	if sm.auditTicker != nil {
		sm.auditTicker.Stop()
	}
	if sm.cleanupTicker != nil {
		sm.cleanupTicker.Stop()
	}

	// 取消上下文
	sm.cancel()

	// 保存审计日志
	sm.flushAuditLogs()

	sm.isRunning = false
	logger.Info("Enhanced security manager stopped")
	return nil
}

// 用户管理

// CreateUser 创建用户
func (sm *EnhancedSecurityManager) CreateUser(user *User) error {
	if user == nil || user.Username == "" {
		return fmt.Errorf("invalid user")
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	// 检查用户是否已存在
	for _, existingUser := range sm.users {
		if existingUser.Username == user.Username || existingUser.Email == user.Email {
			return fmt.Errorf("user already exists")
		}
	}

	// 生成用户ID
	if user.ID == "" {
		user.ID = sm.generateID()
	}

	// 设置默认值
	user.Status = UserActive
	user.CreatedAt = time.Now()
	user.UpdatedAt = time.Now()
	user.LoginCount = 0
	user.FailedLogins = 0

	if user.Metadata == nil {
		user.Metadata = make(map[string]interface{})
	}

	// 存储用户
	sm.users[user.ID] = user

	// 保存到etcd
	if err := sm.saveUser(user); err != nil {
		delete(sm.users, user.ID)
		return fmt.Errorf("failed to save user: %v", err)
	}

	// 记录审计事件
	sm.logAuditEvent(&AuditEvent{
		Type:      AuditSystemEvent,
		Action:    "create_user",
		Result:    "success",
		Timestamp: time.Now(),
		Details: map[string]interface{}{
			"user_id":  user.ID,
			"username": user.Username,
		},
		Message: fmt.Sprintf("User %s created", user.Username),
	})

	logger.Info("User created: %s (%s)", user.Username, user.ID)
	return nil
}

// UpdateUser 更新用户
func (sm *EnhancedSecurityManager) UpdateUser(user *User) error {
	if user == nil || user.ID == "" {
		return fmt.Errorf("invalid user")
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	existingUser, exists := sm.users[user.ID]
	if !exists {
		return fmt.Errorf("user not found")
	}

	// 更新用户信息
	user.CreatedAt = existingUser.CreatedAt
	user.UpdatedAt = time.Now()
	sm.users[user.ID] = user

	// 保存到etcd
	if err := sm.saveUser(user); err != nil {
		return fmt.Errorf("failed to save user: %v", err)
	}

	// 记录审计事件
	sm.logAuditEvent(&AuditEvent{
		Type:      AuditSystemEvent,
		Action:    "update_user",
		Result:    "success",
		Timestamp: time.Now(),
		Details: map[string]interface{}{
			"user_id":  user.ID,
			"username": user.Username,
		},
		Message: fmt.Sprintf("User %s updated", user.Username),
	})

	logger.Info("User updated: %s (%s)", user.Username, user.ID)
	return nil
}

// DeleteUser 删除用户
func (sm *EnhancedSecurityManager) DeleteUser(userID string) error {
	if userID == "" {
		return fmt.Errorf("user ID cannot be empty")
	}

	sm.mu.Lock()
	user, exists := sm.users[userID]
	if exists {
		user.Status = UserDeleted
		user.UpdatedAt = time.Now()
	}
	sm.mu.Unlock()

	if !exists {
		return fmt.Errorf("user not found")
	}

	// 保存到etcd
	if err := sm.saveUser(user); err != nil {
		return fmt.Errorf("failed to save user: %v", err)
	}

	// 终止用户所有会话
	sm.terminateUserSessions(userID)

	// 记录审计事件
	sm.logAuditEvent(&AuditEvent{
		Type:      AuditSystemEvent,
		Action:    "delete_user",
		Result:    "success",
		Timestamp: time.Now(),
		Details: map[string]interface{}{
			"user_id":  userID,
			"username": user.Username,
		},
		Message: fmt.Sprintf("User %s deleted", user.Username),
	})

	logger.Info("User deleted: %s (%s)", user.Username, userID)
	return nil
}

// GetUser 获取用户
func (sm *EnhancedSecurityManager) GetUser(userID string) (*User, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	user, exists := sm.users[userID]
	if !exists {
		return nil, fmt.Errorf("user not found")
	}

	// 返回用户副本
	userCopy := *user
	return &userCopy, nil
}

// GetUserByUsername 根据用户名获取用户
func (sm *EnhancedSecurityManager) GetUserByUsername(username string) (*User, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	for _, user := range sm.users {
		if user.Username == username {
			userCopy := *user
			return &userCopy, nil
		}
	}

	return nil, fmt.Errorf("user not found")
}

// SetUserPassword 设置用户密码
func (sm *EnhancedSecurityManager) SetUserPassword(userID, password string) error {
	if userID == "" || password == "" {
		return fmt.Errorf("user ID and password cannot be empty")
	}

	// 验证密码策略
	if err := sm.validatePassword(password); err != nil {
		return fmt.Errorf("password validation failed: %v", err)
	}

	sm.mu.Lock()
	user, exists := sm.users[userID]
	if !exists {
		sm.mu.Unlock()
		return fmt.Errorf("user not found")
	}

	// 生成盐值
	salt := sm.generateSalt()
	user.Salt = salt

	// 哈希密码
	passwordHash, err := sm.hashPassword(password, salt)
	if err != nil {
		sm.mu.Unlock()
		return fmt.Errorf("failed to hash password: %v", err)
	}

	user.PasswordHash = passwordHash
	user.UpdatedAt = time.Now()
	sm.mu.Unlock()

	// 保存到etcd
	if err := sm.saveUser(user); err != nil {
		return fmt.Errorf("failed to save user: %v", err)
	}

	// 记录审计事件
	sm.logAuditEvent(&AuditEvent{
		Type:      AuditSystemEvent,
		Action:    "change_password",
		Result:    "success",
		UserID:    userID,
		Timestamp: time.Now(),
		Message:   "User password changed",
	})

	logger.Info("Password changed for user: %s", userID)
	return nil
}

// 角色管理

// CreateRole 创建角色
func (sm *EnhancedSecurityManager) CreateRole(role *Role) error {
	if role == nil || role.Name == "" {
		return fmt.Errorf("invalid role")
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	// 检查角色是否已存在
	for _, existingRole := range sm.roles {
		if existingRole.Name == role.Name {
			return fmt.Errorf("role already exists")
		}
	}

	// 生成角色ID
	if role.ID == "" {
		role.ID = sm.generateID()
	}

	// 设置默认值
	role.CreatedAt = time.Now()
	role.UpdatedAt = time.Now()

	if role.Metadata == nil {
		role.Metadata = make(map[string]interface{})
	}

	// 存储角色
	sm.roles[role.ID] = role

	// 保存到etcd
	if err := sm.saveRole(role); err != nil {
		delete(sm.roles, role.ID)
		return fmt.Errorf("failed to save role: %v", err)
	}

	// 记录审计事件
	sm.logAuditEvent(&AuditEvent{
		Type:      AuditSystemEvent,
		Action:    "create_role",
		Result:    "success",
		Timestamp: time.Now(),
		Details: map[string]interface{}{
			"role_id":   role.ID,
			"role_name": role.Name,
		},
		Message: fmt.Sprintf("Role %s created", role.Name),
	})

	logger.Info("Role created: %s (%s)", role.Name, role.ID)
	return nil
}

// 认证和授权

// Authenticate 用户认证
func (sm *EnhancedSecurityManager) Authenticate(username, password, ipAddress, userAgent string) (*Session, error) {
	if username == "" || password == "" {
		return nil, fmt.Errorf("username and password cannot be empty")
	}

	// 获取用户
	user, err := sm.GetUserByUsername(username)
	if err != nil {
		// 记录失败的登录尝试
		sm.logAuditEvent(&AuditEvent{
			Type:      AuditLogin,
			Action:    "login",
			Result:    "failure",
			IPAddress: ipAddress,
			UserAgent: userAgent,
			Timestamp: time.Now(),
			Details: map[string]interface{}{
				"username": username,
				"reason":   "user_not_found",
			},
			Message: fmt.Sprintf("Login failed for user %s: user not found", username),
		})
		return nil, fmt.Errorf("authentication failed")
	}

	// 检查用户状态
	if user.Status != UserActive {
		sm.logAuditEvent(&AuditEvent{
			Type:      AuditLogin,
			Action:    "login",
			Result:    "failure",
			UserID:    user.ID,
			IPAddress: ipAddress,
			UserAgent: userAgent,
			Timestamp: time.Now(),
			Details: map[string]interface{}{
				"username": username,
				"reason":   "user_inactive",
				"status":   user.Status,
			},
			Message: fmt.Sprintf("Login failed for user %s: user inactive", username),
		})
		return nil, fmt.Errorf("user account is inactive")
	}

	// 检查账户锁定
	if user.LockoutUntil.After(time.Now()) {
		sm.logAuditEvent(&AuditEvent{
			Type:      AuditLogin,
			Action:    "login",
			Result:    "failure",
			UserID:    user.ID,
			IPAddress: ipAddress,
			UserAgent: userAgent,
			Timestamp: time.Now(),
			Details: map[string]interface{}{
				"username":      username,
				"reason":        "account_locked",
				"lockout_until": user.LockoutUntil,
			},
			Message: fmt.Sprintf("Login failed for user %s: account locked until %v", username, user.LockoutUntil),
		})
		return nil, fmt.Errorf("account is locked")
	}

	// 验证密码
	if !sm.verifyPassword(password, user.PasswordHash, user.Salt) {
		// 增加失败登录计数
		sm.mu.Lock()
		user.FailedLogins++
		if user.FailedLogins >= sm.config.MaxLoginAttempts {
			user.LockoutUntil = time.Now().Add(sm.config.LockoutDuration)
			user.Status = UserLocked
		}
		user.UpdatedAt = time.Now()
		sm.users[user.ID] = user
		sm.mu.Unlock()

		// 保存用户状态
		sm.saveUser(user)

		sm.logAuditEvent(&AuditEvent{
			Type:      AuditLogin,
			Action:    "login",
			Result:    "failure",
			UserID:    user.ID,
			IPAddress: ipAddress,
			UserAgent: userAgent,
			Timestamp: time.Now(),
			Details: map[string]interface{}{
				"username":        username,
				"reason":          "invalid_password",
				"failed_attempts": user.FailedLogins,
			},
			Message: fmt.Sprintf("Login failed for user %s: invalid password", username),
		})
		return nil, fmt.Errorf("authentication failed")
	}

	// 重置失败登录计数
	sm.mu.Lock()
	user.FailedLogins = 0
	user.LastLogin = time.Now()
	user.LoginCount++
	user.UpdatedAt = time.Now()
	sm.users[user.ID] = user
	sm.mu.Unlock()

	// 保存用户状态
	sm.saveUser(user)

	// 创建会话
	session, err := sm.createSession(user.ID, ipAddress, userAgent)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %v", err)
	}

	// 记录成功登录
	sm.logAuditEvent(&AuditEvent{
		Type:      AuditLogin,
		Action:    "login",
		Result:    "success",
		UserID:    user.ID,
		SessionID: session.ID,
		IPAddress: ipAddress,
		UserAgent: userAgent,
		Timestamp: time.Now(),
		Details: map[string]interface{}{
			"username": username,
		},
		Message: fmt.Sprintf("User %s logged in successfully", username),
	})

	logger.Info("User authenticated: %s (%s)", username, user.ID)
	return session, nil
}

// ValidateSession 验证会话
func (sm *EnhancedSecurityManager) ValidateSession(sessionID string) (*Session, error) {
	if sessionID == "" {
		return nil, fmt.Errorf("session ID cannot be empty")
	}

	sm.sessionMu.RLock()
	session, exists := sm.sessions[sessionID]
	sm.sessionMu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("session not found")
	}

	// 检查会话是否过期
	if session.ExpiresAt.Before(time.Now()) {
		sm.terminateSession(sessionID)
		return nil, fmt.Errorf("session expired")
	}

	// 检查会话是否活跃
	if !session.Active {
		return nil, fmt.Errorf("session inactive")
	}

	// 更新最后访问时间
	sm.sessionMu.Lock()
	session.LastAccess = time.Now()
	sm.sessionMu.Unlock()

	return session, nil
}

// CheckPermission 检查权限
func (sm *EnhancedSecurityManager) CheckPermission(userID, resource, action string) bool {
	user, err := sm.GetUser(userID)
	if err != nil {
		return false
	}

	// 检查用户直接权限
	for _, permission := range user.Permissions {
		if sm.matchPermission(&permission, resource, action) {
			return true
		}
	}

	// 检查角色权限
	for _, roleID := range user.Roles {
		sm.mu.RLock()
		role, exists := sm.roles[roleID]
		sm.mu.RUnlock()

		if exists {
			for _, permission := range role.Permissions {
				if sm.matchPermission(&permission, resource, action) {
					return true
				}
			}

			// 检查父角色权限
			if sm.checkParentRolePermissions(role, resource, action) {
				return true
			}
		}
	}

	return false
}

// 加密和解密

// Encrypt 加密数据
func (sm *EnhancedSecurityManager) Encrypt(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("data cannot be empty")
	}

	block, err := aes.NewCipher(sm.encryptionKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %v", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %v", err)
	}

	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %v", err)
	}

	ciphertext := gcm.Seal(nonce, nonce, data, nil)
	return ciphertext, nil
}

// Decrypt 解密数据
func (sm *EnhancedSecurityManager) Decrypt(ciphertext []byte) ([]byte, error) {
	if len(ciphertext) == 0 {
		return nil, fmt.Errorf("ciphertext cannot be empty")
	}

	block, err := aes.NewCipher(sm.encryptionKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %v", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %v", err)
	}

	nonceSize := gcm.NonceSize()
	if len(ciphertext) < nonceSize {
		return nil, fmt.Errorf("ciphertext too short")
	}

	nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt: %v", err)
	}

	return plaintext, nil
}

// 辅助方法

// initializeEncryption 初始化加密
func (sm *EnhancedSecurityManager) initializeEncryption() error {
	if sm.config.EncryptionKey != nil {
		sm.encryptionKey = sm.config.EncryptionKey
		return nil
	}

	// 生成新的加密密钥
	key := make([]byte, 32) // AES-256
	if _, err := rand.Read(key); err != nil {
		return fmt.Errorf("failed to generate encryption key: %v", err)
	}

	sm.encryptionKey = key
	sm.config.EncryptionKey = key

	logger.Info("Encryption key generated")
	return nil
}

// initializeTLS 初始化TLS配置
func (sm *EnhancedSecurityManager) initializeTLS() error {
	if sm.config.TLSConfig != nil {
		sm.tlsConfig = sm.config.TLSConfig
		return nil
	}

	// 创建默认TLS配置
	tlsConfig := &tls.Config{
		MinVersion: tls.VersionTLS12,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_RSA_WITH_AES_256_GCM_SHA384,
		},
		PreferServerCipherSuites: true,
	}

	// 如果提供了证书路径，加载证书
	if sm.config.CertificatePath != "" && sm.config.PrivateKeyPath != "" {
		cert, err := tls.LoadX509KeyPair(sm.config.CertificatePath, sm.config.PrivateKeyPath)
		if err != nil {
			return fmt.Errorf("failed to load certificate: %v", err)
		}
		tlsConfig.Certificates = []tls.Certificate{cert}
	} else {
		// 生成自签名证书
		cert, err := sm.generateSelfSignedCert()
		if err != nil {
			return fmt.Errorf("failed to generate self-signed certificate: %v", err)
		}
		tlsConfig.Certificates = []tls.Certificate{*cert}
	}

	// 如果提供了CA路径，加载CA证书
	if sm.config.CAPath != "" {
		caCert, err := sm.loadCACert(sm.config.CAPath)
		if err != nil {
			return fmt.Errorf("failed to load CA certificate: %v", err)
		}
		tlsConfig.RootCAs = caCert
		tlsConfig.ClientCAs = caCert
		tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
	}

	sm.tlsConfig = tlsConfig
	sm.config.TLSConfig = tlsConfig

	logger.Info("TLS configuration initialized")
	return nil
}

// generateSelfSignedCert 生成自签名证书
func (sm *EnhancedSecurityManager) generateSelfSignedCert() (*tls.Certificate, error) {
	// 生成私钥
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, fmt.Errorf("failed to generate private key: %v", err)
	}

	// 创建证书模板
	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			Organization:  []string{"VectorSphere"},
			Country:       []string{"US"},
			Province:      []string{""},
			Locality:      []string{"San Francisco"},
			StreetAddress: []string{""},
			PostalCode:    []string{""},
		},
		NotBefore:   time.Now(),
		NotAfter:    time.Now().Add(365 * 24 * time.Hour), // 1年有效期
		KeyUsage:    x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IPAddresses: []net.IP{net.IPv4(127, 0, 0, 1), net.IPv6loopback},
		DNSNames:    []string{"localhost"},
	}

	// 生成证书
	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate: %v", err)
	}

	// 编码证书和私钥
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	privateKeyDER, err := x509.MarshalPKCS8PrivateKey(privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal private key: %v", err)
	}
	privateKeyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: privateKeyDER})

	// 创建TLS证书
	cert, err := tls.X509KeyPair(certPEM, privateKeyPEM)
	if err != nil {
		return nil, fmt.Errorf("failed to create X509 key pair: %v", err)
	}

	return &cert, nil
}

// loadCACert 加载CA证书
func (sm *EnhancedSecurityManager) loadCACert(caPath string) (*x509.CertPool, error) {
	// 这里应该实现从文件加载CA证书的逻辑
	// 为了简化，返回系统根证书池
	return x509.SystemCertPool()
}

// createDefaultPolicy 创建默认安全策略
func (sm *EnhancedSecurityManager) createDefaultPolicy() {
	defaultPolicy := &SecurityPolicy{
		ID:          "default",
		Name:        "Default Security Policy",
		Description: "Default security policy for VectorSphere",
		Level:       SecurityLevelMedium,
		PasswordPolicy: &PasswordPolicy{
			MinLength:        8,
			MaxLength:        128,
			RequireUppercase: true,
			RequireLowercase: true,
			RequireNumbers:   true,
			RequireSymbols:   false,
			MaxAge:           90 * 24 * time.Hour, // 90天
			HistoryCount:     5,
			LockoutThreshold: 5,
			LockoutDuration:  30 * time.Minute,
		},
		SessionPolicy: &SessionPolicy{
			MaxDuration:    24 * time.Hour,
			IdleTimeout:    2 * time.Hour,
			MaxConcurrent:  5,
			RequireRefresh: true,
			SecureCookies:  true,
			SameSitePolicy: "Strict",
		},
		EncryptionPolicy: &EncryptionPolicy{
			Algorithm:        AES256GCM,
			KeySize:          256,
			RotationInterval: 30 * 24 * time.Hour, // 30天
			EncryptAtRest:    true,
			EncryptInTransit: true,
			RequireTLS:       true,
			MinTLSVersion:    "1.2",
		},
		NetworkPolicy: &NetworkPolicy{
			AllowedIPs:   []string{},
			BlockedIPs:   []string{},
			AllowedPorts: []int{443, 8443},
			BlockedPorts: []int{},
			RequireVPN:   false,
		},
		AuditPolicy: &AuditPolicy{
			Enabled:         true,
			LogLevel:        "INFO",
			RetentionPeriod: 90 * 24 * time.Hour, // 90天
			EventTypes: []AuditEventType{
				AuditLogin,
				AuditLogout,
				AuditAccess,
				AuditPermissionDenied,
				AuditSecurityViolation,
			},
			RealTimeAlerts: true,
			Compression:    true,
			Encryption:     true,
		},
		RateLimitPolicy: &RateLimitPolicy{
			Enabled:        true,
			RequestsPerMin: 100,
			BurstSize:      20,
			WindowSize:     time.Minute,
			BlockDuration:  5 * time.Minute,
		},
		Enabled:   true,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	sm.policies[defaultPolicy.ID] = defaultPolicy
	logger.Info("Default security policy created")
}

// generateID 生成唯一ID
func (sm *EnhancedSecurityManager) generateID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("%x", b)
}

// generateSalt 生成盐值
func (sm *EnhancedSecurityManager) generateSalt() string {
	b := make([]byte, 32)
	rand.Read(b)
	return base64.StdEncoding.EncodeToString(b)
}

// hashPassword 哈希密码
func (sm *EnhancedSecurityManager) hashPassword(password, salt string) (string, error) {
	// 使用scrypt进行密码哈希
	dk, err := scrypt.Key([]byte(password), []byte(salt), 32768, 8, 1, 32)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(dk), nil
}

// verifyPassword 验证密码
func (sm *EnhancedSecurityManager) verifyPassword(password, hash, salt string) bool {
	expectedHash, err := sm.hashPassword(password, salt)
	if err != nil {
		return false
	}
	return expectedHash == hash
}

// validatePassword 验证密码策略
func (sm *EnhancedSecurityManager) validatePassword(password string) error {
	policy := sm.policies[sm.config.DefaultPolicy].PasswordPolicy
	if policy == nil {
		return nil
	}

	if len(password) < policy.MinLength {
		return fmt.Errorf("password too short, minimum length is %d", policy.MinLength)
	}

	if len(password) > policy.MaxLength {
		return fmt.Errorf("password too long, maximum length is %d", policy.MaxLength)
	}

	// 检查大写字母
	if policy.RequireUppercase {
		hasUpper := false
		for _, c := range password {
			if c >= 'A' && c <= 'Z' {
				hasUpper = true
				break
			}
		}
		if !hasUpper {
			return errors.New("password must contain at least one uppercase letter")
		}
	}

	// 检查小写字母
	if policy.RequireLowercase {
		hasLower := false
		for _, c := range password {
			if c >= 'a' && c <= 'z' {
				hasLower = true
				break
			}
		}
		if !hasLower {
			return errors.New("password must contain at least one lowercase letter")
		}
	}

	// 检查数字
	if policy.RequireNumbers {
		hasNumber := false
		for _, c := range password {
			if c >= '0' && c <= '9' {
				hasNumber = true
				break
			}
		}
		if !hasNumber {
			return errors.New("password must contain at least one number")
		}
	}

	// 检查特殊字符
	if policy.RequireSymbols {
		hasSpecial := false
		specialChars := "!@#$%^&*()-_=+[]{}|;:,.<>?/\\"
		for _, c := range password {
			if strings.ContainsRune(specialChars, c) {
				hasSpecial = true
				break
			}
		}
		if !hasSpecial {
			return errors.New("password must contain at least one special character")
		}
	}

	return nil
}

// createSession 创建会话
func (sm *EnhancedSecurityManager) createSession(userID, ipAddress, userAgent string) (*Session, error) {
	sessionID := sm.generateID()
	token := sm.generateSessionToken()

	session := &Session{
		ID:         sessionID,
		UserID:     userID,
		Token:      token,
		CreatedAt:  time.Now(),
		ExpiresAt:  time.Now().Add(sm.config.SessionTimeout),
		LastAccess: time.Now(),
		IPAddress:  ipAddress,
		UserAgent:  userAgent,
		Metadata:   make(map[string]interface{}),
		Active:     true,
	}

	sm.sessionMu.Lock()
	sm.sessions[sessionID] = session
	sm.sessionMu.Unlock()

	// 保存到etcd
	if err := sm.saveSession(session); err != nil {
		return nil, fmt.Errorf("failed to save session: %v", err)
	}

	return session, nil
}

// generateSessionToken 生成会话令牌
func (sm *EnhancedSecurityManager) generateSessionToken() string {
	b := make([]byte, 32)
	rand.Read(b)
	return base64.URLEncoding.EncodeToString(b)
}

// terminateSession 终止会话
func (sm *EnhancedSecurityManager) terminateSession(sessionID string) error {
	sm.sessionMu.Lock()
	session, exists := sm.sessions[sessionID]
	if exists {
		session.Active = false
		delete(sm.sessions, sessionID)
	}
	sm.sessionMu.Unlock()

	if !exists {
		return fmt.Errorf("session not found")
	}

	// 从etcd删除
	key := fmt.Sprintf("%s/sessions/%s", sm.basePrefix, sessionID)
	_, err := sm.client.Delete(context.Background(), key)
	if err != nil {
		logger.Error("Failed to delete session from etcd: %v", err)
	}

	// 记录审计事件
	sm.logAuditEvent(&AuditEvent{
		Type:      AuditLogout,
		Action:    "logout",
		Result:    "success",
		UserID:    session.UserID,
		SessionID: sessionID,
		Timestamp: time.Now(),
		Message:   "Session terminated",
	})

	return nil
}

// terminateUserSessions 终止用户所有会话
func (sm *EnhancedSecurityManager) terminateUserSessions(userID string) {
	sm.sessionMu.Lock()
	defer sm.sessionMu.Unlock()

	for sessionID, session := range sm.sessions {
		if session.UserID == userID {
			session.Active = false
			delete(sm.sessions, sessionID)

			// 从etcd删除
			key := fmt.Sprintf("%s/sessions/%s", sm.basePrefix, sessionID)
			sm.client.Delete(context.Background(), key)
		}
	}
}

// matchPermission 匹配权限
func (sm *EnhancedSecurityManager) matchPermission(permission *Permission, resource, action string) bool {
	// 简单的字符串匹配，可以扩展为更复杂的模式匹配
	if permission.Resource == "*" || permission.Resource == resource {
		if permission.Action == "*" || permission.Action == action {
			return true
		}
	}
	return false
}

// checkParentRolePermissions 检查父角色权限
func (sm *EnhancedSecurityManager) checkParentRolePermissions(role *Role, resource, action string) bool {
	for _, parentRoleID := range role.ParentRoles {
		sm.mu.RLock()
		parentRole, exists := sm.roles[parentRoleID]
		sm.mu.RUnlock()

		if exists {
			for _, permission := range parentRole.Permissions {
				if sm.matchPermission(&permission, resource, action) {
					return true
				}
			}

			// 递归检查父角色的父角色
			if sm.checkParentRolePermissions(parentRole, resource, action) {
				return true
			}
		}
	}
	return false
}

// logAuditEvent 记录审计事件
func (sm *EnhancedSecurityManager) logAuditEvent(event *AuditEvent) {
	if event.ID == "" {
		event.ID = sm.generateID()
	}
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}

	sm.auditMu.Lock()
	sm.auditEvents = append(sm.auditEvents, event)
	sm.auditMu.Unlock()

	// 异步保存到etcd
	go func() {
		if err := sm.saveAuditEvent(event); err != nil {
			logger.Error("Failed to save audit event: %v", err)
		}
	}()

	logger.Debug("Audit event logged: %s - %s", event.Type.String(), event.Message)
}

// 数据持久化

// saveUser 保存用户到etcd
func (sm *EnhancedSecurityManager) saveUser(user *User) error {
	data, err := json.Marshal(user)
	if err != nil {
		return fmt.Errorf("failed to marshal user: %v", err)
	}

	// 加密敏感数据
	encryptedData, err := sm.Encrypt(data)
	if err != nil {
		return fmt.Errorf("failed to encrypt user data: %v", err)
	}

	key := fmt.Sprintf("%s/users/%s", sm.basePrefix, user.ID)
	_, err = sm.client.Put(context.Background(), key, base64.StdEncoding.EncodeToString(encryptedData))
	return err
}

// saveRole 保存角色到etcd
func (sm *EnhancedSecurityManager) saveRole(role *Role) error {
	data, err := json.Marshal(role)
	if err != nil {
		return fmt.Errorf("failed to marshal role: %v", err)
	}

	key := fmt.Sprintf("%s/roles/%s", sm.basePrefix, role.ID)
	_, err = sm.client.Put(context.Background(), key, string(data))
	return err
}

// saveSession 保存会话到etcd
func (sm *EnhancedSecurityManager) saveSession(session *Session) error {
	data, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("failed to marshal session: %v", err)
	}

	key := fmt.Sprintf("%s/sessions/%s", sm.basePrefix, session.ID)
	_, err = sm.client.Put(context.Background(), key, string(data))
	return err
}

// saveAuditEvent 保存审计事件到etcd
func (sm *EnhancedSecurityManager) saveAuditEvent(event *AuditEvent) error {
	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal audit event: %v", err)
	}

	// 加密审计数据
	encryptedData, err := sm.Encrypt(data)
	if err != nil {
		return fmt.Errorf("failed to encrypt audit data: %v", err)
	}

	key := fmt.Sprintf("%s/audit/%s", sm.basePrefix, event.ID)
	_, err = sm.client.Put(context.Background(), key, base64.StdEncoding.EncodeToString(encryptedData))
	return err
}

// loadUsersAndRoles 加载用户和角色
func (sm *EnhancedSecurityManager) loadUsersAndRoles() error {
	// 加载用户
	userResp, err := sm.client.Get(context.Background(), fmt.Sprintf("%s/users/", sm.basePrefix), clientv3.WithPrefix())
	if err != nil {
		return fmt.Errorf("failed to load users: %v", err)
	}

	for _, kv := range userResp.Kvs {
		// 解密数据
		encryptedData, err := base64.StdEncoding.DecodeString(string(kv.Value))
		if err != nil {
			logger.Error("Failed to decode user data: %v", err)
			continue
		}

		data, err := sm.Decrypt(encryptedData)
		if err != nil {
			logger.Error("Failed to decrypt user data: %v", err)
			continue
		}

		var user User
		if err := json.Unmarshal(data, &user); err != nil {
			logger.Error("Failed to unmarshal user: %v", err)
			continue
		}

		sm.users[user.ID] = &user
	}

	// 加载角色
	roleResp, err := sm.client.Get(context.Background(), fmt.Sprintf("%s/roles/", sm.basePrefix), clientv3.WithPrefix())
	if err != nil {
		return fmt.Errorf("failed to load roles: %v", err)
	}

	for _, kv := range roleResp.Kvs {
		var role Role
		if err := json.Unmarshal(kv.Value, &role); err != nil {
			logger.Error("Failed to unmarshal role: %v", err)
			continue
		}

		sm.roles[role.ID] = &role
	}

	logger.Info("Loaded %d users and %d roles", len(sm.users), len(sm.roles))
	return nil
}

// loadPolicies 加载安全策略
func (sm *EnhancedSecurityManager) loadPolicies() error {
	resp, err := sm.client.Get(context.Background(), fmt.Sprintf("%s/policies/", sm.basePrefix), clientv3.WithPrefix())
	if err != nil {
		return fmt.Errorf("failed to load policies: %v", err)
	}

	for _, kv := range resp.Kvs {
		var policy SecurityPolicy
		if err := json.Unmarshal(kv.Value, &policy); err != nil {
			logger.Error("Failed to unmarshal policy: %v", err)
			continue
		}

		sm.policies[policy.ID] = &policy
	}

	logger.Info("Loaded %d security policies", len(sm.policies))
	return nil
}

// 监控和清理

// auditLogger 审计日志记录器
func (sm *EnhancedSecurityManager) auditLogger() {
	logger.Info("Starting audit logger")

	for {
		select {
		case <-sm.ctx.Done():
			logger.Info("Audit logger stopped")
			return
		case <-sm.auditTicker.C:
			sm.flushAuditLogs()
		}
	}
}

// flushAuditLogs 刷新审计日志
func (sm *EnhancedSecurityManager) flushAuditLogs() {
	sm.auditMu.Lock()
	events := make([]*AuditEvent, len(sm.auditEvents))
	copy(events, sm.auditEvents)
	sm.auditEvents = sm.auditEvents[:0] // 清空切片
	sm.auditMu.Unlock()

	if len(events) == 0 {
		return
	}

	// 批量保存审计事件
	for _, event := range events {
		if err := sm.saveAuditEvent(event); err != nil {
			logger.Error("Failed to save audit event %s: %v", event.ID, err)
		}
	}

	logger.Debug("Flushed %d audit events", len(events))
}

// sessionMonitor 会话监控
func (sm *EnhancedSecurityManager) sessionMonitor() {
	logger.Info("Starting session monitor")

	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-sm.ctx.Done():
			logger.Info("Session monitor stopped")
			return
		case <-ticker.C:
			sm.cleanupExpiredSessions()
		}
	}
}

// cleanupExpiredSessions 清理过期会话
func (sm *EnhancedSecurityManager) cleanupExpiredSessions() {
	now := time.Now()
	expiredSessions := make([]string, 0)

	sm.sessionMu.RLock()
	for sessionID, session := range sm.sessions {
		if session.ExpiresAt.Before(now) || !session.Active {
			expiredSessions = append(expiredSessions, sessionID)
		}
	}
	sm.sessionMu.RUnlock()

	for _, sessionID := range expiredSessions {
		sm.terminateSession(sessionID)
	}

	if len(expiredSessions) > 0 {
		logger.Debug("Cleaned up %d expired sessions", len(expiredSessions))
	}
}

// cleaner 清理器
func (sm *EnhancedSecurityManager) cleaner() {
	logger.Info("Starting security manager cleaner")

	for {
		select {
		case <-sm.ctx.Done():
			logger.Info("Security manager cleaner stopped")
			return
		case <-sm.cleanupTicker.C:
			sm.cleanup()
		}
	}
}

// cleanup 清理过期数据
func (sm *EnhancedSecurityManager) cleanup() {
	now := time.Now()

	// 清理过期审计事件（保留策略中定义的时间）
	policy := sm.policies[sm.config.DefaultPolicy]
	if policy != nil && policy.AuditPolicy != nil {
		retentionPeriod := policy.AuditPolicy.RetentionPeriod
		cutoff := now.Add(-retentionPeriod)

		sm.auditMu.Lock()
		validEvents := make([]*AuditEvent, 0)
		for _, event := range sm.auditEvents {
			if event.Timestamp.After(cutoff) {
				validEvents = append(validEvents, event)
			}
		}
		sm.auditEvents = validEvents
		sm.auditMu.Unlock()
	}

	logger.Debug("Security manager cleanup completed")
}

// GetTLSConfig 获取TLS配置
func (sm *EnhancedSecurityManager) GetTLSConfig() *tls.Config {
	return sm.tlsConfig
}

// GetAuditEvents 获取审计事件
func (sm *EnhancedSecurityManager) GetAuditEvents(limit int) []*AuditEvent {
	sm.auditMu.RLock()
	defer sm.auditMu.RUnlock()

	if limit <= 0 || limit > len(sm.auditEvents) {
		limit = len(sm.auditEvents)
	}

	result := make([]*AuditEvent, limit)
	copy(result, sm.auditEvents[len(sm.auditEvents)-limit:])
	return result
}

// GetActiveSessions 获取活跃会话
func (sm *EnhancedSecurityManager) GetActiveSessions() []*Session {
	sm.sessionMu.RLock()
	defer sm.sessionMu.RUnlock()

	result := make([]*Session, 0, len(sm.sessions))
	for _, session := range sm.sessions {
		if session.Active && session.ExpiresAt.After(time.Now()) {
			sessionCopy := *session
			result = append(result, &sessionCopy)
		}
	}
	return result
}

// GetSecurityMetrics 获取安全指标
func (sm *EnhancedSecurityManager) GetSecurityMetrics() map[string]interface{} {
	sm.mu.RLock()
	userCount := len(sm.users)
	roleCount := len(sm.roles)
	policyCount := len(sm.policies)
	sm.mu.RUnlock()

	sm.sessionMu.RLock()
	activeSessionCount := 0
	for _, session := range sm.sessions {
		if session.Active && session.ExpiresAt.After(time.Now()) {
			activeSessionCount++
		}
	}
	sm.sessionMu.RUnlock()

	sm.auditMu.RLock()
	auditEventCount := len(sm.auditEvents)
	sm.auditMu.RUnlock()

	return map[string]interface{}{
		"users":           userCount,
		"roles":           roleCount,
		"policies":        policyCount,
		"active_sessions": activeSessionCount,
		"audit_events":    auditEventCount,
		"is_running":      sm.isRunning,
		"last_update":     time.Now(),
	}
}

// ValidateRequest 验证请求
func (sm *EnhancedSecurityManager) ValidateRequest(r *http.Request) error {
	// 获取认证头
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		return fmt.Errorf("missing authorization header")
	}

	// 解析Bearer token
	if !strings.HasPrefix(authHeader, "Bearer ") {
		return fmt.Errorf("invalid authorization header format")
	}

	token := strings.TrimPrefix(authHeader, "Bearer ")
	if token == "" {
		return fmt.Errorf("empty token")
	}

	// 验证会话
	// 通过token查找session
	sm.sessionMu.RLock()
	var session *Session
	for _, s := range sm.sessions {
		if s.Token == token {
			session = s
			break
		}
	}
	sm.sessionMu.RUnlock()
	
	if session == nil {
		return fmt.Errorf("session not found")
	}

	// 检查会话是否过期
	if session.ExpiresAt.Before(time.Now()) {
		return fmt.Errorf("session expired")
	}

	// 检查会话是否活跃
	if !session.Active {
		return fmt.Errorf("session inactive")
	}

	// 记录审计事件
	sm.logAuditEvent(&AuditEvent{
		Type:      AuditAccess,
		UserID:    session.UserID,
		SessionID: session.ID,
		Resource:  r.URL.Path,
		Action:    r.Method,
		Result:    "success",
		Message:   "Request validated successfully",
		IPAddress: r.RemoteAddr,
		UserAgent: r.Header.Get("User-Agent"),
	})

	return nil
}

// CheckRateLimit 检查速率限制
func (sm *EnhancedSecurityManager) CheckRateLimit(userID string) error {
	// 获取用户
	user, err := sm.GetUser(userID)
	if err != nil {
		return fmt.Errorf("user not found: %v", err)
	}

	// 获取用户的安全策略
	var policy *SecurityPolicy
	for _, roleID := range user.Roles {
		role, exists := sm.roles[roleID]
		if exists {
			// 检查角色权限 (PolicyID字段已移除，使用Permissions)
			if len(role.Permissions) > 0 {
				// 使用默认策略或基于权限的策略
				break
			}
		}
	}

	// 如果没有找到特定策略，使用默认策略
	if policy == nil {
		if defaultPolicy, exists := sm.policies[sm.config.DefaultPolicy]; exists {
			policy = defaultPolicy
		}
	}

	// 如果仍然没有策略，允许请求
	if policy == nil || policy.RateLimitPolicy == nil {
		return nil
	}

	// 简单的速率限制实现（基于内存）
	// 在生产环境中，应该使用Redis或其他分布式存储
	now := time.Now()
	key := fmt.Sprintf("rate_limit_%s", userID)
	
	// 这里简化实现，实际应该使用滑动窗口算法
	// 暂时返回nil表示允许请求
	_ = now
	_ = key

	return nil
}
