package test

import (
	"VectorSphere/src/security"
	"context"
	"fmt"
	"os"
	"testing"
	"time"
)

// TestNewSecurityManager 测试基础安全管理器创建
func TestNewSecurityManager(t *testing.T) {
	// 测试不启用RBAC
	sm := security.NewSecurityManager(nil, false)
	if sm == nil {
		t.Fatal("安全管理器创建失败")
	}

	if sm.GetTLSConfig() != nil {
		t.Error("期望TLS配置为nil")
	}

	// 测试启用RBAC
	smWithRBAC := security.NewSecurityManager(nil, true)
	if smWithRBAC == nil {
		t.Fatal("启用RBAC的安全管理器创建失败")
	}
}

// TestNewEnhancedSecurityManager 测试增强安全管理器创建
func TestNewEnhancedSecurityManager(t *testing.T) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{
			Enabled:     true,
			DefaultRole: "readonly",
			Roles: map[string][]string{
				"admin": {"service:*", "config:*"},
				"user":  {"service:read", "config:read"},
			},
		},
		Audit: &security.AuditConfig{
			Enabled:   true,
			LogFile:   "test_audit.log",
			Retention: 30 * 24 * time.Hour,
			LogLevel:  "INFO",
			Encrypted: true,
		},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{"192.168.1.0/24", "10.0.0.0/8"},
			BlockedIPs: []string{"192.168.1.100/32"},
			RateLimit: &security.RateLimitConfig{
				Enabled:   true,
				Requests:  100,
				Window:    time.Minute,
				BurstSize: 10,
			},
		},
		JWT: &security.JWTConfig{
			Secret:     "test-secret-key",
			Expiration: 24 * time.Hour,
			Issuer:     "VectorSphere",
			Algorithm:  "HS256",
		},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		t.Fatalf("增强安全管理器创建失败: %v", err)
	}

	if esm == nil {
		t.Fatal("增强安全管理器不应为nil")
	}
}

// TestAddUserAndRole 测试用户和角色管理
func TestAddUserAndRole(t *testing.T) {
	sm := security.NewSecurityManager(nil, true)

	// 添加用户
	sm.AddUser("user1", []string{"admin", "service"})
	sm.AddUser("user2", []string{"readonly"})

	// 添加自定义角色
	sm.AddRole("custom", []string{"custom:read", "custom:write"})

	// 测试认证和授权
	err := sm.AuthenticateAndAuthorize(context.Background(), "user1", "service:register")
	if err != nil {
		t.Errorf("用户user1应该有service:register权限: %v", err)
	}

	err = sm.AuthenticateAndAuthorize(context.Background(), "user2", "service:register")
	if err == nil {
		t.Error("用户user2不应该有service:register权限")
	}
}

// TestAuthenticateAndAuthorizeWithoutRBAC 测试未启用RBAC时的认证
func TestAuthenticateAndAuthorizeWithoutRBAC(t *testing.T) {
	sm := security.NewSecurityManager(nil, false)

	// 未启用RBAC时应该跳过检查
	err := sm.AuthenticateAndAuthorize(context.Background(), "any-token", "any-permission")
	if err != nil {
		t.Errorf("未启用RBAC时不应该有权限检查错误: %v", err)
	}
}

// TestEncryptionAndDecryption 测试数据加密和解密
func TestEncryptionAndDecryption(t *testing.T) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{Enabled: false},
		Audit: &security.AuditConfig{
			Enabled: false,
			LogFile: "test.log",
		},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{},
			BlockedIPs: []string{},
		},
		JWT: &security.JWTConfig{
			Secret: "test-secret",
		},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		t.Fatalf("创建增强安全管理器失败: %v", err)
	}

	// 测试数据
	originalData := []byte("这是需要加密的敏感数据")

	// 加密
	encryptedData, err := esm.EncryptData(originalData)
	if err != nil {
		t.Fatalf("数据加密失败: %v", err)
	}

	if len(encryptedData) == 0 {
		t.Error("加密数据不应为空")
	}

	// 解密
	decryptedData, err := esm.DecryptData(encryptedData)
	if err != nil {
		t.Fatalf("数据解密失败: %v", err)
	}

	if string(decryptedData) != string(originalData) {
		t.Errorf("解密后的数据不匹配，期望: %s，实际: %s", string(originalData), string(decryptedData))
	}
}

// TestJWTGeneration 测试JWT令牌生成和验证
func TestJWTGeneration(t *testing.T) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{Enabled: false},
		Audit: &security.AuditConfig{
			Enabled: false,
			LogFile: "test.log",
		},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{},
			BlockedIPs: []string{},
		},
		JWT: &security.JWTConfig{
			Secret:     "test-secret-key-for-jwt",
			Expiration: 24 * time.Hour,
			Issuer:     "VectorSphere",
			Algorithm:  "HS256",
		},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		t.Fatalf("创建增强安全管理器失败: %v", err)
	}

	// 生成JWT令牌
	userID := "test-user"
	roles := []string{"admin", "service"}

	token, err := esm.GenerateJWT(userID, roles)
	if err != nil {
		t.Fatalf("JWT令牌生成失败: %v", err)
	}

	if token == "" {
		t.Error("生成的JWT令牌不应为空")
	}

	// 验证JWT令牌
	claims, err := esm.ValidateJWT(token)
	if err != nil {
		t.Fatalf("JWT令牌验证失败: %v", err)
	}

	if (*claims)["user_id"] != userID {
		t.Errorf("用户ID不匹配，期望: %s，实际: %s", userID, (*claims)["user_id"])
	}

	if (*claims)["iss"] != "VectorSphere" {
		t.Errorf("发行者不匹配，期望: VectorSphere，实际: %s", (*claims)["iss"])
	}
}

// TestInvalidJWTValidation 测试无效JWT令牌验证
func TestInvalidJWTValidation(t *testing.T) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{Enabled: false},
		Audit: &security.AuditConfig{
			Enabled: false,
			LogFile: "test.log",
		},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{},
			BlockedIPs: []string{},
		},
		JWT: &security.JWTConfig{
			Secret: "test-secret-key",
		},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		t.Fatalf("创建增强安全管理器失败: %v", err)
	}

	// 测试无效令牌
	invalidTokens := []string{
		"",
		"invalid.token.format",
		"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature",
	}

	for _, token := range invalidTokens {
		_, err := esm.ValidateJWT(token)
		if err == nil {
			t.Errorf("无效令牌应该验证失败: %s", token)
		}
	}
}

// TestNetworkAccessControl 测试网络访问控制
func TestNetworkAccessControl(t *testing.T) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{Enabled: false},
		Audit: &security.AuditConfig{
			Enabled: false,
			LogFile: "test.log",
		},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{"192.168.1.0/24", "10.0.0.0/8"},
			BlockedIPs: []string{"192.168.1.100/32", "172.16.0.0/16"},
		},
		JWT: &security.JWTConfig{
			Secret: "test-secret",
		},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		t.Fatalf("创建增强安全管理器失败: %v", err)
	}

	// 测试允许的IP
	allowedIPs := []string{"192.168.1.50", "10.0.0.1"}
	for _, ip := range allowedIPs {
		err := esm.CheckNetworkAccess(ip)
		if err != nil {
			t.Errorf("IP %s 应该被允许访问: %v", ip, err)
		}
	}

	// 测试被阻止的IP
	blockedIPs := []string{"192.168.1.100", "172.16.0.1"}
	for _, ip := range blockedIPs {
		err := esm.CheckNetworkAccess(ip)
		if err == nil {
			t.Errorf("IP %s 应该被阻止访问", ip)
		}
	}

	// 测试不在允许列表中的IP
	notAllowedIPs := []string{"8.8.8.8", "1.1.1.1"}
	for _, ip := range notAllowedIPs {
		err := esm.CheckNetworkAccess(ip)
		if err == nil {
			t.Errorf("IP %s 不在允许列表中，应该被拒绝", ip)
		}
	}

	// 测试无效IP
	err = esm.CheckNetworkAccess("invalid-ip")
	if err == nil {
		t.Error("无效IP应该被拒绝")
	}
}

// TestRateLimit 测试速率限制
func TestRateLimit(t *testing.T) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{Enabled: false},
		Audit: &security.AuditConfig{
			Enabled: false,
			LogFile: "test.log",
		},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{},
			BlockedIPs: []string{},
		},
		JWT: &security.JWTConfig{
			Secret: "test-secret",
		},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		t.Fatalf("创建增强安全管理器失败: %v", err)
	}

	clientIP := "192.168.1.1"
	maxRequests := 5
	window := time.Second

	// 在限制内的请求应该成功
	for i := 0; i < maxRequests; i++ {
		err := esm.CheckRateLimit(clientIP, maxRequests, window)
		if err != nil {
			t.Errorf("请求 %d 应该被允许: %v", i+1, err)
		}
	}

	// 超出限制的请求应该失败
	err = esm.CheckRateLimit(clientIP, maxRequests, window)
	if err == nil {
		t.Error("超出速率限制的请求应该被拒绝")
	}

	// 等待窗口重置
	time.Sleep(window + 100*time.Millisecond)

	// 重置后的请求应该成功
	err = esm.CheckRateLimit(clientIP, maxRequests, window)
	if err != nil {
		t.Errorf("窗口重置后的请求应该被允许: %v", err)
	}
}

// TestAuditLogging 测试审计日志记录
func TestAuditLogging(t *testing.T) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{Enabled: false},
		Audit: &security.AuditConfig{
			Enabled:   true,
			LogFile:   "test_audit.log",
			Retention: 24 * time.Hour,
			LogLevel:  "INFO",
			Encrypted: false,
		},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{},
			BlockedIPs: []string{},
		},
		JWT: &security.JWTConfig{
			Secret: "test-secret",
		},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		t.Fatalf("创建增强安全管理器失败: %v", err)
	}

	// 创建审计事件
	event := &security.AuditEvent{
		Timestamp: time.Now(),
		UserID:    "test-user",
		Action:    "login",
		Resource:  "system",
		Result:    "SUCCESS",
		IP:        "192.168.1.1",
		UserAgent: "test-agent",
		Details:   "用户登录成功",
		RiskLevel: "LOW",
	}

	// 记录审计事件
	err = esm.LogAuditEvent(event)
	if err != nil {
		t.Errorf("审计事件记录失败: %v", err)
	}
}

// TestEnhancedAuthenticateAndAuthorize 测试增强的认证和授权
func TestEnhancedAuthenticateAndAuthorize(t *testing.T) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{
			Enabled: true,
			Roles: map[string][]string{
				"admin": {"service:*", "config:*"},
			},
		},
		Audit: &security.AuditConfig{
			Enabled: true,
			LogFile: "test_audit.log",
		},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{"192.168.1.0/24"},
			BlockedIPs: []string{},
		},
		JWT: &security.JWTConfig{
			Secret: "test-secret-key",
		},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		t.Fatalf("创建增强安全管理器失败: %v", err)
	}

	// 生成有效的JWT令牌
	token, err := esm.GenerateJWT("test-user", []string{"admin"})
	if err != nil {
		t.Fatalf("JWT令牌生成失败: %v", err)
	}

	// 测试成功的认证和授权
	err = esm.EnhancedAuthenticateAndAuthorize(
		context.Background(),
		"Bearer "+token,
		"192.168.1.50",
		"test-agent",
		"read",
		"service",
	)
	if err != nil {
		t.Errorf("增强认证和授权应该成功: %v", err)
	}

	// 测试被阻止的IP
	err = esm.EnhancedAuthenticateAndAuthorize(
		context.Background(),
		"Bearer "+token,
		"8.8.8.8", // 不在允许列表中
		"test-agent",
		"read",
		"service",
	)
	if err == nil {
		t.Error("不在允许列表中的IP应该被拒绝")
	}
}

// TestPermissionMatching 测试权限匹配
func TestPermissionMatching(t *testing.T) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{
			Enabled: true,
			Roles: map[string][]string{
				"admin":    {"*:*"},
				"service":  {"service:*"},
				"readonly": {"*:read"},
				"specific": {"config:write", "service:register"},
			},
		},
		Audit: &security.AuditConfig{Enabled: false, LogFile: "test.log"},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{"0.0.0.0/0"}, // 允许所有IP
			BlockedIPs: []string{},
		},
		JWT: &security.JWTConfig{Secret: "test-secret"},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		t.Fatalf("创建增强安全管理器失败: %v", err)
	}

	testCases := []struct {
		roles    []string
		action   string
		resource string
		expected bool
	}{
		{[]string{"admin"}, "read", "service", true},     // *:* 匹配所有
		{[]string{"admin"}, "write", "config", true},    // *:* 匹配所有
		{[]string{"service"}, "read", "service", true},  // service:* 匹配
		{[]string{"service"}, "write", "service", true}, // service:* 匹配
		{[]string{"service"}, "read", "config", false},  // service:* 不匹配config
		{[]string{"readonly"}, "read", "service", true}, // *:read 匹配
		{[]string{"readonly"}, "read", "config", true},  // *:read 匹配
		{[]string{"readonly"}, "write", "config", false}, // *:read 不匹配write
		{[]string{"specific"}, "write", "config", true},   // 精确匹配
		{[]string{"specific"}, "register", "service", true}, // 精确匹配
		{[]string{"specific"}, "read", "config", false},    // 无匹配权限
	}

	for i, tc := range testCases {
		token, err := esm.GenerateJWT("test-user", tc.roles)
		if err != nil {
			t.Fatalf("测试用例 %d: JWT令牌生成失败: %v", i, err)
		}

		err = esm.EnhancedAuthenticateAndAuthorize(
			context.Background(),
			"Bearer "+token,
			"192.168.1.1",
			"test-agent",
			tc.action,
			tc.resource,
		)

		if tc.expected && err != nil {
			t.Errorf("测试用例 %d: 期望成功但失败了，角色: %v, 动作: %s, 资源: %s, 错误: %v",
				i, tc.roles, tc.action, tc.resource, err)
		} else if !tc.expected && err == nil {
			t.Errorf("测试用例 %d: 期望失败但成功了，角色: %v, 动作: %s, 资源: %s",
				i, tc.roles, tc.action, tc.resource)
		}
	}
}

// TestEncryptionKeyRotation 测试加密密钥轮换
func TestEncryptionKeyRotation(t *testing.T) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{Enabled: false},
		Audit: &security.AuditConfig{Enabled: false, LogFile: "test.log"},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{},
			BlockedIPs: []string{},
		},
		JWT: &security.JWTConfig{Secret: "test-secret"},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		t.Fatalf("创建增强安全管理器失败: %v", err)
	}

	// 加密一些数据
	originalData := []byte("测试数据")
	encryptedData1, err := esm.EncryptData(originalData)
	if err != nil {
		t.Fatalf("初始加密失败: %v", err)
	}

	// 轮换密钥
	err = esm.RotateEncryptionKey()
	if err != nil {
		t.Fatalf("密钥轮换失败: %v", err)
	}

	// 使用新密钥加密数据
	encryptedData2, err := esm.EncryptData(originalData)
	if err != nil {
		t.Fatalf("密钥轮换后加密失败: %v", err)
	}

	// 新加密的数据应该与旧的不同
	if string(encryptedData1) == string(encryptedData2) {
		t.Error("密钥轮换后加密结果应该不同")
	}

	// 新密钥应该能解密新数据
	decryptedData, err := esm.DecryptData(encryptedData2)
	if err != nil {
		t.Fatalf("新密钥解密失败: %v", err)
	}

	if string(decryptedData) != string(originalData) {
		t.Errorf("解密数据不匹配，期望: %s，实际: %s", string(originalData), string(decryptedData))
	}
}

// TestConcurrentAccess 测试并发访问安全性
func TestSecurityManagerConcurrentAccess(t *testing.T) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{
			Enabled: true,
			Roles: map[string][]string{
				"user": {"service:read"},
			},
		},
		Audit: &security.AuditConfig{Enabled: true, LogFile: "test_concurrent.log"},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{"192.168.1.0/24"},
			BlockedIPs: []string{},
		},
		JWT: &security.JWTConfig{Secret: "test-secret"},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		t.Fatalf("创建增强安全管理器失败: %v", err)
	}

	const numGoroutines = 10
	const operationsPerGoroutine = 50

	done := make(chan bool, numGoroutines*3)

	// 并发加密/解密
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			for j := 0; j < operationsPerGoroutine; j++ {
				data := []byte(fmt.Sprintf("test-data-%d-%d", id, j))
				encrypted, err := esm.EncryptData(data)
				if err != nil {
					t.Errorf("并发加密失败: %v", err)
					return
				}
				_, err = esm.DecryptData(encrypted)
				if err != nil {
					t.Errorf("并发解密失败: %v", err)
					return
				}
			}
			done <- true
		}(i)
	}

	// 并发JWT操作
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			for j := 0; j < operationsPerGoroutine; j++ {
				userID := fmt.Sprintf("user-%d-%d", id, j)
				token, err := esm.GenerateJWT(userID, []string{"user"})
				if err != nil {
					t.Errorf("并发JWT生成失败: %v", err)
					return
				}
				_, err = esm.ValidateJWT(token)
				if err != nil {
					t.Errorf("并发JWT验证失败: %v", err)
					return
				}
			}
			done <- true
		}(i)
	}

	// 并发网络访问检查
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			for j := 0; j < operationsPerGoroutine; j++ {
				ip := fmt.Sprintf("192.168.1.%d", (id*10+j)%254+1)
				err := esm.CheckNetworkAccess(ip)
				if err != nil {
					t.Errorf("并发网络访问检查失败: %v", err)
					return
				}
			}
			done <- true
		}(i)
	}

	// 等待所有goroutine完成
	for i := 0; i < numGoroutines*3; i++ {
		<-done
	}
}

// TestInvalidEncryptionData 测试无效加密数据处理
func TestInvalidEncryptionData(t *testing.T) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{Enabled: false},
		Audit: &security.AuditConfig{Enabled: false, LogFile: "test.log"},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{},
			BlockedIPs: []string{},
		},
		JWT: &security.JWTConfig{Secret: "test-secret"},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		t.Fatalf("创建增强安全管理器失败: %v", err)
	}

	// 测试解密无效数据
	invalidData := [][]byte{
		{}, // 空数据
		{1, 2, 3}, // 太短的数据
		[]byte("invalid-encrypted-data"), // 无效格式
	}

	for i, data := range invalidData {
		_, err := esm.DecryptData(data)
		if err == nil {
			t.Errorf("无效数据 %d 应该解密失败", i)
		}
	}
}

// BenchmarkSecurityOperations 安全操作基准测试
func BenchmarkSecurityOperations(b *testing.B) {
	config := &security.SecurityConfig{
		RBAC: &security.RBACConfig{Enabled: false},
		Audit: &security.AuditConfig{Enabled: false, LogFile: "bench.log"},
		Network: &security.NetworkConfig{
			AllowedIPs: []string{},
			BlockedIPs: []string{},
		},
		JWT: &security.JWTConfig{Secret: "bench-secret"},
	}

	esm, err := security.NewEnhancedSecurityManager(config)
	if err != nil {
		b.Fatalf("创建增强安全管理器失败: %v", err)
	}

	testData := []byte("benchmark test data for encryption")

	b.Run("Encryption", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := esm.EncryptData(testData)
			if err != nil {
				b.Fatalf("加密失败: %v", err)
			}
		}
	})

	encryptedData, _ := esm.EncryptData(testData)
	b.Run("Decryption", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := esm.DecryptData(encryptedData)
			if err != nil {
				b.Fatalf("解密失败: %v", err)
			}
		}
	})

	b.Run("JWTGeneration", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := esm.GenerateJWT("bench-user", []string{"user"})
			if err != nil {
				b.Fatalf("JWT生成失败: %v", err)
			}
		}
	})

	token, _ := esm.GenerateJWT("bench-user", []string{"user"})
	b.Run("JWTValidation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := esm.ValidateJWT(token)
			if err != nil {
				b.Fatalf("JWT验证失败: %v", err)
			}
		}
	})

	b.Run("NetworkAccessCheck", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = esm.CheckNetworkAccess("192.168.1.1")
		}
	})
}

// 清理测试文件
func TestMain(m *testing.M) {
	// 运行测试
	code := m.Run()

	// 清理测试文件
	testFiles := []string{
		"test_audit.log",
		"test.log",
		"test_concurrent.log",
		"bench.log",
	}

	for _, file := range testFiles {
		os.Remove(file)
	}

	os.Exit(code)
}