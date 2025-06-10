package server

import (
	"VectorSphere/src/security"
	"context"
	"fmt"
	"github.com/sony/gobreaker"
	"net/http"
	"strconv"
	"strings"
	"time"

	"VectorSphere/src/library/log"
)

// SecurityMiddleware 安全中间件
type SecurityMiddleware struct {
	securityManager *security.EnhancedSecurityManager
}

// NewSecurityMiddleware 创建安全中间件
func NewSecurityMiddleware(sm *security.EnhancedSecurityManager) *SecurityMiddleware {
	return &SecurityMiddleware{
		securityManager: sm,
	}
}

// EnhancedAuthMiddleware 增强的认证中间件
func (sm *SecurityMiddleware) EnhancedAuthMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 获取客户端信息
		clientIP := getClientIP(r)
		userAgent := r.UserAgent()
		token := r.Header.Get("Authorization")
		action := r.Method
		resource := r.URL.Path

		// 设置安全头
		setSecurityHeaders(w)

		// 执行增强的认证和授权
		if err := sm.securityManager.EnhancedAuthenticateAndAuthorize(
			r.Context(), token, clientIP, userAgent, action, resource,
		); err != nil {
			log.Warning("Security check failed for %s from %s: %v", resource, clientIP, err)
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		// 在请求上下文中添加安全信息
		ctx := context.WithValue(r.Context(), "client_ip", clientIP)
		ctx = context.WithValue(ctx, "user_agent", userAgent)

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// TLSMiddleware TLS中间件
func (sm *SecurityMiddleware) TLSMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 强制HTTPS
		if r.TLS == nil {
			https := "https://" + r.Host + r.RequestURI
			http.Redirect(w, r, https, http.StatusMovedPermanently)
			return
		}

		// 验证客户端证书（如果需要）
		if r.TLS.PeerCertificates != nil && len(r.TLS.PeerCertificates) > 0 {
			cert := r.TLS.PeerCertificates[0]
			if time.Now().After(cert.NotAfter) {
				http.Error(w, "Client certificate expired", http.StatusUnauthorized)
				return
			}
		}

		next.ServeHTTP(w, r)
	})
}

// getClientIP 获取客户端真实IP
func getClientIP(r *http.Request) string {
	// 检查X-Forwarded-For头
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		ips := strings.Split(xff, ",")
		if len(ips) > 0 {
			return strings.TrimSpace(ips[0])
		}
	}

	// 检查X-Real-IP头
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}

	// 使用RemoteAddr
	ip := r.RemoteAddr
	if idx := strings.LastIndex(ip, ":"); idx != -1 {
		ip = ip[:idx]
	}

	return ip
}

// setSecurityHeaders 设置安全头
func setSecurityHeaders(w http.ResponseWriter) {
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("X-XSS-Protection", "1; mode=block")
	w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
	w.Header().Set("Content-Security-Policy", "default-src 'self'")
	w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
}

// LoadBalancingMiddleware 负载均衡中间件
func LoadBalancingMiddleware(balancer Balancer) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// 获取客户端IP用于源IP哈希
			clientIP := getClientIP(r)

			// 如果是源IP哈希负载均衡器，传递客户端IP
			if sipBalancer, ok := balancer.(*SourceIPHashBalancer); ok {
				endpoint := sipBalancer.TakeWithContext(clientIP)
				// 将选中的端点信息添加到请求头
				r.Header.Set("X-Target-Endpoint", fmt.Sprintf("%s:%d", endpoint.Ip, endpoint.Port))
			} else {
				endpoint := balancer.Take()
				r.Header.Set("X-Target-Endpoint", fmt.Sprintf("%s:%d", endpoint.Ip, endpoint.Port))
			}

			// 记录请求开始时间
			startTime := time.Now()

			// 创建响应记录器来捕获响应信息
			rec := &responseRecorder{
				ResponseWriter: w,
				status:         200,
			}

			next.ServeHTTP(rec, r)

			// 记录响应时间
			responseTime := time.Since(startTime)

			// 如果是响应时间加权负载均衡器，记录响应时间
			if rtBalancer, ok := balancer.(*ResponseTimeWeightedBalancer); ok {
				targetEndpoint := r.Header.Get("X-Target-Endpoint")
				if targetEndpoint != "" {
					parts := strings.Split(targetEndpoint, ":")
					if len(parts) == 2 {
						port, _ := strconv.Atoi(parts[1])
						endpoint := EndPoint{Ip: parts[0], Port: port}
						rtBalancer.RecordResponseTime(endpoint, responseTime)
					}
				}
			}
		})
	}
}

// responseRecorder 用于记录响应信息
type responseRecorder struct {
	http.ResponseWriter
	status int
}

func (r *responseRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

// CircuitBreakerMiddleware 熔断器中间件
func CircuitBreakerMiddleware(cb *gobreaker.CircuitBreaker) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			result, err := cb.Execute(func() (interface{}, error) {
				rec := &responseRecorder{
					ResponseWriter: w,
					status:         200,
				}
				next.ServeHTTP(rec, r)

				// 如果响应状态码表示错误，返回错误
				if rec.status >= 500 {
					return nil, fmt.Errorf("server error: %d", rec.status)
				}
				return nil, nil
			})

			if err != nil {
				http.Error(w, "Service temporarily unavailable", http.StatusServiceUnavailable)
				return
			}

			_ = result // 忽略结果
		})
	}
}
