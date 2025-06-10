package bootstrap

import (
	"VectorSphere/src/server"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// LoadBalancerMetrics 负载均衡指标
type LoadBalancerMetrics struct {
	TotalRequests   int64                      `json:"total_requests"`
	SuccessRequests int64                      `json:"success_requests"`
	FailedRequests  int64                      `json:"failed_requests"`
	AverageLatency  float64                    `json:"average_latency_ms"`
	EndpointMetrics map[string]*EndpointMetric `json:"endpoint_metrics"`
	LastUpdated     time.Time                  `json:"last_updated"`
	mu              sync.RWMutex
}

// EndpointMetric 端点指标
type EndpointMetric struct {
	Endpoint        server.EndPoint `json:"endpoint"`
	RequestCount    int64           `json:"request_count"`
	SuccessCount    int64           `json:"success_count"`
	FailureCount    int64           `json:"failure_count"`
	AverageLatency  float64         `json:"average_latency_ms"`
	LastRequestTime time.Time       `json:"last_request_time"`
	Healthy         bool            `json:"healthy"`
}

// LoadBalancerMonitor 负载均衡监控器
type LoadBalancerMonitor struct {
	metrics  *LoadBalancerMetrics
	balancer server.Balancer
	ctx      context.Context
	cancel   context.CancelFunc
}

// NewLoadBalancerMonitor 创建负载均衡监控器
func NewLoadBalancerMonitor(balancer server.Balancer) *LoadBalancerMonitor {
	ctx, cancel := context.WithCancel(context.Background())
	return &LoadBalancerMonitor{
		metrics: &LoadBalancerMetrics{
			EndpointMetrics: make(map[string]*EndpointMetric),
			LastUpdated:     time.Now(),
		},
		balancer: balancer,
		ctx:      ctx,
		cancel:   cancel,
	}
}

// RecordRequest 记录请求
func (lbm *LoadBalancerMonitor) RecordRequest(endpoint server.EndPoint, latency time.Duration, success bool) {
	lbm.metrics.mu.Lock()
	defer lbm.metrics.mu.Unlock()

	key := fmt.Sprintf("%s:%d", endpoint.Ip, endpoint.Port)

	// 更新全局指标
	lbm.metrics.TotalRequests++
	if success {
		lbm.metrics.SuccessRequests++
	} else {
		lbm.metrics.FailedRequests++
	}

	// 更新平均延迟
	totalLatency := lbm.metrics.AverageLatency * float64(lbm.metrics.TotalRequests-1)
	lbm.metrics.AverageLatency = (totalLatency + float64(latency.Milliseconds())) / float64(lbm.metrics.TotalRequests)

	// 更新端点指标
	if metric, exists := lbm.metrics.EndpointMetrics[key]; exists {
		metric.RequestCount++
		if success {
			metric.SuccessCount++
		} else {
			metric.FailureCount++
		}

		// 更新端点平均延迟
		totalEndpointLatency := metric.AverageLatency * float64(metric.RequestCount-1)
		metric.AverageLatency = (totalEndpointLatency + float64(latency.Milliseconds())) / float64(metric.RequestCount)
		metric.LastRequestTime = time.Now()
	} else {
		lbm.metrics.EndpointMetrics[key] = &EndpointMetric{
			Endpoint:     endpoint,
			RequestCount: 1,
			SuccessCount: func() int64 {
				if success {
					return 1
				}
				return 0
			}(),
			FailureCount: func() int64 {
				if !success {
					return 1
				}
				return 0
			}(),
			AverageLatency:  float64(latency.Milliseconds()),
			LastRequestTime: time.Now(),
			Healthy:         success,
		}
	}

	lbm.metrics.LastUpdated = time.Now()
}

// GetMetrics 获取指标
func (lbm *LoadBalancerMonitor) GetMetrics() *LoadBalancerMetrics {
	lbm.metrics.mu.RLock()
	defer lbm.metrics.mu.RUnlock()

	// 深拷贝指标
	metrics := &LoadBalancerMetrics{
		TotalRequests:   lbm.metrics.TotalRequests,
		SuccessRequests: lbm.metrics.SuccessRequests,
		FailedRequests:  lbm.metrics.FailedRequests,
		AverageLatency:  lbm.metrics.AverageLatency,
		EndpointMetrics: make(map[string]*EndpointMetric),
		LastUpdated:     lbm.metrics.LastUpdated,
	}

	for k, v := range lbm.metrics.EndpointMetrics {
		metrics.EndpointMetrics[k] = &EndpointMetric{
			Endpoint:        v.Endpoint,
			RequestCount:    v.RequestCount,
			SuccessCount:    v.SuccessCount,
			FailureCount:    v.FailureCount,
			AverageLatency:  v.AverageLatency,
			LastRequestTime: v.LastRequestTime,
			Healthy:         v.Healthy,
		}
	}

	return metrics
}

// ServeHTTP 提供HTTP指标接口
func (lbm *LoadBalancerMonitor) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	metrics := lbm.GetMetrics()

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(metrics); err != nil {
		http.Error(w, "Failed to encode metrics", http.StatusInternalServerError)
		return
	}
}

// Start 启动监控
func (lbm *LoadBalancerMonitor) Start() {
	go lbm.periodicCleanup()
}

// Stop 停止监控
func (lbm *LoadBalancerMonitor) Stop() {
	lbm.cancel()
}

// periodicCleanup 定期清理过期数据
func (lbm *LoadBalancerMonitor) periodicCleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-lbm.ctx.Done():
			return
		case <-ticker.C:
			lbm.cleanup()
		}
	}
}

// cleanup 清理过期的端点指标
func (lbm *LoadBalancerMonitor) cleanup() {
	lbm.metrics.mu.Lock()
	defer lbm.metrics.mu.Unlock()

	now := time.Now()
	for key, metric := range lbm.metrics.EndpointMetrics {
		// 如果端点超过10分钟没有请求，删除其指标
		if now.Sub(metric.LastRequestTime) > 10*time.Minute {
			delete(lbm.metrics.EndpointMetrics, key)
		}
	}
}
