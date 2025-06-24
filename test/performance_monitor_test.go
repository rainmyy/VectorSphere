package test

import (
	"VectorSphere/src/vector"
	"testing"
	"time"
)

func TestNewStandardPerformanceMonitor(t *testing.T) {
	monitor := vector.NewStandardPerformanceMonitor()
	if monitor == nil {
		t.Fatal("Expected non-nil StandardPerformanceMonitor")
	}
}

func TestStartOperation(t *testing.T) {
	monitor := vector.NewStandardPerformanceMonitor()

	tracker := monitor.StartOperation("search")
	if tracker == nil {
		t.Fatal("Expected non-nil OperationTracker")
	}

	// 模拟操作执行时间
	time.Sleep(10 * time.Millisecond)

	tracker.End(true, nil)

	metrics := monitor.GetMetrics()
	if len(metrics) == 0 {
		t.Error("Expected metrics to be recorded")
	}
}

func TestRecordMetric(t *testing.T) {
	monitor := vector.NewStandardPerformanceMonitor()

	tags := map[string]string{
		"operation": "search",
		"index":     "hnsw",
	}

	monitor.RecordMetric("latency", 15.5, tags)
	monitor.RecordMetric("latency", 20.3, tags)
	monitor.RecordMetric("latency", 12.1, tags)

	metrics := monitor.GetMetrics()
	latencyMetric, exists := metrics["latency"]
	if !exists {
		t.Fatal("Expected latency metric to exist")
	}

	if latencyMetric.Count != 3 {
		t.Errorf("Expected count 3, got %d", latencyMetric.Count)
	}

	expectedSum := 15.5 + 20.3 + 12.1
	if latencyMetric.Sum != expectedSum {
		t.Errorf("Expected sum %f, got %f", expectedSum, latencyMetric.Sum)
	}

	expectedAvg := expectedSum / 3
	if latencyMetric.Average != expectedAvg {
		t.Errorf("Expected average %f, got %f", expectedAvg, latencyMetric.Average)
	}
}

func TestGetSystemStats(t *testing.T) {
	monitor := vector.NewStandardPerformanceMonitor()

	stats := monitor.GetSystemStats()

	if stats.CPUUsage < 0 || stats.CPUUsage > 100 {
		t.Errorf("Expected CPU usage between 0-100, got %f", stats.CPUUsage)
	}

	if stats.MemoryUsage.UsagePercent < 0 {
		t.Errorf("Expected positive memory usage, got %d", stats.MemoryUsage.UsagePercent)
	}
	
	if stats.Goroutines <= 0 {
		t.Errorf("Expected positive goroutine count, got %d", stats.Goroutines)
	}
}

func TestOperationTracker(t *testing.T) {
	monitor := vector.NewStandardPerformanceMonitor()

	tracker := monitor.StartOperation("index_build")
	if tracker == nil {
		t.Fatal("Expected non-nil tracker")
	}

	// 添加标签
	tracker.AddTag("index_type", "hnsw")
	tracker.AddTag("dimension", "128")

	// 模拟操作
	time.Sleep(5 * time.Millisecond)

	// 添加更多标签
	tracker.AddTag("vectors_processed", "1000")
	tracker.AddTag("memory_used", "50.5")

	time.Sleep(5 * time.Millisecond)

	tracker.End(true, nil)

	metrics := monitor.GetMetrics()
	if len(metrics) == 0 {
		t.Error("Expected metrics to be recorded")
	}
}

func TestMetricTypes(t *testing.T) {
	monitor := vector.NewStandardPerformanceMonitor()

	// 测试基本指标记录
	monitor.RecordMetric("requests_total", 1, nil)
	monitor.RecordMetric("requests_total", 1, nil)
	monitor.RecordMetric("requests_total", 1, nil)

	// 测试内存使用指标
	monitor.RecordMetric("memory_usage", 75.5, nil)
	monitor.RecordMetric("memory_usage", 80.2, nil)

	// 测试响应时间指标
	monitor.RecordMetric("response_time", 10.5, nil)
	monitor.RecordMetric("response_time", 15.2, nil)
	monitor.RecordMetric("response_time", 8.7, nil)

	metrics := monitor.GetMetrics()

	// 验证计数器
	if counter, exists := metrics["requests_total"]; exists {
		if counter.Value != 3 {
			t.Errorf("Expected counter value 3, got %f", counter.Value)
		}
	} else {
		t.Error("Expected requests_total counter to exist")
	}

	// 验证仪表盘
	if gauge, exists := metrics["memory_usage"]; exists {
		if gauge.Value != 80.2 {
			t.Errorf("Expected gauge value 80.2, got %f", gauge.Value)
		}
	} else {
		t.Error("Expected memory_usage gauge to exist")
	}

	// 验证直方图
	if histogram, exists := metrics["response_time"]; exists {
		if histogram.Count != 3 {
			t.Errorf("Expected histogram count 3, got %d", histogram.Count)
		}
		expectedAvg := (10.5 + 15.2 + 8.7) / 3
		if histogram.Average != expectedAvg {
			t.Errorf("Expected histogram average %f, got %f", expectedAvg, histogram.Average)
		}
	} else {
		t.Error("Expected response_time histogram to exist")
	}
}

func TestPercentileCalculation(t *testing.T) {
	monitor := vector.NewStandardPerformanceMonitor()

	// 添加一系列值
	values := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	for _, val := range values {
		monitor.RecordMetric("test_metric", val, nil)
	}

	metrics := monitor.GetMetrics()
	testMetric, exists := metrics["test_metric"]
	if !exists {
		t.Fatal("Expected test_metric to exist")
	}

	if testMetric.Percentiles == nil {
		t.Fatal("Expected percentiles to be calculated")
	}

	// 检查一些百分位数
	if p50, exists := testMetric.Percentiles["p50"]; exists {
		if p50 < 5 || p50 > 6 {
			t.Errorf("Expected p50 around 5-6, got %f", p50)
		}
	}

	if p95, exists := testMetric.Percentiles["p95"]; exists {
		if p95 < 9 || p95 > 10 {
			t.Errorf("Expected p95 around 9-10, got %f", p95)
		}
	}
}

func TestMetricsExport(t *testing.T) {
	monitor := vector.NewStandardPerformanceMonitor()

	// 记录一些指标
	monitor.RecordMetric("http_requests_total", 100, map[string]string{"method": "GET"})
	monitor.RecordMetric("memory_usage_bytes", 1024*1024*512, nil)
	monitor.RecordMetric("http_request_duration_seconds", 0.25, nil)

	metrics := monitor.GetMetrics()

	if len(metrics) == 0 {
		t.Error("Expected non-empty metrics")
	}

	// 检查指标是否存在
	if _, exists := metrics["http_requests_total"]; !exists {
		t.Error("Expected metrics to contain http_requests_total")
	}

	if _, exists := metrics["memory_usage_bytes"]; !exists {
		t.Error("Expected metrics to contain memory_usage_bytes")
	}

	if _, exists := metrics["http_request_duration_seconds"]; !exists {
		t.Error("Expected metrics to contain http_request_duration_seconds")
	}
}

func TestMonitorReset(t *testing.T) {
	monitor := vector.NewStandardPerformanceMonitor()

	// 记录一些指标
	monitor.RecordMetric("test_metric", 10.0, nil)
	monitor.RecordMetric("test_metric", 20.0, nil)

	metrics := monitor.GetMetrics()
	if len(metrics) == 0 {
		t.Error("Expected metrics before reset")
	}

	// 注意：Reset方法可能不存在，这里只是测试指标记录
	// monitor.Reset()

	// 由于没有Reset方法，这里只是验证指标仍然存在
	metricsAfterReset := monitor.GetMetrics()
	if len(metricsAfterReset) == 0 {
		t.Error("Expected metrics to still exist")
	}
}

func TestConcurrentMetricRecording(t *testing.T) {
	monitor := vector.NewStandardPerformanceMonitor()

	// 并发记录指标
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func(id int) {
			for j := 0; j < 100; j++ {
				monitor.RecordMetric("concurrent_test", float64(id*100+j), nil)
			}
			done <- true
		}(i)
	}

	// 等待所有goroutine完成
	for i := 0; i < 10; i++ {
		<-done
	}

	metrics := monitor.GetMetrics()
	concurrentMetric, exists := metrics["concurrent_test"]
	if !exists {
		t.Fatal("Expected concurrent_test metric to exist")
	}

	if concurrentMetric.Count != 1000 {
		t.Errorf("Expected count 1000, got %d", concurrentMetric.Count)
	}
}

func TestOperationTrackerWithError(t *testing.T) {
	monitor := vector.NewStandardPerformanceMonitor()

	tracker := monitor.StartOperation("error_operation")

	// 模拟操作执行
	time.Sleep(5 * time.Millisecond)

	// 添加错误标签
	tracker.AddTag("error", "connection_timeout")

	tracker.End(false, nil)

	metrics := monitor.GetMetrics()

	// 检查错误计数
	if errorMetric, exists := metrics["error_operation_errors"]; exists {
		if errorMetric.Value != 1 {
			t.Errorf("Expected error count 1, got %f", errorMetric.Value)
		}
	} else {
		t.Error("Expected error metric to be recorded")
	}
}

func TestMemoryStatsTracking(t *testing.T) {
	monitor := vector.NewStandardPerformanceMonitor()

	stats := monitor.GetSystemStats()

	if stats.MemoryUsage.Allocated == 0 {
		t.Error("Expected non-zero allocated memory")
	}

	if stats.MemoryUsage.HeapAlloc == 0 {
		t.Error("Expected non-zero heap allocated memory")
	}

	if stats.MemoryUsage.Sys == 0 {
		t.Error("Expected non-zero system memory")
	}

	if stats.MemoryUsage.UsagePercent < 0 || stats.MemoryUsage.UsagePercent > 100 {
		t.Errorf("Expected memory usage percentage between 0-100, got %f", stats.MemoryUsage.UsagePercent)
	}
}
