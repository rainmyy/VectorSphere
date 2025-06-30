package test

import (
	"VectorSphere/src/library/acceler"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

func TestLoggerLogsMessagesAtConfiguredLevel(t *testing.T) {
	logger := &acceler.Logger{
		Level:  acceler.LogLevelInfo,
		Prefix: "[Test]",
	}

	// Test that Info level message is logged when level is Info
	logger.Info("Test info message")

	// Test that Error level message is logged when level is Info
	logger.Error("Test error message")

	// Test that Warn level message is logged when level is Info
	logger.Warn("Test warn message")
}

func TestPerformanceMonitorTracksSuccessfulOperations(t *testing.T) {
	pm := &acceler.PerformanceMonitor{
		Metrics:   make(map[string]*acceler.OperationMetrics),
		StartTime: time.Now(),
	}

	operation := "test_operation"
	tracker := pm.StartOperation(operation)

	// Simulate some work
	time.Sleep(10 * time.Millisecond)

	tracker.End(true)

	metrics := pm.GetMetrics(operation)
	assert.NotNil(t, metrics)
	assert.Equal(t, int64(1), metrics.TotalCalls)
	assert.Equal(t, int64(1), metrics.SuccessCalls)
	assert.Equal(t, int64(0), metrics.FailedCalls)
	assert.True(t, metrics.TotalDuration > 0)
	assert.Equal(t, 1.0, pm.GetSuccessRate(operation))
}

func TestValidateConfigAcceptsValidConfigurations(t *testing.T) {
	// Test valid CPU config
	cpuConfig := &acceler.CPUConfig{
		Enable:      true,
		IndexType:   "ivf",
		DeviceID:    0,
		Threads:     4,
		VectorWidth: 256,
	}
	err := acceler.ValidateConfig(cpuConfig)
	assert.NoError(t, err)

	// Test valid GPU config
	gpuConfig := &acceler.GPUConfig{
		Enable:      true,
		DeviceIDs:   []int{0, 1},
		MemoryLimit: 1024 * 1024 * 1024,
		BatchSize:   32,
		Precision:   "fp32",
		IndexType:   "ivf",
	}
	err = acceler.ValidateConfig(gpuConfig)
	assert.NoError(t, err)

	// Test valid FPGA config
	fpgaConfig := &acceler.FPGAConfig{
		Enable:          true,
		DeviceIDs:       []int{0},
		Bitstream:       "/path/to/bitstream.bit",
		ClockFrequency:  200,
		MemoryBandwidth: 1000000000,
		PipelineDepth:   8,
	}
	err = acceler.ValidateConfig(fpgaConfig)
	assert.NoError(t, err)

	// Test valid RDMA config
	rdmaConfig := &acceler.RDMAConfig{
		Enable:    true,
		DeviceID:  0,
		PortNum:   8080,
		QueueSize: 1024,
		Protocol:  "IB",
	}
	err = acceler.ValidateConfig(rdmaConfig)
	assert.NoError(t, err)

	// Test valid PMem config
	pmemConfig := &acceler.PMemConfig{
		Enable:      true,
		DevicePaths: []string{"/dev/pmem0"},
		Mode:        "app_direct",
		PoolSize:    1024 * 1024 * 1024,
	}
	err = acceler.ValidateConfig(pmemConfig)
	assert.NoError(t, err)
}

func TestLoggerFiltersMessagesBelowThreshold(t *testing.T) {
	logger := &acceler.Logger{
		Level:  acceler.LogLevelError,
		Prefix: "[Test]",
	}

	// These should be filtered out (not logged)
	logger.Debug("Debug message should be filtered")
	logger.Info("Info message should be filtered")
	logger.Warn("Warn message should be filtered")

	// This should be logged
	logger.Error("Error message should be logged")
	logger.Fatal("Fatal message should be logged")

	// Verify the level filtering logic
	assert.True(t, acceler.LogLevelDebug < logger.Level)
	assert.True(t, acceler.LogLevelInfo < logger.Level)
	assert.True(t, acceler.LogLevelWarn < logger.Level)
	assert.False(t, acceler.LogLevelError < logger.Level)
	assert.False(t, acceler.LogLevelFatal < logger.Level)
}

func TestPerformanceMonitorConcurrentAccess(t *testing.T) {
	pm := &acceler.PerformanceMonitor{
		Metrics:   make(map[string]*acceler.OperationMetrics),
		StartTime: time.Now(),
	}

	const numGoroutines = 100
	const numOperations = 10
	var wg sync.WaitGroup

	// Launch multiple goroutines to access metrics concurrently
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			for j := 0; j < numOperations; j++ {
				operation := "concurrent_test"
				tracker := pm.StartOperation(operation)
				time.Sleep(time.Millisecond)
				tracker.End(true)

				// Also test concurrent reads
				metrics := pm.GetMetrics(operation)
				if metrics != nil {
					_ = pm.GetAverageLatency(operation)
					_ = pm.GetSuccessRate(operation)
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify final metrics
	metrics := pm.GetMetrics("concurrent_test")
	assert.NotNil(t, metrics)
	assert.Equal(t, int64(numGoroutines*numOperations), metrics.TotalCalls)
	assert.Equal(t, int64(numGoroutines*numOperations), metrics.SuccessCalls)
	assert.Equal(t, int64(0), metrics.FailedCalls)
}

func TestRetryOperationExhaustsMaxRetries(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	maxRetries := 3
	interval := 10 * time.Millisecond
	callCount := 0
	expectedError := errors.New("operation failed")

	// Function that always fails
	failingFn := func() error {
		callCount++
		return expectedError
	}

	start := time.Now()
	err := acceler.RetryOperation("test_accelerator", "failing_operation", failingFn, maxRetries, interval)
	duration := time.Since(start)

	// Verify error is returned
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "在 3 次重试后仍然失败")

	// Verify function was called maxRetries + 1 times (initial + retries)
	assert.Equal(t, maxRetries+1, callCount)

	// Verify total duration includes retry delays
	expectedMinDuration := time.Duration(maxRetries) * interval
	assert.True(t, duration >= expectedMinDuration, "Duration should include retry delays")
}
