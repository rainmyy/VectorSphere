package test

import (
	"context"
	"testing"
	"time"

	"VectorSphere/src/vector"
)

func TestNewAutoScaler(t *testing.T) {
	alertManager := &vector.AlertManager{}
	
	scaler := vector.NewAutoScaler(alertManager)
	if scaler == nil {
		t.Fatal("Expected non-nil AutoScaler")
	}
}

func TestScalingPolicy(t *testing.T) {
	policy := &vector.ScalingPolicy{
		Name:           "test-policy",
		MetricType:     vector.CPUUtilization,
		Threshold:      80.0,
		Direction:      vector.ScaleUp,
		CooldownPeriod: 5 * time.Minute,
		MinInstances:   1,
		MaxInstances:   10,
		ScalingFactor:  1.5,
		Enabled:        true,
	}
	
	if policy.Name != "test-policy" {
		t.Errorf("Expected policy name 'test-policy', got %s", policy.Name)
	}
	
	if policy.MetricType != vector.CPUUtilization {
		t.Errorf("Expected metric type CPUUtilization, got %s", policy.MetricType)
	}
}

func TestAutoScalerStart(t *testing.T) {
	alertManager := &vector.AlertManager{}
	
	scaler := vector.NewAutoScaler(alertManager)
	
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	
	// 启动自动扩缩容器
	go scaler.Start(ctx)
	
	// 等待一段时间让扩缩容器运行
	time.Sleep(100 * time.Millisecond)
	
	// 测试通过没有panic来验证启动成功
	scaler.Stop()
}

func TestAutoScalerStop(t *testing.T) {
	alertManager := &vector.AlertManager{}
	
	scaler := vector.NewAutoScaler(alertManager)
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	
	go scaler.Start(ctx)
	
	time.Sleep(100 * time.Millisecond)
	scaler.Stop()
	
	// 测试通过没有panic来验证停止成功
	time.Sleep(50 * time.Millisecond)
}

func TestAddScalingPolicy(t *testing.T) {
	alertManager := &vector.AlertManager{}
	
	scaler := vector.NewAutoScaler(alertManager)
	
	policy := &vector.ScalingPolicy{
		Name:           "cpu-scale-up",
		MetricType:     vector.CPUUtilization,
		Threshold:      80.0,
		Direction:      vector.ScaleUp,
		CooldownPeriod: 5 * time.Minute,
		MinInstances:   1,
		MaxInstances:   10,
		ScalingFactor:  0.5,
		Enabled:        true,
	}
	
	// 测试添加策略不会panic
	scaler.AddScalingPolicy(policy)
	
	// 由于没有GetScalingPolicies方法，我们只能测试添加操作不会出错
}

func TestCreateDefaultScalingPolicies(t *testing.T) {
	policies := vector.CreateDefaultScalingPolicies()
	
	if len(policies) == 0 {
		t.Error("Expected default scaling policies to be created")
	}
	
	// 检查默认策略的基本属性
	for _, policy := range policies {
		if policy.Name == "" {
			t.Error("Expected policy to have a name")
		}
		if policy.Threshold <= 0 {
			t.Error("Expected policy to have a positive threshold")
		}
		if policy.MinInstances <= 0 {
			t.Error("Expected policy to have positive min instances")
		}
		if policy.MaxInstances <= policy.MinInstances {
			t.Error("Expected max instances to be greater than min instances")
		}
	}
}