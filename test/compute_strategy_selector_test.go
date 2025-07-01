package test

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"errors"
	"reflect"
	"testing"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

//go:generate mockgen -source=compute_strategy_selector.go -destination=mocks/mock_accelerator.go

type MockUnifiedAccelerator struct {
	ctrl     *gomock.Controller
	recorder *MockUnifiedAcceleratorMockRecorder
}

type MockUnifiedAcceleratorMockRecorder struct {
	mock *MockUnifiedAccelerator
}

func NewMockUnifiedAccelerator(ctrl *gomock.Controller) *MockUnifiedAccelerator {
	mock := &MockUnifiedAccelerator{ctrl: ctrl}
	mock.recorder = &MockUnifiedAcceleratorMockRecorder{mock}
	return mock
}

func (m *MockUnifiedAccelerator) EXPECT() *MockUnifiedAcceleratorMockRecorder {
	return m.recorder
}

// 基础生命周期管理
func (m *MockUnifiedAccelerator) GetType() string {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetType")
	ret0, _ := ret[0].(string)
	return ret0
}

func (mr *MockUnifiedAcceleratorMockRecorder) GetType() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetType", reflect.TypeOf((*MockUnifiedAccelerator)(nil).GetType))
}

func (m *MockUnifiedAccelerator) IsAvailable() bool {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "IsAvailable")
	ret0, _ := ret[0].(bool)
	return ret0
}

func (mr *MockUnifiedAcceleratorMockRecorder) IsAvailable() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "IsAvailable", reflect.TypeOf((*MockUnifiedAccelerator)(nil).IsAvailable))
}

func (m *MockUnifiedAccelerator) Initialize() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Initialize")
	ret0, _ := ret[0].(error)
	return ret0
}

func (mr *MockUnifiedAcceleratorMockRecorder) Initialize() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Initialize", reflect.TypeOf((*MockUnifiedAccelerator)(nil).Initialize))
}

func (m *MockUnifiedAccelerator) Shutdown() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Shutdown")
	ret0, _ := ret[0].(error)
	return ret0
}

func (mr *MockUnifiedAcceleratorMockRecorder) Shutdown() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Shutdown", reflect.TypeOf((*MockUnifiedAccelerator)(nil).Shutdown))
}

// 核心计算功能
func (m *MockUnifiedAccelerator) ComputeDistance(query []float64, vectors [][]float64) ([]float64, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ComputeDistance", query, vectors)
	ret0, _ := ret[0].([]float64)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

func (mr *MockUnifiedAcceleratorMockRecorder) ComputeDistance(query, vectors interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ComputeDistance", reflect.TypeOf((*MockUnifiedAccelerator)(nil).ComputeDistance), query, vectors)
}

func (m *MockUnifiedAccelerator) BatchComputeDistance(queries [][]float64, vectors [][]float64) ([][]float64, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "BatchComputeDistance", queries, vectors)
	ret0, _ := ret[0].([][]float64)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

func (mr *MockUnifiedAcceleratorMockRecorder) BatchComputeDistance(queries, vectors interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "BatchComputeDistance", reflect.TypeOf((*MockUnifiedAccelerator)(nil).BatchComputeDistance), queries, vectors)
}

func (m *MockUnifiedAccelerator) BatchSearch(queries [][]float64, database [][]float64, k int) ([][]acceler.AccelResult, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "BatchSearch", queries, database, k)
	ret0, _ := ret[0].([][]acceler.AccelResult)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

func (mr *MockUnifiedAcceleratorMockRecorder) BatchSearch(queries, database, k interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "BatchSearch", reflect.TypeOf((*MockUnifiedAccelerator)(nil).BatchSearch), queries, database, k)
}

func (m *MockUnifiedAccelerator) BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "BatchCosineSimilarity", queries, database)
	ret0, _ := ret[0].([][]float64)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

func (mr *MockUnifiedAcceleratorMockRecorder) BatchCosineSimilarity(queries, database interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "BatchCosineSimilarity", reflect.TypeOf((*MockUnifiedAccelerator)(nil).BatchCosineSimilarity), queries, database)
}

// 高级功能
func (m *MockUnifiedAccelerator) AccelerateSearch(query []float64, database [][]float64, options entity.SearchOptions) ([]acceler.AccelResult, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "AccelerateSearch", query, database, options)
	ret0, _ := ret[0].([]acceler.AccelResult)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

func (mr *MockUnifiedAcceleratorMockRecorder) AccelerateSearch(query, database, options interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "AccelerateSearch", reflect.TypeOf((*MockUnifiedAccelerator)(nil).AccelerateSearch), query, database, options)
}

// 能力和统计信息
func (m *MockUnifiedAccelerator) GetCapabilities() acceler.HardwareCapabilities {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetCapabilities")
	ret0, _ := ret[0].(acceler.HardwareCapabilities)
	return ret0
}

func (mr *MockUnifiedAcceleratorMockRecorder) GetCapabilities() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetCapabilities", reflect.TypeOf((*MockUnifiedAccelerator)(nil).GetCapabilities))
}

func (m *MockUnifiedAccelerator) GetStats() acceler.HardwareStats {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetStats")
	ret0, _ := ret[0].(acceler.HardwareStats)
	return ret0
}

func (mr *MockUnifiedAcceleratorMockRecorder) GetStats() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetStats", reflect.TypeOf((*MockUnifiedAccelerator)(nil).GetStats))
}

func (m *MockUnifiedAccelerator) GetPerformanceMetrics() acceler.PerformanceMetrics {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetPerformanceMetrics")
	ret0, _ := ret[0].(acceler.PerformanceMetrics)
	return ret0
}

func (mr *MockUnifiedAcceleratorMockRecorder) GetPerformanceMetrics() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetPerformanceMetrics", reflect.TypeOf((*MockUnifiedAccelerator)(nil).GetPerformanceMetrics))
}

// 配置和调优
func (m *MockUnifiedAccelerator) AutoTune(workload acceler.WorkloadProfile) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "AutoTune", workload)
	ret0, _ := ret[0].(error)
	return ret0
}

func (mr *MockUnifiedAcceleratorMockRecorder) AutoTune(workload interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "AutoTune", reflect.TypeOf((*MockUnifiedAccelerator)(nil).AutoTune), workload)
}

type MockHardwareDetector struct {
	ctrl     *gomock.Controller
	recorder *MockHardwareDetectorMockRecorder
}

type MockHardwareDetectorMockRecorder struct {
	mock *MockHardwareDetector
}

func NewMockHardwareDetector(ctrl *gomock.Controller) *MockHardwareDetector {
	mock := &MockHardwareDetector{ctrl: ctrl}
	mock.recorder = &MockHardwareDetectorMockRecorder{mock}
	return mock
}

func (m *MockHardwareDetector) EXPECT() *MockHardwareDetectorMockRecorder {
	return m.recorder
}

func (m *MockHardwareDetector) GetHardwareCapabilities() acceler.HardwareCapabilities {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetHardwareCapabilities")
	ret0, _ := ret[0].(acceler.HardwareCapabilities)
	return ret0
}

func (mr *MockHardwareDetectorMockRecorder) GetHardwareCapabilities() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetHardwareCapabilities", reflect.TypeOf((*MockHardwareDetector)(nil).GetHardwareCapabilities))
}

// HardwareManagerInterface 定义硬件管理器接口
type HardwareManagerInterface interface {
	GetAccelerator(name string) (acceler.UnifiedAccelerator, bool)
	GetAcceleratorConfig(acceleratorType string) (interface{}, error)
}

type MockHardwareManager struct {
	ctrl     *gomock.Controller
	recorder *MockHardwareManagerMockRecorder
}

type MockHardwareManagerMockRecorder struct {
	mock *MockHardwareManager
}

func NewMockHardwareManager(ctrl *gomock.Controller) *MockHardwareManager {
	mock := &MockHardwareManager{ctrl: ctrl}
	mock.recorder = &MockHardwareManagerMockRecorder{mock}
	return mock
}

func (m *MockHardwareManager) EXPECT() *MockHardwareManagerMockRecorder {
	return m.recorder
}

func (m *MockHardwareManager) GetAccelerator(name string) (acceler.UnifiedAccelerator, bool) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetAccelerator", name)
	ret0, _ := ret[0].(acceler.UnifiedAccelerator)
	ret1, _ := ret[1].(bool)
	return ret0, ret1
}

func (mr *MockHardwareManagerMockRecorder) GetAccelerator(name interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetAccelerator", reflect.TypeOf((*MockHardwareManager)(nil).GetAccelerator), name)
}

func (m *MockHardwareManager) GetAcceleratorConfig(acceleratorType string) (interface{}, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetAcceleratorConfig", acceleratorType)
	ret0 := ret[0]
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

func (mr *MockHardwareManagerMockRecorder) GetAcceleratorConfig(acceleratorType interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetAcceleratorConfig", reflect.TypeOf((*MockHardwareManager)(nil).GetAcceleratorConfig), acceleratorType)
}

// TestComputeStrategySelector 测试用的策略选择器结构体
type TestComputeStrategySelector struct {
	Detector        *MockHardwareDetector
	HardwareManager HardwareManagerInterface
	GpuThreshold    int
	FpgaThreshold   int
	RdmaThreshold   int
}

// SelectOptimalStrategy 选择最优策略的测试方法
func (css *TestComputeStrategySelector) SelectOptimalStrategy(dataSize int, vectorDim int) acceler.ComputeStrategy {
	caps := css.Detector.GetHardwareCapabilities()
	return css.selectStrategyWithHardwareManager(dataSize, vectorDim, caps)
}

// selectStrategyWithHardwareManager 使用硬件管理器选择策略
func (css *TestComputeStrategySelector) selectStrategyWithHardwareManager(dataSize int, vectorDim int, caps acceler.HardwareCapabilities) acceler.ComputeStrategy {
	// 如果没有硬件管理器，直接返回标准策略
	if css.HardwareManager == nil {
		return acceler.StrategyStandard
	}

	// 超大数据量优先考虑RDMA分布式计算
	if dataSize >= css.RdmaThreshold {
		if rdmaAcc, exists := css.HardwareManager.GetAccelerator(acceler.AcceleratorRDMA); exists && rdmaAcc != nil && rdmaAcc.IsAvailable() {
			return acceler.StrategyRDMA
		}
	}

	// 大数据量考虑FPGA
	if dataSize >= css.FpgaThreshold {
		if fpgaAcc, exists := css.HardwareManager.GetAccelerator(acceler.AcceleratorFPGA); exists && fpgaAcc != nil && fpgaAcc.IsAvailable() {
			return acceler.StrategyFPGA
		}
	}

	// 中等数据量考虑GPU
	if dataSize >= css.GpuThreshold && caps.HasGPU {
		if gpuAcc, exists := css.HardwareManager.GetAccelerator(acceler.AcceleratorGPU); exists && gpuAcc != nil && gpuAcc.IsAvailable() {
			return acceler.StrategyGPU
		}
	}

	// CPU策略选择
	if cpuAcc, exists := css.HardwareManager.GetAccelerator(acceler.AcceleratorCPU); exists && cpuAcc != nil && cpuAcc.IsAvailable() {
		if caps.HasAVX512 && vectorDim%8 == 0 {
			if config, err := css.HardwareManager.GetAcceleratorConfig(acceler.AcceleratorCPU); err == nil {
				if cpuConfig, ok := config.(acceler.CPUConfig); ok && cpuConfig.Enable && cpuConfig.VectorWidth >= 512 {
					return acceler.StrategyAVX512
				}
			}
		}
		if caps.HasAVX2 && vectorDim%8 == 0 {
			return acceler.StrategyAVX2
		}
	}

	return acceler.StrategyStandard
}

// VerifyCPUSupport 验证CPU支持
func (css *TestComputeStrategySelector) VerifyCPUSupport(acceleratorType string, instructionSet string) bool {
	config, err := css.HardwareManager.GetAcceleratorConfig(acceleratorType)
	if err != nil {
		return false
	}

	cpuConfig, ok := config.(acceler.CPUConfig)
	if !ok {
		return false
	}

	return cpuConfig.Enable
}

func TestSelectOptimalStrategy_RDMAForLargeDataset(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockDetector := NewMockHardwareDetector(ctrl)
	mockHardwareManager := NewMockHardwareManager(ctrl)
	mockRDMAAcc := NewMockUnifiedAccelerator(ctrl)

	css := &TestComputeStrategySelector{
		Detector:        mockDetector,
		HardwareManager: mockHardwareManager,
		RdmaThreshold:   100000,
		FpgaThreshold:   50000,
		GpuThreshold:    10000,
	}

	caps := acceler.HardwareCapabilities{
		HasAVX2:   true,
		HasAVX512: true,
		HasGPU:    true,
	}

	mockDetector.EXPECT().GetHardwareCapabilities().Return(caps).AnyTimes()
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorRDMA).Return(mockRDMAAcc, true).AnyTimes()
	mockRDMAAcc.EXPECT().IsAvailable().Return(true).AnyTimes()

	strategy := css.SelectOptimalStrategy(150000, 128)
	assert.Equal(t, acceler.StrategyRDMA, strategy)
}

func TestSelectOptimalStrategy_GPUForMediumDataset(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockDetector := NewMockHardwareDetector(ctrl)
	mockHardwareManager := NewMockHardwareManager(ctrl)
	mockGPUAcc := NewMockUnifiedAccelerator(ctrl)

	css := &TestComputeStrategySelector{
		Detector:        mockDetector,
		HardwareManager: mockHardwareManager,
		GpuThreshold:    10000,
		FpgaThreshold:   50000,
		RdmaThreshold:   100000,
	}

	caps := acceler.HardwareCapabilities{
		HasGPU: true,
	}

	mockDetector.EXPECT().GetHardwareCapabilities().Return(caps)
	// 数据大小25000满足GPU阈值10000，但不满足FPGA(50000)和RDMA(100000)阈值
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorGPU).Return(mockGPUAcc, true)
	mockGPUAcc.EXPECT().IsAvailable().Return(true)

	strategy := css.SelectOptimalStrategy(25000, 128)
	assert.Equal(t, acceler.StrategyGPU, strategy)
}

func TestSelectOptimalStrategy_AVX512WithAlignedVectors(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockDetector := NewMockHardwareDetector(ctrl)
	mockHardwareManager := NewMockHardwareManager(ctrl)
	mockCPUAcc := NewMockUnifiedAccelerator(ctrl)

	css := &TestComputeStrategySelector{
		Detector:        mockDetector,
		HardwareManager: mockHardwareManager,
		GpuThreshold:    10000,
		FpgaThreshold:   50000,
		RdmaThreshold:   100000,
	}

	caps := acceler.HardwareCapabilities{
		HasAVX512: true,
	}

	cpuConfig := acceler.CPUConfig{
		Enable:      true,
		VectorWidth: 512,
	}

	mockDetector.EXPECT().GetHardwareCapabilities().Return(caps)
	// 数据大小5000不满足GPU(10000)、FPGA(50000)、RDMA(100000)阈值，直接检查CPU
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorCPU).Return(mockCPUAcc, true)
	mockCPUAcc.EXPECT().IsAvailable().Return(true)
	mockHardwareManager.EXPECT().GetAcceleratorConfig(acceler.AcceleratorCPU).Return(cpuConfig, nil)

	strategy := css.SelectOptimalStrategy(5000, 128)
	assert.Equal(t, acceler.StrategyAVX512, strategy)
}

func TestSelectOptimalStrategy_FallbackToStandardWithoutHardware(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockDetector := NewMockHardwareDetector(ctrl)

	css := &TestComputeStrategySelector{
		Detector:        mockDetector,
		HardwareManager: nil,
		GpuThreshold:    10000,
	}

	caps := acceler.HardwareCapabilities{}
	mockDetector.EXPECT().GetHardwareCapabilities().Return(caps)

	strategy := css.SelectOptimalStrategy(5000, 127)
	assert.Equal(t, acceler.StrategyStandard, strategy)
}

func TestSelectOptimalStrategy_AcceleratorUnavailable(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockDetector := NewMockHardwareDetector(ctrl)
	mockHardwareManager := NewMockHardwareManager(ctrl)
	mockGPUAcc := NewMockUnifiedAccelerator(ctrl)

	css := &TestComputeStrategySelector{
		Detector:        mockDetector,
		HardwareManager: mockHardwareManager,
		GpuThreshold:    10000,
		FpgaThreshold:   50000,
		RdmaThreshold:   100000,
	}

	caps := acceler.HardwareCapabilities{
		HasGPU: true,
	}

	mockDetector.EXPECT().GetHardwareCapabilities().Return(caps)
	// 数据大小25000只会触发GPU检查，不会检查RDMA和FPGA
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorGPU).Return(mockGPUAcc, true)
	mockGPUAcc.EXPECT().IsAvailable().Return(false)
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorCPU).Return(nil, false)

	strategy := css.SelectOptimalStrategy(25000, 128)
	assert.Equal(t, acceler.StrategyStandard, strategy)
}

func TestVerifyCPUSupport_InvalidConfigTypeAssertion(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockHardwareManager := NewMockHardwareManager(ctrl)

	css := &TestComputeStrategySelector{
		HardwareManager: mockHardwareManager,
	}

	// Return invalid config type (string instead of CPUConfig)
	mockHardwareManager.EXPECT().GetAcceleratorConfig(acceler.AcceleratorCPU).Return("invalid_config", nil)

	result := css.VerifyCPUSupport(acceler.AcceleratorCPU, "avx512")
	assert.False(t, result)

	// Test with error from GetAcceleratorConfig
	mockHardwareManager.EXPECT().GetAcceleratorConfig(acceler.AcceleratorCPU).Return(nil, errors.New("config error"))

	result = css.VerifyCPUSupport(acceler.AcceleratorCPU, "avx512")
	assert.False(t, result)
}
