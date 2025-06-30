package test

import (
	"VectorSphere/src/library/acceler"
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

func TestSelectOptimalStrategy_RDMAForLargeDataset(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockDetector := NewMockHardwareDetector(ctrl)
	mockHardwareManager := NewMockHardwareManager(ctrl)
	mockRDMAAcc := NewMockUnifiedAccelerator(ctrl)

	css := &acceler.ComputeStrategySelector{
		Detector:        mockDetector,
		HardwareManager: mockHardwareManager,
		RdmaThreshold:   100000,
	}

	caps := acceler.HardwareCapabilities{
		HasAVX2:   true,
		HasAVX512: true,
		HasGPU:    true,
	}

	mockDetector.EXPECT().GetHardwareCapabilities().Return(caps)
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorRDMA).Return(mockRDMAAcc, true)
	mockRDMAAcc.EXPECT().IsAvailable().Return(true)

	strategy := css.SelectOptimalStrategy(150000, 128)
	assert.Equal(t, acceler.StrategyRDMA, strategy)
}

func TestSelectOptimalStrategy_GPUForMediumDataset(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockDetector := NewMockHardwareDetector(ctrl)
	mockHardwareManager := NewMockHardwareManager(ctrl)
	mockGPUAcc := NewMockUnifiedAccelerator(ctrl)

	css := &acceler.ComputeStrategySelector{
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
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorRDMA).Return(nil, false)
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorFPGA).Return(nil, false)
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

	css := &acceler.ComputeStrategySelector{
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
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorRDMA).Return(nil, false)
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorFPGA).Return(nil, false)
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorGPU).Return(nil, false)
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorCPU).Return(mockCPUAcc, true)
	mockCPUAcc.EXPECT().IsAvailable().Return(true)
	mockHardwareManager.EXPECT().GetAcceleratorConfig(acceler.AcceleratorCPU).Return(cpuConfig, nil)

	strategy := css.SelectOptimalStrategy(5000, 128)
	assert.Equal(t, acceler.StrategyAVX512, strategy)
}

func TestSelectOptimalStrategy_FallbackToStandardWithoutHardware(t *testing.T) {
	mockDetector := &acceler.HardwareDetector{}

	css := &acceler.ComputeStrategySelector{
		Detector:        mockDetector,
		HardwareManager: nil,
		GpuThreshold:    10000,
	}

	strategy := css.SelectOptimalStrategy(5000, 127)
	assert.Equal(t, acceler.StrategyStandard, strategy)
}

func TestSelectOptimalStrategy_AcceleratorUnavailable(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockDetector := NewMockHardwareDetector(ctrl)
	mockHardwareManager := NewMockHardwareManager(ctrl)
	mockGPUAcc := NewMockUnifiedAccelerator(ctrl)

	css := &acceler.ComputeStrategySelector{
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
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorRDMA).Return(nil, false)
	mockHardwareManager.EXPECT().GetAccelerator(acceler.AcceleratorFPGA).Return(nil, false)
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

	css := &acceler.ComputeStrategySelector{
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
