package test

import (
	"VectorSphere/src/library/acceler"
	"encoding/json"
	"fmt"
	"gopkg.in/yaml.v2"
	"reflect"
	"sync"
	"testing"
	"time"
	"unsafe"
)

func TestCPUConfigMarshaling(t *testing.T) {
	config := acceler.CPUConfig{
		Enable:      true,
		IndexType:   "HNSW",
		DeviceID:    0,
		Threads:     8,
		VectorWidth: 256,
	}

	// Test JSON marshaling
	jsonData, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("Failed to marshal CPUConfig to JSON: %v", err)
	}

	var unmarshaledJSON acceler.CPUConfig
	err = json.Unmarshal(jsonData, &unmarshaledJSON)
	if err != nil {
		t.Fatalf("Failed to unmarshal CPUConfig from JSON: %v", err)
	}

	if !reflect.DeepEqual(config, unmarshaledJSON) {
		t.Errorf("JSON marshaling/unmarshaling failed. Expected: %+v, Got: %+v", config, unmarshaledJSON)
	}

	// Test YAML marshaling
	yamlData, err := yaml.Marshal(config)
	if err != nil {
		t.Fatalf("Failed to marshal CPUConfig to YAML: %v", err)
	}

	var unmarshaledYAML acceler.CPUConfig
	err = yaml.Unmarshal(yamlData, &unmarshaledYAML)
	if err != nil {
		t.Fatalf("Failed to unmarshal CPUConfig from YAML: %v", err)
	}

	if !reflect.DeepEqual(config, unmarshaledYAML) {
		t.Errorf("YAML marshaling/unmarshaling failed. Expected: %+v, Got: %+v", config, unmarshaledYAML)
	}
}

func TestAccelResultDataStorage(t *testing.T) {
	metadata := map[string]interface{}{
		"category": "test",
		"score":    0.95,
		"tags":     []string{"tag1", "tag2"},
	}

	result := acceler.AccelResult{
		ID:         "doc123",
		Similarity: 0.85,
		Distance:   0.15,
		Metadata:   metadata,
		DocIds:     []string{"doc123", "doc456"},
		Vector:     []float64{1.0, 2.0, 3.0, 4.0},
		Index:      42,
	}

	// Verify all fields are properly stored
	if result.ID != "doc123" {
		t.Errorf("Expected ID 'doc123', got '%s'", result.ID)
	}

	if result.Similarity != 0.85 {
		t.Errorf("Expected Similarity 0.85, got %f", result.Similarity)
	}

	if result.Distance != 0.15 {
		t.Errorf("Expected Distance 0.15, got %f", result.Distance)
	}

	if len(result.DocIds) != 2 || result.DocIds[0] != "doc123" || result.DocIds[1] != "doc456" {
		t.Errorf("Expected DocIds ['doc123', 'doc456'], got %v", result.DocIds)
	}

	if len(result.Vector) != 4 || result.Vector[0] != 1.0 {
		t.Errorf("Expected Vector [1.0, 2.0, 3.0, 4.0], got %v", result.Vector)
	}

	if result.Index != 42 {
		t.Errorf("Expected Index 42, got %d", result.Index)
	}

	// Verify metadata integrity
	if result.Metadata["category"] != "test" {
		t.Errorf("Expected metadata category 'test', got '%v'", result.Metadata["category"])
	}

	if result.Metadata["score"] != 0.95 {
		t.Errorf("Expected metadata score 0.95, got %v", result.Metadata["score"])
	}
}

func TestCPUAcceleratorTypeAlias(t *testing.T) {
	// Create instances using both types
	var cpuAccel acceler.CPUAccelerator
	var cpuAccelAlias acceler.CpuAccelerator

	// Verify they are the same type
	cpuAccelType := reflect.TypeOf(cpuAccel)
	aliasType := reflect.TypeOf(cpuAccelAlias)

	if cpuAccelType != aliasType {
		t.Errorf("Type alias failed: CPUAccelerator and CpuAccelerator are not the same type")
	}

	// Test assignment compatibility
	cpuAccelAlias = cpuAccel
	cpuAccel = cpuAccelAlias

	// Verify pointer compatibility
	var cpuAccelPtr *acceler.CPUAccelerator
	var aliasPtr *acceler.CpuAccelerator

	cpuAccelPtr = &cpuAccel
	aliasPtr = cpuAccelPtr
	cpuAccelPtr = aliasPtr

	if cpuAccelPtr == nil || aliasPtr == nil {
		t.Error("Pointer assignment between CPUAccelerator and CpuAccelerator failed")
	}
}

func TestGPUAcceleratorUnsafePointers(t *testing.T) {
	gpu := &acceler.GPUAccelerator{
		OperationCount: 0,
		ErrorCount:     0,
		BatchSize:      32,
		StreamCount:    4,
		MemoryUsed:     0,
		MemoryTotal:    1024 * 1024 * 1024, // 1GB
		DeviceCount:    1,
	}

	// Test safe initialization of unsafe pointers
	gpu.GpuWrapper = unsafe.Pointer(uintptr(0))
	gpu.GpuResources = unsafe.Pointer(uintptr(0))

	// Verify pointers are properly set to null
	if gpu.GpuWrapper != unsafe.Pointer(uintptr(0)) {
		t.Error("gpuWrapper should be initialized to null pointer")
	}

	if gpu.GpuResources != unsafe.Pointer(uintptr(0)) {
		t.Error("gpuResources should be initialized to null pointer")
	}

	// Test pointer assignment and retrieval
	testData := make([]byte, 64)
	gpu.GpuWrapper = unsafe.Pointer(&testData[0])

	// Verify pointer is not null after assignment
	if gpu.GpuWrapper == unsafe.Pointer(uintptr(0)) {
		t.Error("gpuWrapper should not be null after assignment")
	}

	// Test pointer comparison
	originalPtr := gpu.GpuWrapper
	gpu.GpuWrapper = unsafe.Pointer(&testData[0])

	if gpu.GpuWrapper != originalPtr {
		t.Error("Pointer assignment should maintain same address for same data")
	}

	// Reset to null for cleanup
	gpu.GpuWrapper = unsafe.Pointer(uintptr(0))
	gpu.GpuResources = unsafe.Pointer(uintptr(0))
}

func TestPMemAcceleratorConcurrentCacheAccess(t *testing.T) {
	pmem := &acceler.PMemAccelerator{
		VectorCache: make(map[string][]float64),
		CacheMutex:  sync.RWMutex{},
		MemoryPool:  make(map[string][]float64),
		Namespaces:  make(map[string]*acceler.PMemNamespace),
	}

	const numGoroutines = 10
	const numOperations = 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines * 2) // readers and writers

	// Start writer goroutines
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				key := fmt.Sprintf("vector_%d_%d", id, j)
				vector := []float64{float64(id), float64(j), float64(id + j)}

				pmem.CacheMutex.Lock()
				pmem.VectorCache[key] = vector
				pmem.CacheMutex.Unlock()
			}
		}(i)
	}

	// Start reader goroutines
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				key := fmt.Sprintf("vector_%d_%d", id, j)

				pmem.CacheMutex.RLock()
				_, exists := pmem.VectorCache[key]
				pmem.CacheMutex.RUnlock()

				// Don't assert existence since readers and writers run concurrently
				_ = exists
			}
		}(i)
	}

	wg.Wait()

	// Verify final state
	pmem.CacheMutex.RLock()
	cacheSize := len(pmem.VectorCache)
	pmem.CacheMutex.RUnlock()

	expectedSize := numGoroutines * numOperations
	if cacheSize != expectedSize {
		t.Errorf("Expected cache size %d, got %d", expectedSize, cacheSize)
	}
}

func TestComplexConfigValidation(t *testing.T) {
	// Test PMemConfig with nested structures
	pmemConfig := acceler.PMemConfig{
		Enable:      true,
		DevicePaths: []string{"/dev/pmem0", "/dev/pmem1"},
		Mode:        "app_direct",
		Namespaces: []acceler.PMemNamespace{
			{
				Name:       "ns1",
				Size:       1024 * 1024 * 1024,
				Mode:       "fsdax",
				Alignment:  4096,
				SectorSize: 512,
				MapSync:    true,
			},
		},
		Interleaving: acceler.PMemInterleavingConfig{
			Enable:      true,
			Ways:        2,
			Granularity: 256,
			Alignment:   64,
		},
		Persistence: acceler.PMemPersistenceConfig{
			FlushStrategy:      "sync",
			FlushInterval:      time.Millisecond * 100,
			Checkpointing:      true,
			CheckpointInterval: time.Second * 30,
			RecoveryMode:       "fast",
			FlushMode:          "auto",
			SyncOnWrite:        true,
			ChecksumEnabled:    true,
			BackupEnabled:      false,
			BackupInterval:     time.Hour,
		},
		Performance: acceler.PMemPerformanceConfig{
			ReadAhead:        true,
			WriteBehind:      true,
			BatchSize:        64,
			QueueDepth:       32,
			NUMAOptimization: true,
			PrefetchEnabled:  true,
			CacheSize:        1024 * 1024,
			CompressionLevel: 3,
		},
		Reliability: acceler.PMemReliabilityConfig{
			ECC:                true,
			Scrubbing:          true,
			ScrubbingInterval:  time.Hour * 24,
			BadBlockManagement: true,
			WearLeveling:       true,
			ECCEnabled:         true,
			ScrubInterval:      time.Hour * 12,
			ErrorThreshold:     10,
			RepairEnabled:      true,
			MirrorEnabled:      false,
			MirrorDevices:      []string{},
		},
		DevicePath:        "/dev/pmem0",
		PoolSize:          1024 * 1024 * 1024,
		EnableCompression: true,
		EnableEncryption:  false,
	}

	// Test FPGAConfig with nested structures
	fpgaConfig := acceler.FPGAConfig{
		Enable:          true,
		DeviceIDs:       []int{0, 1},
		Bitstream:       "/path/to/bitstream.bit",
		ClockFrequency:  200,
		MemoryBandwidth: 1024 * 1024 * 1024,
		PipelineDepth:   8,
		Parallelism: acceler.FPGAParallelismConfig{
			ComputeUnits:   16,
			VectorWidth:    512,
			UnrollFactor:   4,
			PipelineStages: 8,
		},
		Optimization: acceler.FPGAOptimizationConfig{
			ResourceSharing:    true,
			MemoryOptimization: true,
			TimingOptimization: true,
			PowerOptimization:  false,
			AreaOptimization:   true,
		},
		Reconfiguration: acceler.FPGAReconfigurationConfig{
			Enable:                 true,
			PartialReconfiguration: true,
			ReconfigurationTime:    time.Second * 5,
			BitstreamCache:         true,
			HotSwap:                false,
		},
	}

	// Validate PMemConfig fields
	if !pmemConfig.Enable {
		t.Error("PMemConfig Enable should be true")
	}

	if len(pmemConfig.DevicePaths) != 2 {
		t.Errorf("Expected 2 device paths, got %d", len(pmemConfig.DevicePaths))
	}

	if pmemConfig.Namespaces[0].Size != 1024*1024*1024 {
		t.Errorf("Expected namespace size 1GB, got %d", pmemConfig.Namespaces[0].Size)
	}

	if pmemConfig.Persistence.FlushInterval != time.Millisecond*100 {
		t.Errorf("Expected flush interval 100ms, got %v", pmemConfig.Persistence.FlushInterval)
	}

	// Validate FPGAConfig fields
	if !fpgaConfig.Enable {
		t.Error("FPGAConfig Enable should be true")
	}

	if len(fpgaConfig.DeviceIDs) != 2 {
		t.Errorf("Expected 2 device IDs, got %d", len(fpgaConfig.DeviceIDs))
	}

	if fpgaConfig.Parallelism.ComputeUnits != 16 {
		t.Errorf("Expected 16 compute units, got %d", fpgaConfig.Parallelism.ComputeUnits)
	}

	if fpgaConfig.Reconfiguration.ReconfigurationTime != time.Second*5 {
		t.Errorf("Expected reconfiguration time 5s, got %v", fpgaConfig.Reconfiguration.ReconfigurationTime)
	}

	// Test JSON marshaling of complex configs
	pmemJSON, err := json.Marshal(pmemConfig)
	if err != nil {
		t.Fatalf("Failed to marshal PMemConfig: %v", err)
	}

	var unmarshaledPMem acceler.PMemConfig
	err = json.Unmarshal(pmemJSON, &unmarshaledPMem)
	if err != nil {
		t.Fatalf("Failed to unmarshal PMemConfig: %v", err)
	}

	fpgaJSON, err := json.Marshal(fpgaConfig)
	if err != nil {
		t.Fatalf("Failed to marshal FPGAConfig: %v", err)
	}

	var unmarshaledFPGA acceler.FPGAConfig
	err = json.Unmarshal(fpgaJSON, &unmarshaledFPGA)
	if err != nil {
		t.Fatalf("Failed to unmarshal FPGAConfig: %v", err)
	}
}
