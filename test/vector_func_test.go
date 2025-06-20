package test

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"math"
	"testing"
)

func TestCosineSimilarity(t *testing.T) {
	vecA := []float64{1, 2, 3}
	vecB := []float64{4, 5, 6}
	// Expected: (1*4 + 2*5 + 3*6) / (sqrt(1^2+2^2+3^2) * sqrt(4^2+5^2+6^2))
	// = 32 / (sqrt(14) * sqrt(77)) = 32 / (3.7416 * 8.7749) = 32 / 32.83 = 0.9747
	expected := 0.974631846

	result := acceler.CosineSimilarity(vecA, vecB)

	if math.Abs(result-expected) > 1e-9 {
		t.Errorf("CosineSimilarity() = %v, want %v", result, expected)
	}

	// Test with zero vector
	vecC := []float64{0, 0, 0}
	result = acceler.CosineSimilarity(vecA, vecC)
	if result != 0 {
		t.Errorf("CosineSimilarity with zero vector = %v, want 0", result)
	}

	// Test with different lengths
	vecD := []float64{1, 2}
	result = acceler.CosineSimilarity(vecA, vecD)
	if result != -1 {
		t.Errorf("CosineSimilarity with different lengths = %v, want -1", result)
	}
}

func TestNormalizeVector(t *testing.T) {
	vec := []float64{3, 4}
	// norm = sqrt(3^2 + 4^2) = 5
	// expected = [3/5, 4/5] = [0.6, 0.8]
	expected := []float64{0.6, 0.8}

	result := acceler.NormalizeVector(vec)

	if len(result) != len(expected) {
		t.Fatalf("NormalizeVector() len = %v, want %v", len(result), len(expected))
	}

	for i := range result {
		if math.Abs(result[i]-expected[i]) > 1e-9 {
			t.Errorf("NormalizeVector() at index %d = %v, want %v", i, result[i], expected[i])
		}
	}

	// Test with zero vector
	vecZero := []float64{0, 0, 0}
	resultZero := acceler.NormalizeVector(vecZero)
	for i, v := range resultZero {
		if v != 0 {
			t.Errorf("NormalizeVector of zero vector at index %d = %v, want 0", i, v)
		}
	}
}

func TestOptimizedCosineSimilarity(t *testing.T) {
	vecA := []float64{1, 2, 3}
	vecB := []float64{4, 5, 6}

	normA := acceler.NormalizeVector(vecA)
	normB := acceler.NormalizeVector(vecB)

	expected := acceler.CosineSimilarity(vecA, vecB)
	result := acceler.OptimizedCosineSimilarity(normA, normB)

	if math.Abs(result-expected) > 1e-9 {
		t.Errorf("OptimizedCosineSimilarity() = %v, want %v", result, expected)
	}

	// Test with different lengths
	vecD := []float64{1, 2}
	result = acceler.OptimizedCosineSimilarity(normA, vecD)
	if result != -1 {
		t.Errorf("OptimizedCosineSimilarity with different lengths = %v, want -1", result)
	}
}

func TestPrecomputedDistanceTable(t *testing.T) {
	queryVector := []float64{0.1, 0.8}
	codebook := [][]entity.Point{
		{{0.0}, {1.0}}, // Subspace 0 centroids
		{{0.0}, {1.0}}, // Subspace 1 centroids
	}
	numSubvectors := 2

	dt, err := acceler.NewPrecomputedDistanceTable(queryVector, codebook, numSubvectors)
	if err != nil {
		t.Fatalf("NewPrecomputedDistanceTable() error = %v", err)
	}

	// Expected distances for subspace 0: (0.1-0)^2=0.01, (0.1-1)^2=0.81
	// Expected distances for subspace 1: (0.8-0)^2=0.64, (0.8-1)^2=0.04
	expectedTables := [][]float64{
		{0.01, 0.81},
		{0.64, 0.04},
	}

	for m, table := range dt.Tables {
		for c, dist := range table {
			if math.Abs(dist-expectedTables[m][c]) > 1e-9 {
				t.Errorf("Distance table at [%d][%d] is %f, want %f", m, c, dist, expectedTables[m][c])
			}
		}
	}

	compressedVector := entity.CompressedVector{Data: []byte{0, 1}} // Corresponds to centroids {0.0} and {1.0}
	totalDist, err := dt.ComputeDistance(compressedVector)
	if err != nil {
		t.Fatalf("ComputeDistance() error = %v", err)
	}

	// Expected total distance = dist(q_0, c_0_0) + dist(q_1, c_1_1) = 0.01 + 0.04 = 0.05
	expectedTotalDist := 0.05
	if math.Abs(totalDist-expectedTotalDist) > 1e-9 {
		t.Errorf("ComputeDistance() = %f, want %f", totalDist, expectedTotalDist)
	}
}

func TestCompressByPQ(t *testing.T) {
	// Mock codebook
	codebook := [][]entity.Point{
		{ // Subspace 0
			{0.0}, // Centroid 0
			{1.0}, // Centroid 1
		},
		{ // Subspace 1
			{0.0}, // Centroid 0
			{1.0}, // Centroid 1
		},
	}
	numSubVectors := 2
	numCentroidsPerSubVector := 2

	// Test vector
	vec := []float64{0.2, 0.9}

	// Expected compression
	// Sub-vector 0: {0.2}. Closest centroid is {0.0} (index 0).
	// Sub-vector 1: {0.9}. The closest centroid is {1.0} (index 1).
	expectedData := []byte{0, 1}

	compressed, err := acceler.CompressByPQ(vec, codebook, numSubVectors, numCentroidsPerSubVector)
	if err != nil {
		t.Fatalf("CompressByPQ failed: %v", err)
	}

	if len(compressed.Data) != len(expectedData) {
		t.Fatalf("Compressed data length is %d, want %d", len(compressed.Data), len(expectedData))
	}

	for i := range expectedData {
		if compressed.Data[i] != expectedData[i] {
			t.Errorf("Compressed data at index %d is %d, want %d", i, compressed.Data[i], expectedData[i])
		}
	}
}

func TestBatchCompressByPQ(t *testing.T) {
	codebook := [][]entity.Point{
		{{0.0}, {1.0}},
		{{0.0}, {1.0}},
	}
	numSubVectors := 2
	numCentroidsPerSubVector := 2

	vectors := [][]float64{
		{0.2, 0.9}, // expects {0, 1}
		{0.8, 0.1}, // expects {1, 0}
	}

	expectedCompressed := []entity.CompressedVector{
		{Data: []byte{0, 1}},
		{Data: []byte{1, 0}},
	}

	compressed, err := acceler.BatchCompressByPQ(vectors, codebook, numSubVectors, numCentroidsPerSubVector, 2)
	if err != nil {
		t.Fatalf("BatchCompressByPQ failed: %v", err)
	}

	if len(compressed) != len(expectedCompressed) {
		t.Fatalf("BatchCompressByPQ returned %d vectors, want %d", len(compressed), len(expectedCompressed))
	}

	for i, cv := range compressed {
		expectedData := expectedCompressed[i].Data
		if len(cv.Data) != len(expectedData) {
			t.Errorf("Vector %d: compressed data length is %d, want %d", i, len(cv.Data), len(expectedData))
			continue
		}
		for j := range expectedData {
			if cv.Data[j] != expectedData[j] {
				t.Errorf("Vector %d: compressed data at index %d is %d, want %d", i, j, cv.Data[j], expectedData[j])
			}
		}
	}
}
