package entity

// CentroidDist 用于堆操作的簇中心距离结构
type CentroidDist struct {
	Index    int
	Distance float64
}

// CentroidHeap 簇中心距离的最小堆实现
type CentroidHeap []CentroidDist

func (h CentroidHeap) Len() int           { return len(h) }
func (h CentroidHeap) Less(i, j int) bool { return h[i].Distance < h[j].Distance }
func (h CentroidHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *CentroidHeap) Push(x interface{}) {
	*h = append(*h, x.(CentroidDist))
}

func (h *CentroidHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
