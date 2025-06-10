package entity

// Result 用于堆操作的结果结构
type Result struct {
	Id         string
	Similarity float64 // 使用相似度而非距离，值越大越相似
	WordCount  int
	Distance   float64
}

// ResultHeap 结果的最小堆实现（按相似度排序，保留最大的k个）
type ResultHeap []Result

func (h ResultHeap) Len() int           { return len(h) }
func (h ResultHeap) Less(i, j int) bool { return h[i].Similarity < h[j].Similarity } // 最小堆，相似度小的在顶部
func (h ResultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *ResultHeap) Push(x interface{}) {
	*h = append(*h, x.(Result))
}

func (h *ResultHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
