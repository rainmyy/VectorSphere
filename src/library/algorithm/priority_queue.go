package algorithm

import "container/heap"

// Item 表示优先队列中的元素
type Item struct {
	value    interface{} // 元素的值
	priority int         // 元素的优先级
	index    int         // 元素在堆中的索引
}

// PriorityQueue 实现了 heap.Interface 接口
type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	// 这里使用大于号，实现最大优先队列
	// 如果要实现最小优先队列，使用小于号
	return pq[i].priority > pq[j].priority
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

// Push 向优先队列中添加元素
func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*Item)
	item.index = n
	*pq = append(*pq, item)
}

// Pop 从优先队列中移除并返回优先级最高的元素
func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil  // 避免内存泄漏
	item.index = -1 // 表示元素已不在队列中
	*pq = old[0 : n-1]
	return item
}

// update 修改优先队列中元素的优先级和值
func (pq *PriorityQueue) update(item *Item, value interface{}, priority int) {
	item.value = value
	item.priority = priority
	heap.Fix(pq, item.index)
}
