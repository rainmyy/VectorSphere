package algorithm

func maxHeapify(arr []int, n, i int) {
	largest := i
	left := 2*i + 1
	right := 2*i + 2

	// 如果左子节点大于根节点
	if left < n && arr[left] > arr[largest] {
		largest = left
	}

	// 如果右子节点大于当前最大节点
	if right < n && arr[right] > arr[largest] {
		largest = right
	}

	// 如果最大节点不是根节点
	if largest != i {
		arr[i], arr[largest] = arr[largest], arr[i]

		// 递归调整受影响的子树
		maxHeapify(arr, n, largest)
	}
}

// buildMaxHeap 构建最大堆
func buildMaxHeap(arr []int) {
	n := len(arr)
	for i := n/2 - 1; i >= 0; i-- {
		maxHeapify(arr, n, i)
	}
}

// heapSort 堆排序函数
func heapSort(arr []int) {
	n := len(arr)

	// 构建最大堆
	buildMaxHeap(arr)

	// 一个个从堆中取出元素
	for i := n - 1; i > 0; i-- {
		// 将堆顶元素（最大值）与当前未排序部分的最后一个元素交换
		arr[0], arr[i] = arr[i], arr[0]

		// 重新调整堆，排除已排序的元素
		maxHeapify(arr, i, 0)
	}
}
