package algorithm

func quickSort(array []int, startIndex, endIndex int) {
	//递归结束条件：startIndex >= endIndex 时
	if startIndex > endIndex {
		return
	}
	//得到基准元素的位置
	pivotIndex := partition(array, startIndex, endIndex)
	//用分治发递归基准元素的前面和后面两个部分
	quickSort(array, startIndex, pivotIndex-1)
	quickSort(array, pivotIndex+1, endIndex)
}
func partition(array []int, startIndex, endIndex int) int {
	//取第一个元素作为基准元素
	pivot := array[startIndex]
	left, right := startIndex, endIndex

	//在左右指针重合 或者 交错的时候结束
	for right != left {
		//right指针从右向左进行比较
		for left < right && array[right] > pivot {
			right--
		}
		//left指针从左向右进行比较
		for left < right && array[left] <= pivot {
			left++
		}
		//交换 left 和 right 指向的元素
		if left < right {
			array[left], array[right] = array[right], array[left]
		}
	}
	//pivot和指针重合点交换
	array[left], array[startIndex] = array[startIndex], array[left]
	return left
}
