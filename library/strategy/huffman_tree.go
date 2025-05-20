package strategy

type MiniHeap struct {
	Size int
	Heap []*HuffmanTree
}

type HuffmanTree struct {
	Left, Right *HuffmanTree
	Weight      int64
	Value       int64
}

func NewMinHeap() *MiniHeap {
	h := &MiniHeap{
		Size: 0,
		Heap: make([]*HuffmanTree, 0),
	}
	h.Heap[0] = &HuffmanTree{}
	return h
}
func (minH *MiniHeap) Less(i, j int) bool {
	return minH.Heap[i].Weight < minH.Heap[j].Weight
}

func (minH MiniHeap) Swap(i, j int) {
	minH.Heap[i], minH.Heap[j] = minH.Heap[j], minH.Heap[i]
}

func (minH *MiniHeap) Insert(item *HuffmanTree) {
	minH.Size++
	i := minH.Size
	minH.Heap = append(minH.Heap, &HuffmanTree{})
	for minH.Heap[i/2].Weight > item.Weight {
		minH.Heap[i] = minH.Heap[i/2]
		i /= 2
	}
	minH.Heap[i] = item
}

func (minH *MiniHeap) IsEmpty() bool {
	return minH.Size == 0
}

func (minH *MiniHeap) Delete() *HuffmanTree {
	if minH.IsEmpty() {
		return nil
	}

	var parent, child int
	minItem := minH.Heap[1]
	for parent = 1; parent*2 <= minH.Size; parent = child {
		child = parent * 2
		if child != minH.Size && minH.Heap[child].Weight > minH.Heap[child+1].Weight {
			child++
		}
		if minH.Heap[minH.Size].Weight <= minH.Heap[child].Weight {
			break
		}
		minH.Heap[parent] = minH.Heap[minH.Size]
	}
	minH.Heap[parent] = minH.Heap[minH.Size]
	minH.Size--

	return minItem
}

func (minH *MiniHeap) GetHuffmanTree() *HuffmanTree {
	for minH.Size > 1 {
		T := &HuffmanTree{}
		T.Left = minH.Delete()
		T.Right = minH.Delete()
		T.Weight = T.Left.Weight + T.Right.Weight
		minH.Insert(T)
	}
	return minH.Delete()
}

func (hum *HuffmanTree) Traversal() {
	if hum == nil {
		return
	}
	hum.Left.Traversal()
	hum.Right.Traversal()
}
