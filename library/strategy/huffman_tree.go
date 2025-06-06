package strategy

import (
	"bytes"
	"container/heap"
	"errors"
)

type MiniHeap struct {
	Size int
	Heap []*HuffmanTree
}

type HuffmanTree struct {
	Left, Right *HuffmanTree
	Weight      int64
	Value       int64
}

// huffmanNode is used for encoding/decoding
type huffmanNode struct {
	Value  byte
	Weight int
	Left   *huffmanNode
	Right  *huffmanNode
}

type huffmanHeap []*huffmanNode

func (h huffmanHeap) Len() int            { return len(h) }
func (h huffmanHeap) Less(i, j int) bool  { return h[i].Weight < h[j].Weight }
func (h huffmanHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *huffmanHeap) Push(x interface{}) { *h = append(*h, x.(*huffmanNode)) }
func (h *huffmanHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
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

// buildHuffmanTree builds a huffman tree from data
func buildHuffmanTree(data []byte) *huffmanNode {
	freq := make(map[byte]int)
	for _, b := range data {
		freq[b]++
	}
	h := &huffmanHeap{}
	heap.Init(h)
	for b, w := range freq {
		heap.Push(h, &huffmanNode{Value: b, Weight: w})
	}
	for h.Len() > 1 {
		n1 := heap.Pop(h).(*huffmanNode)
		n2 := heap.Pop(h).(*huffmanNode)
		heap.Push(h, &huffmanNode{
			Weight: n1.Weight + n2.Weight,
			Left:   n1,
			Right:  n2,
		})
	}
	if h.Len() == 1 {
		return heap.Pop(h).(*huffmanNode)
	}
	return nil
}

// buildCodeTable builds a map[byte]string for encoding
func buildCodeTable(root *huffmanNode) map[byte]string {
	table := make(map[byte]string)
	var walk func(n *huffmanNode, code string)
	walk = func(n *huffmanNode, code string) {
		if n == nil {
			return
		}
		if n.Left == nil && n.Right == nil {
			table[n.Value] = code
			return
		}
		walk(n.Left, code+"0")
		walk(n.Right, code+"1")
	}
	walk(root, "")
	return table
}

// HuffmanCompress compresses data using Huffman coding
func HuffmanCompress(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return nil, nil
	}
	root := buildHuffmanTree(data)
	table := buildCodeTable(root)
	// Write tree as a pre-order traversal (for decoding)
	var treeBuf bytes.Buffer
	var writeTree func(n *huffmanNode)
	writeTree = func(n *huffmanNode) {
		if n == nil {
			treeBuf.WriteByte(0)
			return
		}
		if n.Left == nil && n.Right == nil {
			treeBuf.WriteByte(1)
			treeBuf.WriteByte(n.Value)
			return
		}
		treeBuf.WriteByte(2)
		writeTree(n.Left)
		writeTree(n.Right)
	}
	writeTree(root)
	// Encode data
	var bitBuf []byte
	var curByte byte
	var nbits uint8
	for _, b := range data {
		code := table[b]
		for _, c := range code {
			curByte <<= 1
			if c == '1' {
				curByte |= 1
			}
			nbits++
			if nbits == 8 {
				bitBuf = append(bitBuf, curByte)
				curByte = 0
				nbits = 0
			}
		}
	}
	if nbits > 0 {
		curByte <<= (8 - nbits)
		bitBuf = append(bitBuf, curByte)
	}
	// Output: [treeLen][tree][nbits][bitBuf]
	out := bytes.Buffer{}
	treeBytes := treeBuf.Bytes()
	treeLen := int32(len(treeBytes))
	out.Write([]byte{
		byte(treeLen >> 24), byte(treeLen >> 16), byte(treeLen >> 8), byte(treeLen),
	})
	out.Write(treeBytes)
	out.WriteByte(nbits) // how many bits used in last byte
	out.Write(bitBuf)
	return out.Bytes(), nil
}

// HuffmanDecompress decompresses data using Huffman coding
func HuffmanDecompress(data []byte) ([]byte, error) {
	if len(data) < 5 {
		return nil, errors.New("data too short")
	}
	treeLen := int32(data[0])<<24 | int32(data[1])<<16 | int32(data[2])<<8 | int32(data[3])
	if int(treeLen)+5 > len(data) {
		return nil, errors.New("invalid tree length")
	}
	treeBytes := data[4 : 4+treeLen]
	nbits := data[4+treeLen]
	bitData := data[5+treeLen:]
	// Rebuild tree
	var idx int
	var readTree func() *huffmanNode
	readTree = func() *huffmanNode {
		if idx >= len(treeBytes) {
			return nil
		}
		t := treeBytes[idx]
		idx++
		if t == 0 {
			return nil
		}
		if t == 1 {
			v := treeBytes[idx]
			idx++
			return &huffmanNode{Value: v}
		}
		left := readTree()
		right := readTree()
		return &huffmanNode{Left: left, Right: right}
	}
	root := readTree()
	// Decode bits
	var out []byte
	n := len(bitData)
	if n == 0 {
		return nil, nil
	}
	cur := root
	bitsTotal := (n-1)*8 + int(nbits)
	for i := 0; i < bitsTotal; i++ {
		byteIdx := i / 8
		bitPos := 7 - uint(i%8)
		b := (bitData[byteIdx] >> bitPos) & 1
		if b == 0 {
			cur = cur.Left
		} else {
			cur = cur.Right
		}
		if cur.Left == nil && cur.Right == nil {
			out = append(out, cur.Value)
			cur = root
		}
	}
	return out, nil
}
