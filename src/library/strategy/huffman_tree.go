package strategy

import (
	"bytes"
	"container/heap"
	"errors"
	"fmt"
	"sort"
	"sync"
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

// CanonicalCodeEntry stores a character and its Huffman code length.
type CanonicalCodeEntry struct {
	Value  byte
	Length int
}

// ByLength implements sort.Interface for []CanonicalCodeEntry based on the Length field.
type ByLength []CanonicalCodeEntry

func (a ByLength) Len() int      { return len(a) }
func (a ByLength) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByLength) Less(i, j int) bool {
	if a[i].Length != a[j].Length {
		return a[i].Length < a[j].Length
	}
	return a[i].Value < a[j].Value // Secondary sort by value for stability
}

// buildCodeLengths builds a map of character to its Huffman code length.
func buildCodeLengths(root *huffmanNode) map[byte]int {
	lengths := make(map[byte]int)
	var walk func(n *huffmanNode, length int)
	walk = func(n *huffmanNode, length int) {
		if n == nil {
			return
		}
		if n.Left == nil && n.Right == nil { // Leaf node
			lengths[n.Value] = length
			return
		}
		walk(n.Left, length+1)
		walk(n.Right, length+1)
	}
	walk(root, 0)
	return lengths
}

// generateCanonicalCodes generates canonical Huffman codes from code lengths.
func generateCanonicalCodes(codeLengths map[byte]int) map[byte]string {
	var entries []CanonicalCodeEntry
	for val, length := range codeLengths {
		entries = append(entries, CanonicalCodeEntry{Value: val, Length: length})
	}
	sort.Sort(ByLength(entries))

	codes := make(map[byte]string)
	var currentCode uint64 = 0
	var lastLength int = 0

	for _, entry := range entries {
		if entry.Length > lastLength {
			currentCode <<= (entry.Length - lastLength)
		}
		codes[entry.Value] = fmt.Sprintf("%b", currentCode) // Convert to binary string
		// Pad with leading zeros if necessary
		for len(codes[entry.Value]) < entry.Length {
			codes[entry.Value] = "0" + codes[entry.Value]
		}
		currentCode++
		lastLength = entry.Length
	}
	return codes
}

// HuffmanCompress compresses data using Canonical Huffman coding
func HuffmanCompress(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return nil, nil
	}
	root := buildHuffmanTree(data)
	codeLengths := buildCodeLengths(root)
	canonicalTable := generateCanonicalCodes(codeLengths)

	// Store code lengths for reconstruction
	var codeLenBuf bytes.Buffer
	codeLenBuf.WriteByte(byte(len(codeLengths))) // Number of unique characters
	for val, length := range codeLengths {
		codeLenBuf.WriteByte(val)
		codeLenBuf.WriteByte(byte(length))
	}

	// Encode data using canonical codes
	var bitBuf []byte
	var curByte byte
	var nbits uint8
	for _, b := range data {
		code := canonicalTable[b]
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

	// Output: [codeLenTableLen][codeLenTable][nbits][bitBuf]
	out := bytes.Buffer{}
	codeLenBytes := codeLenBuf.Bytes()
	codeLenTableLen := int32(len(codeLenBytes))
	out.Write([]byte{
		byte(codeLenTableLen >> 24), byte(codeLenTableLen >> 16), byte(codeLenTableLen >> 8), byte(codeLenTableLen),
	})
	out.Write(codeLenBytes)
	out.WriteByte(nbits) // how many bits used in last byte
	out.Write(bitBuf)
	return out.Bytes(), nil
}

// HuffmanDecompress decompresses data using Canonical Huffman coding
func HuffmanDecompress(data []byte) ([]byte, error) {
	if len(data) < 5 { // Minimum 1 byte for num_chars, 2 bytes per char-len pair, 1 byte for nbits, 1 byte for data
		return nil, errors.New("data too short")
	}

	codeLenTableLen := int32(data[0])<<24 | int32(data[1])<<16 | int32(data[2])<<8 | int32(data[3])
	currentIdx := 4

	if int(codeLenTableLen)+currentIdx > len(data) {
		return nil, errors.New("invalid code length table length")
	}

	codeLenBytes := data[currentIdx : currentIdx+int(codeLenTableLen)]
	currentIdx += int(codeLenTableLen)

	numChars := int(codeLenBytes[0])
	codeLengths := make(map[byte]int)
	for i := 0; i < numChars; i++ {
		val := codeLenBytes[1+i*2]
		length := int(codeLenBytes[1+i*2+1])
		codeLengths[val] = length
	}

	canonicalTable := generateCanonicalCodes(codeLengths)

	// Reconstruct Huffman tree from canonical codes for decoding
	// This is a simplified approach for decoding canonical codes.
	// A more efficient way would be to use a lookup table or a trie.
	root := &huffmanNode{}
	for val, code := range canonicalTable {
		cur := root
		for _, bit := range code {
			if bit == '0' {
				if cur.Left == nil {
					cur.Left = &huffmanNode{}
				}
				cur = cur.Left
			} else {
				if cur.Right == nil {
					cur.Right = &huffmanNode{}
				}
				cur = cur.Right
			}
		}
		cur.Value = val
	}

	nbits := data[currentIdx]
	currentIdx++
	bitData := data[currentIdx:]

	var out []byte

	if len(bitData) == 0 && nbits == 0 { // Handle empty bitData for single character input
		return out, nil
	}

	cur := root
	bitsTotal := (len(bitData)-1)*8 + int(nbits)
	// If there's only one character, bitsTotal might be 0 if nbits is 0 and len(bitData) is 1 (for 0-length data)
	// Adjust bitsTotal for single character case where code length is 1
	if len(codeLengths) == 1 && bitsTotal == 0 && len(bitData) == 1 && nbits == 0 {
		// This case happens if the input data was a single character and its code length was 1.
		// The bitData would be 1 byte, nbits 0, and bitsTotal would be calculated as 0.
		// We need to ensure at least one bit is processed for a single character.
		for _, length := range codeLengths {
			if length == 1 {
				bitsTotal = 1
				break
			}
		}
	}

	for i := 0; i < bitsTotal; i++ {
		byteIdx := i / 8
		bitPos := 7 - uint(i%8)
		b := (bitData[byteIdx] >> bitPos) & 1
		if b == 0 {
			cur = cur.Left
		} else {
			cur = cur.Right
		}
		if cur.Left == nil && cur.Right == nil { // Leaf node
			out = append(out, cur.Value)
			cur = root
		}
	}
	return out, nil
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

// AdaptiveHuffman represents an adaptive Huffman encoder/decoder.
type AdaptiveHuffman struct {
	root *huffmanNode // Current Huffman tree root
	freq map[byte]int // Current frequency map
	// Add other necessary fields for FGK/Vitter algorithm, e.g., NYT node, parent pointers, etc.
}

// NewAdaptiveHuffman creates a new adaptive Huffman instance.
func NewAdaptiveHuffman() *AdaptiveHuffman {
	// Initialize with a basic tree (e.g., a single NYT node) and empty frequency map.
	return &AdaptiveHuffman{
		freq: make(map[byte]int),
		// root: initialize with NYT node based on chosen algorithm
	}
}

// EncodeByte encodes a single byte and updates the tree.
func (ah *AdaptiveHuffman) EncodeByte(b byte) ([]byte, error) {
	// Implement encoding logic: find code for 'b', update frequency, update tree.
	// This will involve complex tree restructuring based on FGK/Vitter rules.
	return nil, errors.New("not implemented")
}

// DecodeByte decodes bits from the stream and updates the tree.
func (ah *AdaptiveHuffman) DecodeByte(bitStream *bytes.Buffer) (byte, error) {
	// Implement decoding logic: traverse tree based on bits, update frequency, update tree.
	return 0, errors.New("not implemented")
}

// ParallelBuildFrequencyMap builds a frequency map in parallel using multiple goroutines.
func ParallelBuildFrequencyMap(data []byte, numWorkers int) map[byte]int {
	if len(data) == 0 {
		return make(map[byte]int)
	}

	if numWorkers <= 0 {
		numWorkers = 1 // Ensure at least one worker
	}

	chunkSize := (len(data) + numWorkers - 1) / numWorkers

	freqChan := make(chan map[byte]int, numWorkers)
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := (i + 1) * chunkSize
		if end > len(data) {
			end = len(data)
		}

		if start >= end {
			continue // Skip empty chunks
		}

		wg.Add(1)
		go func(chunk []byte) {
			defer wg.Done()
			localFreq := make(map[byte]int)
			for _, b := range chunk {
				localFreq[b]++
			}
			freqChan <- localFreq
		}(data[start:end])
	}

	wg.Wait()
	close(freqChan)

	globalFreq := make(map[byte]int)
	for localFreq := range freqChan {
		for b, count := range localFreq {
			globalFreq[b] += count
		}
	}

	return globalFreq
}
