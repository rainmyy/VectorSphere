package tree

import (
	"time"
	"unsafe"
)

type TreeStruct struct {
	node         []*NodeStruct
	children     []*TreeStruct
	parent       *TreeStruct
	high         int //计算层高
	leaf         bool
	childLeafNum int
}

type NodeStruct struct {
	data   []byte
	name   []byte
	length int
	/**
	* 数据的创建时间，隐藏数据不展示，通过该key值进行树的检索和排序
	* 第一次创建数据时初始化该值，直到NodeStruct被回收前该值保持不变
	 */
	createtime time.Time
	/**
	* 数据的更新时间，隐藏数据不展示，通过该key值进行树的检索和排序
	* 第一次创建数据时初始化该值，元素修改时修改该值
	 */
	updatetime time.Time

	/**
	* 每次修改数据备份已有数据
	 */
	backup []byte
}

func TreeStruct2Bytes(treeList []*TreeStruct) [][]byte {
	var byteList = make([][]byte, 0)
	for _, tree := range treeList {
		//var sli reflect.SliceHeader
		//sli.Len = int(unsafe.Sizeof(tree))
		//sli.Cap = int(unsafe.Sizeof(tree))
		//sli.Data = uintptr(unsafe.Pointer(&tree))
		//bytes := *(*[]byte)(unsafe.Pointer(&sli))

		byt := unsafe.Slice(&tree, unsafe.Sizeof(tree))
		byteList = append(byteList, *(*[]byte)(unsafe.Pointer(&byt)))
	}

	return byteList
}

func Bytes2TreeStruct(b [][]byte) []TreeStruct {
	var resList []TreeStruct
	for _, val := range b {
		treeStruct := *(*TreeStruct)(unsafe.Pointer(&val[0]))
		//for _, val := range treeStruct.GetNode() {
		//	fmt.Print(val)
		//}
		//print(len(treeStruct.GetNode()))
		resList = append(resList, treeStruct)
	}
	return resList
}

func (s *TreeStruct) GetNode() []*NodeStruct {
	return s.node
}

func (s *TreeStruct) SetNode(node *NodeStruct) {
	if node == nil {
		return
	}
	s.node = append(s.node, node)
	if len(node.data) > 0 {
		s.leaf = true
	}
}

func (s *TreeStruct) GetChildren() []*TreeStruct {
	return s.children
}

func (s *TreeStruct) SetChildren(children *TreeStruct) *TreeStruct {
	if children == nil {
		return s
	}
	children.SetParent(s)
	children.SetHeight(s.high + 1)
	s.children = append(s.children, children)
	if children.IsLeaf() {
		s.childLeafNum++
	}
	return s
}

// GetParent /**
func (s *TreeStruct) GetParent() *TreeStruct {
	return s.parent
}

func (s *TreeStruct) SetParent(tree *TreeStruct) {
	s.parent = tree
}

func (s *TreeStruct) GetRoot() *TreeStruct {
	if s.IsRoot() == true {
		return s
	}
	cur := s
	for cur.parent != nil {
		cur = cur.parent
	}
	return cur
}

func (s *TreeStruct) GetHeight() int {
	return s.high
}

func (s *TreeStruct) SetHeight(height int) {
	s.high = height
}

func (s *TreeStruct) IsLeaf() bool {
	return s.leaf
}
func (s *TreeStruct) IsRoot() bool {
	if s.parent != nil {
		return false
	}
	return true
}

func TreeInstance() *TreeStruct {
	return &TreeStruct{
		node:     make([]*NodeStruct, 0),
		children: make([]*TreeStruct, 0),
		parent:   nil,
	}
}

func (s *NodeStruct) UpdateData(value []byte) {
	if len(s.data) == len(value) {
		same := true
		for i := 0; i < len(s.data); i++ {
			if s.data[i] != value[i] {
				same = false
				break
			}
		}
		if same {
			return
		}
	}
	s.backup = s.data
	s.data = value
	s.updatetime = time.Now()
}

func NodeInstance(key []byte, value []byte) *NodeStruct {
	return &NodeStruct{
		name:       key,
		data:       value,
		length:     len(value),
		createtime: time.Now(),
		updatetime: time.Now(),
	}
}

func (s *NodeStruct) GetData() []byte {
	return s.data
}
func (s *NodeStruct) GetName() []byte {
	return s.name
}
