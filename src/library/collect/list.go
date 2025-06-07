package collect

import (
	"sync"
)

// Node 双向链表
type Node struct {
	data interface{}
	prev *Node
	next *Node
}

type ListObj struct {
	head   *Node
	tail   *Node
	length uint
	mutex  *sync.RWMutex
}

func ListInstance() *ListObj {
	return &ListObj{mutex: new(sync.RWMutex)}
}

// Append 尾部压入数据
func (o *ListObj) Append(data interface{}) bool {
	if data == nil {
		return false
	}
	o.mutex.Lock()
	defer o.mutex.Unlock()
	var node = new(Node)
	node.data = data
	if o.length == 0 {
		o.head = node
		o.tail = node
		o.length = 1
		return true
	}
	tail := o.tail
	node.prev = tail
	tail.next = node
	o.tail = node
	o.length += 1
	return true
}

func (o *ListObj) Insert(index uint, data interface{}) bool {
	if data == nil {
		return false
	}
	if index > o.length {
		return false
	}
	o.mutex.Lock()
	defer o.mutex.Unlock()
	var node = new(Node)
	if index == 0 {
		node.next = o.head
		o.head.prev = node
		o.head = node
		o.length += 1
		return true
	}
	var i uint
	ptr := o.head
	for i = 1; i < index; i++ {
		ptr = ptr.next
	}
	next := ptr.next
	ptr.next = node
	node.prev = ptr
	next.prev = node
	node.next = next
	o.length += 1
	return true
}

func (o *ListObj) Delete(index uint) bool {
	if o == nil || index > o.length-1 {
		return false
	}

	o.mutex.Lock()
	defer o.mutex.Unlock()
	if index == 0 {
		head := o.head.next
		o.head = head
		o.head.prev = nil
		if o.length == 1 {
			o.tail = nil
		}
		o.length -= 1
		return true
	}
	ptr := o.head
	var i uint
	for i = 1; i < index; i++ {
		ptr = ptr.next
	}
	next := ptr.next
	ptr.next = next.next
	next.next.prev = ptr
	if index == o.length-1 {
		o.tail = ptr
	}
	o.length -= 1
	return true
}

func (o *ListObj) Get(index uint) *Node {
	if o == nil || index > o.length-1 {
		return nil
	}
	o.mutex.RLock()
	defer o.mutex.RUnlock()
	node := o.head
	for i := 0; i < int(index); i++ {
		node = node.next
	}
	return node
}

// Find /
func (o *ListObj) Find(data interface{}) *Node {
	if o == nil {
		return nil
	}
	o.mutex.RLock()
	defer o.mutex.RUnlock()
	head := o.head
	tail := o.tail
	var start uint = 0
	var end uint = o.length - 1
	for start != end {
		if head.data == data {
			return head
		} else if tail.data == data {
			return tail
		}
		head = head.next
		tail = tail.prev
		start++
		end--
	}
	return nil
}
