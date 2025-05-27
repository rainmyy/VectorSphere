package strategy

// LinkedNode 数据读写最近最少使用算法
type LinkedNode struct {
	key   interface{}
	value *interface{}
	pre   *LinkedNode //上一个节点
	post  *LinkedNode //下一个节点
}
type LRUCache struct {
	count      int
	capacity   int
	cache      map[interface{}]*LinkedNode
	head, tail *LinkedNode
}

func NewLRUCache(capacity int) *LRUCache {
	l := &LRUCache{}
	l.capacity = capacity
	l.cache = make(map[interface{}]*LinkedNode)
	l.head = newLinkNode(nil, nil)
	l.head.pre = nil
	l.tail = newLinkNode(nil, nil)
	l.tail.post = nil
	l.head.post = l.tail
	l.tail.pre = l.head
	return l
}
func newLinkNode(key, value interface{}) *LinkedNode {
	return &LinkedNode{key: key, value: &value}
}
func (l *LRUCache) Len() int {
	return l.count
}
func (l *LRUCache) Get(key interface{}) (interface{}, bool) {
	node, ok := l.cache[key]
	if !ok {
		return nil, false
	}

	l.moveToHead(node)
	return node.value, true
}

func (l *LRUCache) Put(key interface{}, value interface{}) {
	node, ok := l.cache[key]
	if ok {
		node.value = &value
		l.moveToHead(node)
		return
	}
	linkNode := newLinkNode(key, value)
	l.cache[key] = linkNode
	l.addNode(linkNode) // addNode 会将节点加到头部
	l.count++

	if l.count > l.capacity {
		tail := l.popTail()
		if tail != nil { // 确保 tail 不是 nil
			delete(l.cache, tail.key)
			l.count--
		}
	}
}
func (l *LRUCache) Delete(key interface{}) {
	node, ok := l.cache[key]
	if !ok {
		return
	}
	l.removeNode(node)
	delete(l.cache, key)
	l.count--
}
func (l *LRUCache) addNode(node *LinkedNode) {
	node.pre = l.head
	node.post = l.head.post
	l.head.post.pre = node
	l.head.post = node
}

func (l *LRUCache) removeNode(node *LinkedNode) {
	if node == nil || node.pre == nil || node.post == nil {
		return
	}
	pre := node.pre
	post := node.post
	pre.post = post
	post.pre = pre
}

func (l *LRUCache) moveToHead(node *LinkedNode) {
	l.removeNode(node)
	l.addNode(node)
}

func (l *LRUCache) popTail() *LinkedNode {
	res := l.tail.pre
	if res == nil {
		return nil
	}
	l.removeNode(res)
	return res
}
