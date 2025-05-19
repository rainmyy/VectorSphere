package strategy

/**
* 数据读写最近最少使用算法
 */
type linkedNode struct {
	key   interface{}
	value interface{}
	pre   *linkedNode //上一个节点
	post  *linkedNode //下一个节点
}
type LRUCache struct {
	count      int
	capacity   int
	cache      map[interface{}]*linkedNode
	head, tail *linkedNode
}

func NewLRUCache(capacity int) *LRUCache {
	l := &LRUCache{}
	l.capacity = capacity
	l.cache = make(map[interface{}]*linkedNode)
	l.head = newLinkNode()
	l.head.pre = nil
	l.tail = newLinkNode()
	l.tail.post = nil
	l.head.post = l.tail
	l.tail.pre = l.head
	return l
}
func newLinkNode() *linkedNode {
	return &linkedNode{}
}
func (l *LRUCache) Get(key interface{}) interface{} {
	node, ok := l.cache[key]
	if !ok {
		return -1
	}

	l.moveToHead(node)
	return node.value
}

func (l *LRUCache) Put(key interface{}, value interface{}) {
	node, ok := l.cache[key]
	if ok {
		node.value = value
		l.moveToHead(node)
		return
	}
	linkNode := newLinkNode()
	linkNode.key = key
	linkNode.value = value
	l.cache[key] = linkNode
	l.addNode(linkNode)
	l.count++
	if l.count > l.capacity {
		tail := l.popTail()
		if tail == nil {
			return
		}
		if _, ok := l.cache[tail.key]; ok {
			delete(l.cache, tail.key)
		}
		l.count--
	}
}

func (l *LRUCache) addNode(node *linkedNode) {
	node.pre = l.head
	node.post = l.head.post
	l.head.post.pre = node
	l.head.post = node
}

func (l *LRUCache) removeNode(node *linkedNode) {
	node.pre = newLinkNode()
	pre := node.pre
	post := node.post
	pre.post = post
	post.pre = pre
}

func (l *LRUCache) moveToHead(node *linkedNode) {
	l.removeNode(node)
	l.addNode(node)
}

func (l *LRUCache) popTail() *linkedNode {
	res := l.tail.pre

	if res == nil {
		return nil
	}
	l.removeNode(res)
	return res
}
