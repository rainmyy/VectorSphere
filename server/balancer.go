package server

import (
	"hash/crc32"
	"math/rand"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
)

const (
	Random = iota
	RoundRobin
	WeightedRoundRobin
	ConsistentHash
)

func LoadBalanceFactory(lbType int) Balancer {
	switch lbType {
	case Random:
		return &RandomBalancer{}
	case RoundRobin:
		return &RoundRobinBalancer{}
	case WeightedRoundRobin:
		return &WeightRandomBalance{}
	case ConsistentHash:
		return NewConsistentHashBalancer(10, nil)
	default:
		return &RoundRobinBalancer{}
	}
}

type Balancer interface {
	Take() EndPoint
	Set(endpoints ...EndPoint) bool
}

type RandomBalancer struct {
	endpoints []EndPoint
}

func (r *RandomBalancer) Set(endpoints ...EndPoint) bool {
	r.endpoints = endpoints
	return true
}

func (r *RandomBalancer) Take() EndPoint {
	if len(r.endpoints) == 0 {
		return EndPoint{}
	}
	index := rand.Intn(len(r.endpoints))
	return r.endpoints[index]
}

type WeightRandomBalance struct {
	adders []EndPoint
	totals []int
	max    int
}

func (b *WeightedBalancer) Take() EndPoint {
	if len(b.endpoints) == 0 {
		return EndPoint{}
	}
	total := 0
	for _, w := range b.weights {
		total += w
	}
	if total == 0 {
		return b.endpoints[0]
	}
	r := rand.Intn(total)
	for i, w := range b.weights {
		if r < w {
			return b.endpoints[i]
		}
		r -= w
	}
	return b.endpoints[0]
}

type WeightedBalancer struct {
	endpoints []EndPoint
	weights   []int
}

func (b *WeightedBalancer) Set(eps []EndPoint) {
	b.endpoints = eps
	b.weights = make([]int, len(eps))
	for i, ep := range eps {
		if ep.weight > 0 {
			b.weights[i] = int(ep.weight)
		} else {
			b.weights[i] = 1
		}
	}
}

type LeastConnBalancer struct {
	endpoints []EndPoint
	connMap   map[string]int
}

func (b *LeastConnBalancer) Set(eps []EndPoint) bool {
	b.endpoints = eps
	if b.connMap == nil {
		b.connMap = make(map[string]int)
	}
	return true
}
func (b *LeastConnBalancer) Take() EndPoint {
	if len(b.endpoints) == 0 {
		return EndPoint{}
	}
	minIdx := 0
	minConn := b.connMap[b.endpoints[0].Ip]
	for i, ep := range b.endpoints {
		if b.connMap[ep.Ip] < minConn {
			minConn = b.connMap[ep.Ip]
			minIdx = i
		}
	}
	b.connMap[b.endpoints[minIdx].Ip]++
	return b.endpoints[minIdx]
}

func (w *WeightRandomBalance) Set(endpoints ...EndPoint) bool {
	if w == nil {
		return false
	}
	sort.Slice(endpoints, func(i, j int) bool {
		return endpoints[i].weight < endpoints[j].weight
	})
	totals := make([]int, len(endpoints))
	runningTotal := 0
	for i, e := range endpoints {
		runningTotal += int(e.weight)
		totals[i] = runningTotal
	}
	w.adders = endpoints
	w.totals = totals
	w.max = runningTotal
	return true
}

func (w *WeightRandomBalance) Take() EndPoint {
	r := rand.Intn(w.max) + 1
	i := sort.SearchInts(w.totals, r)
	return w.adders[i]
}

type RoundRobinBalancer struct {
	endpoints []EndPoint
	acc       int64
}

func (r *RoundRobinBalancer) Set(endpoints ...EndPoint) bool {
	r.endpoints = endpoints
	return true
}

func (r *RoundRobinBalancer) Take() EndPoint {
	if len(r.endpoints) == 0 {
		return EndPoint{}
	}
	n := atomic.AddInt64(&r.acc, 1)
	current := r.endpoints[r.acc]
	r.acc = int64(int(n) % len(r.endpoints))

	return current
}

type HashFunc func(data []byte) uint32

type Uint32Slice []uint32

func (s Uint32Slice) Len() int {
	return len(s)
}

func (s Uint32Slice) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s Uint32Slice) Less(i, j int) bool {
	return s[i] < s[j]
}

type ConsistentHashBalancer struct {
	mux      sync.RWMutex
	hash     HashFunc
	replicas int
	keys     Uint32Slice
	hashMap  map[uint32]EndPoint
}

func NewConsistentHashBalancer(replicas int, hash HashFunc) *ConsistentHashBalancer {
	c := &ConsistentHashBalancer{
		replicas: replicas,
		hash:     hash,
		hashMap:  make(map[uint32]EndPoint),
	}
	if c.hash == nil {
		c.hash = crc32.ChecksumIEEE
	}

	return c
}

func (c *ConsistentHashBalancer) Set(points ...EndPoint) bool {
	if len(points) == 0 {
		return false
	}
	endpoint := points[0]
	c.mux.Lock()
	defer c.mux.Unlock()
	for i := 0; i < c.replicas; i++ {
		hash := c.hash([]byte(strconv.Itoa(i) + endpoint.Ip))
		c.keys = append(c.keys, hash)
		c.hashMap[hash] = endpoint
	}

	sort.Sort(c.keys)
	return true
}

func (c *ConsistentHashBalancer) IsEmpty() bool {
	return len(c.keys) == 0
}

func (c *ConsistentHashBalancer) Take() EndPoint {
	if c.IsEmpty() {
		return EndPoint{}
	}
	hash := c.hash([]byte(""))
	idx := sort.Search(len(c.keys), func(i int) bool { return c.keys[i] >= hash })
	if idx == len(c.keys) {
		idx = 0
	}
	c.mux.RLock()
	defer c.mux.RUnlock()

	return c.hashMap[c.keys[idx]]
}
