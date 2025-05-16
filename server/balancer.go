package server

import (
	"math/rand"
	"sort"
	"sync/atomic"
)

const (
	Random = iota
	RoundRobin
)

type Balancer interface {
	Take() EndPoint
	Set(endpoints []EndPoint) bool
}

type RandomBalancer struct {
	endpoints []EndPoint
}

func (r *RandomBalancer) Set(endpoints []EndPoint) bool {
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
	addrs  []EndPoint
	totals []int
	max    int
}

func (w *WeightRandomBalance) Set(endpoints []EndPoint) bool {
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
	w.addrs = endpoints
	w.totals = totals
	w.max = runningTotal
	return true
}

func (w *WeightRandomBalance) Take() EndPoint {
	r := rand.Intn(w.max) + 1
	i := sort.SearchInts(w.totals, r)
	return w.addrs[i]
}

type RoundRobinBalancer struct {
	endpoints []EndPoint
	acc       int64
}

func (r *RoundRobinBalancer) Set(endpoints []EndPoint) bool {
	r.endpoints = endpoints
	return true
}

func (r *RoundRobinBalancer) Take() EndPoint {
	if len(r.endpoints) == 0 {
		return EndPoint{}
	}
	n := atomic.AddInt64(&r.acc, 1)
	index := int(n) % len(r.endpoints)
	return r.endpoints[index]
}
