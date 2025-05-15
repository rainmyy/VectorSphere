package server

import (
	"math/rand"
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
