package server

import (
	"math/rand"
	"sync/atomic"
)

type Balancer interface {
	Take(endpoints []EndPoint) EndPoint
}

type RandomBalancer struct{}

func (r *RandomBalancer) Take(endpoints []EndPoint) EndPoint {
	if len(endpoints) == 0 {
		return EndPoint{}
	}
	index := rand.Intn(len(endpoints))
	return endpoints[index]
}

type RoundRobinBalancer struct {
	acc int64
}

func (r *RoundRobinBalancer) Take(endpoints []EndPoint) EndPoint {
	if len(endpoints) == 0 {
		return EndPoint{}
	}
	n := atomic.AddInt64(&r.acc, 1)
	index := int(n) % len(endpoints)
	return endpoints[index]
}
