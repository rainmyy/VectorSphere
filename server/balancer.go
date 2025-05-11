package server

import (
	"math/rand"
	"sync/atomic"
)

type Balancer interface {
	Take(endpoints []string) string
}

type RandomBalancer struct{}

func (r *RandomBalancer) Take(endpoints []string) string {
	if len(endpoints) == 0 {
		return ""
	}
	index := rand.Intn(len(endpoints))
	return endpoints[index]
}

type RoundRobinBalancer struct {
	acc int64
}

func (r *RoundRobinBalancer) Take(endpoints []string) string {
	if len(endpoints) == 0 {
		return ""
	}
	n := atomic.AddInt64(&r.acc, 1)
	index := int(n) % len(endpoints)
	return endpoints[index]
}
