package server

import "sync"

type Sentinel struct {
	hub      ServiceHub
	connPool sync.Map
}

var _ IndexInterface = (*Sentinel)(nil)

func NewSentinel(serviceNames []string) *Sentinel {
	return &Sentinel{
		hub:      GetHubProxy(serviceNames, 3, 100),
		connPool: sync.Map{},
	}
}
