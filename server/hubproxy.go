package server

import (
	"context"
	etcdv3 "go.etcd.io/etcd/client/v3"
	"golang.org/x/time/rate"
	"strings"
	"sync"
	"time"
)

type HubProxy struct {
	*EtcdServiceHub
	endpointCache sync.Map
	limiter       *rate.Limiter
}

var (
	hubProxy  *HubProxy
	proxyOnce sync.Once
)

func GetHubProxy(endPoints []string, heartBeat int64, qps int, serviceName string) *HubProxy {
	if hubProxy != nil {
		return hubProxy
	}
	err, hub := GetHub(endPoints, heartBeat, serviceName)
	if err != nil {
		return nil
	}
	proxyOnce.Do(func() {
		hubProxy = &HubProxy{
			EtcdServiceHub: hub,
			endpointCache:  sync.Map{},
			limiter:        rate.NewLimiter(rate.Every(time.Duration(1e9/qps)*time.Nanosecond), qps),
		}
	})
	return hubProxy
}

func (h *HubProxy) GetClient() *etcdv3.Client {
	return h.client
}

func (h *HubProxy) GetEndpoints(serviceName string) []EndPoint {
	if !h.limiter.Allow() {
		return nil
	}
	h.WatchEndpoints(serviceName)
	cacheEndpoints, ok := h.endpointCache.Load(serviceName)
	if ok {
		return cacheEndpoints.([]EndPoint)
	}
	endpoints := h.EtcdServiceHub.GetServiceEndpoints(serviceName)
	if len(endpoints) > 0 {
		h.endpointCache.Store(serviceName, endpoints)
	}

	return endpoints
}

func (h *HubProxy) WatchEndpoints(serviceName string) {
	_, ok := h.watched.LoadOrStore(serviceName, true)
	if ok {
		return
	}
	prefix := ServiceRootPath + "/" + serviceName + "/"
	watchChan := h.client.Watch(context.Background(), prefix, etcdv3.WithPrefix())
	go func() {
		for response := range watchChan {
			for _, event := range response.Events {
				path := strings.Split(string(event.Kv.Key), "/")
				if len(path) <= 2 {
					continue
				}
				serviceName := path[len(path)-2]
				endpoints := h.EtcdServiceHub.GetServiceEndpoints(serviceName)
				if len(path) > 0 {
					h.endpointCache.Store(serviceName, endpoints)
				} else {
					h.endpointCache.Delete(serviceName)
				}
			}
		}
	}()
}
