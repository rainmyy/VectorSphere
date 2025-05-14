package common

import (
	"errors"
	"net"
)

func GetLocalHost() (ipv4 string, err error) {
	var (
		addrs   []net.Addr
		addr    net.Addr
		ipNet   *net.IPNet
		isIpNet bool
	)
	if addrs, err = net.InterfaceAddrs(); err != nil {
		return
	}
	for _, addr = range addrs {
		if ipNet, isIpNet = addr.(*net.IPNet); !isIpNet {
			continue
		}
		if ipNet.IP.IsLoopback() {
			continue
		}
		if !ipNet.IP.IsPrivate() {
			continue
		}
		if ipNet.IP.To4() != nil {
			ipv4 = ipNet.IP.String()
			return
		}
	}
	err = errors.New("ERR_NO_LOCAL_IP_FOUND")
	return
}
