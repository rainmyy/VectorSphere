package server

type EndPoint struct {
	Ip              string `yaml:"ip"`
	IsMaster        bool   `yaml:"isMaster,omitempty"`
	Port            int    `yaml:"port,omitempty"`
	Name            string
	weight          int64
	currentWeight   int64
	effectiveWeight int64
}
