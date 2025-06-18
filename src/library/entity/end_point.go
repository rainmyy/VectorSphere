package entity

type EndPoint struct {
	Ip              string `yaml:"ip"`
	IsMaster        bool   `yaml:"isMaster,omitempty"`
	Port            int    `yaml:"port,omitempty"`
	Tags            map[string]string
	Name            string
	Weight          int64
	CurrentWeight   int64
	EffectiveWeight int64
}
