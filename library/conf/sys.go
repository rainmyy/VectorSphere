package conf

import (
	"github.com/shirou/gopsutil/v3/process"
	"os"
)

// GetPidCpu 获取系统执行的配置
func GetPidCpu() (float64, error) {
	pid := int32(os.Getpid())
	p, err := process.NewProcess(pid)
	if err != nil {
		return 0, err
	}

	return p.CPUPercent()
}
