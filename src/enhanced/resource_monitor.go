package enhanced

import (
	"VectorSphere/src/library/logger"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/net"
)

// getCPUUsage 获取CPU使用率
func getCPUUsage() (float64, error) {
	// 使用gopsutil库获取CPU使用率
	percentages, err := cpu.Percent(time.Second, false)
	if err != nil {
		return 0, fmt.Errorf("获取CPU使用率失败: %v", err)
	}

	if len(percentages) == 0 {
		return 0, fmt.Errorf("无法获取CPU使用率数据")
	}

	// 返回总体CPU使用率
	return percentages[0], nil
}

// getMemoryUsage 获取内存使用率
func getMemoryUsage() (float64, error) {
	// 使用gopsutil库获取内存信息
	memInfo, err := mem.VirtualMemory()
	if err != nil {
		return 0, fmt.Errorf("获取内存信息失败: %v", err)
	}

	// 返回内存使用百分比
	return memInfo.UsedPercent, nil
}

// getDiskUsage 获取磁盘使用率
func getDiskUsage(path string) (float64, error) {
	// 使用gopsutil库获取磁盘使用情况
	usage, err := disk.Usage(path)
	if err != nil {
		return 0, fmt.Errorf("获取磁盘使用情况失败: %v", err)
	}

	// 返回磁盘使用百分比
	return usage.UsedPercent, nil
}

// getNetworkStats 获取网络状态
func getNetworkStats(interfaceName string) (map[string]float64, error) {
	// 初始化结果
	result := make(map[string]float64)

	// 获取网络接口信息
	interfaces, err := net.Interfaces()
	if err != nil {
		return nil, fmt.Errorf("获取网络接口信息失败: %v", err)
	}

	// 如果没有指定接口名称，使用第一个非回环接口
	if interfaceName == "" {
		for _, iface := range interfaces {
			// 检查是否为回环接口 (Flags & net.FlagLoopback) != 0
			if (iface.Flags & net.FlagLoopback) == 0 {
				interfaceName = iface.Name
				break
			}
		}
	}

	// 如果仍然没有找到合适的接口，返回错误
	if interfaceName == "" {
		return nil, fmt.Errorf("未找到合适的网络接口")
	}

	// 获取网络IO计数器
	countersBefore, err := net.IOCounters(true)
	if err != nil {
		return nil, fmt.Errorf("获取网络IO计数器失败: %v", err)
	}

	// 等待一段时间以计算速率
	time.Sleep(500 * time.Millisecond)

	// 再次获取网络IO计数器
	countersAfter, err := net.IOCounters(true)
	if err != nil {
		return nil, fmt.Errorf("获取网络IO计数器失败: %v", err)
	}

	// 查找指定接口的计数器
	var before, after net.IOCountersStat
	found := false

	for i := range countersAfter {
		if countersAfter[i].Name == interfaceName {
			after = countersAfter[i]
			before = countersBefore[i]
			found = true
			break
		}
	}

	if !found {
		return nil, fmt.Errorf("未找到指定的网络接口: %s", interfaceName)
	}

	// 计算网络速率（0.5秒内的变化）
	result["network_interface"] = float64(0) // 仅作为标记
	result["bytes_sent"] = float64(after.BytesSent)
	result["bytes_recv"] = float64(after.BytesRecv)
	result["packets_sent"] = float64(after.PacketsSent)
	result["packets_recv"] = float64(after.PacketsRecv)
	result["errin"] = float64(after.Errin)
	result["errout"] = float64(after.Errout)
	result["dropin"] = float64(after.Dropin)
	result["dropout"] = float64(after.Dropout)

	// 计算每秒发送和接收的字节数
	result["bytes_sent_per_sec"] = float64(after.BytesSent-before.BytesSent) * 2 // 乘以2转换为每秒
	result["bytes_recv_per_sec"] = float64(after.BytesRecv-before.BytesRecv) * 2

	// 计算丢包率
	totalPacketsBefore := before.PacketsSent + before.PacketsRecv
	totalPacketsAfter := after.PacketsSent + after.PacketsRecv
	totalDropsBefore := before.Dropin + before.Dropout
	totalDropsAfter := after.Dropin + after.Dropout

	packetLossRate := 0.0
	if totalPacketsAfter-totalPacketsBefore > 0 {
		packetLossRate = float64(totalDropsAfter-totalDropsBefore) / float64(totalPacketsAfter-totalPacketsBefore)
	}
	result["packet_loss_rate"] = packetLossRate

	// 模拟网络延迟（实际应该使用ping或其他方法测量）
	result["network_latency"] = simulateNetworkLatency(interfaceName)

	return result, nil
}

// simulateNetworkLatency 模拟网络延迟
func simulateNetworkLatency(interfaceName string) float64 {
	// 这里使用简单的ping来测量延迟
	// 在实际生产环境中，应该使用更可靠的方法

	// 默认ping目标
	target := "8.8.8.8"

	// 创建一个临时文件来存储ping结果
	tmpFile, err := os.CreateTemp("", "ping_result")
	if err != nil {
		logger.Warning("创建临时文件失败: %v", err)
		return 50.0 // 返回默认值
	}
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	// 执行ping命令
	var cmd string
	if runtime.GOOS == "windows" {
		cmd = fmt.Sprintf("ping -n 3 %s > %s", target, tmpFile.Name())
	} else {
		cmd = fmt.Sprintf("ping -c 3 %s > %s", target, tmpFile.Name())
	}

	// 执行命令
	err = exec.Command("sh", "-c", cmd).Run()
	if err != nil {
		logger.Warning("执行ping命令失败: %v", err)
		return 100.0 // 返回默认值
	}

	// 读取结果
	output, err := os.ReadFile(tmpFile.Name())
	if err != nil {
		logger.Warning("读取ping结果失败: %v", err)
		return 75.0 // 返回默认值
	}

	// 解析结果获取平均延迟
	outputStr := string(output)
	var latency float64 = 50.0 // 默认值

	// 根据不同操作系统解析输出
	if runtime.GOOS == "windows" {
		// Windows格式: Average = 21ms
		if avgIndex := strings.Index(outputStr, "Average = "); avgIndex != -1 {
			avgStr := outputStr[avgIndex+len("Average = "):]
			avgStr = strings.TrimSpace(strings.Split(avgStr, "ms")[0])
			if avg, err := strconv.ParseFloat(avgStr, 64); err == nil {
				latency = avg
			}
		}
	} else {
		// Linux/Unix格式: rtt min/avg/max/mdev = 20.104/21.302/22.041/0.889 ms
		if rttIndex := strings.Index(outputStr, "rtt min/avg/max/mdev = "); rttIndex != -1 {
			rttStr := outputStr[rttIndex+len("rtt min/avg/max/mdev = "):]
			rttParts := strings.Split(rttStr, "/")
			if len(rttParts) >= 2 {
				if avg, err := strconv.ParseFloat(rttParts[1], 64); err == nil {
					latency = avg
				}
			}
		}
	}

	return latency
}
