package monitor

import "github.com/prometheus/client_golang/prometheus"

var (
	// API请求
	apiRequests = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vectorsphere_api_requests_total",
			Help: "Total number of API requests",
		},
		[]string{"path", "method", "status"},
	)
	apiDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "vectorsphere_api_duration_seconds",
			Help:    "API request latency",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"path", "method"},
	)

	// 向量数据库IO
	vectorDBReads = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vectorsphere_vectordb_reads_total",
			Help: "Total number of vector DB read operations",
		},
		[]string{"operation"},
	)
	vectorDBWrites = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vectorsphere_vectordb_writes_total",
			Help: "Total number of vector DB write operations",
		},
		[]string{"operation"},
	)
	vectorDBIOTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "vectorsphere_vectordb_io_duration_seconds",
			Help:    "Vector DB IO operation latency",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"operation"},
	)

	// 数据存储
	vectorDBStorageBytes = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "vectorsphere_vectordb_storage_bytes",
			Help: "Current vector DB storage size in bytes",
		},
	)

	// 进程资源
	cpuUsage = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "vectorsphere_process_cpu_percent",
			Help: "Process CPU usage percent",
		},
	)
	memUsage = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "vectorsphere_process_memory_bytes",
			Help: "Process memory usage in bytes",
		},
	)
)

func init() {
	prometheus.MustRegister(apiRequests, apiDuration)
	prometheus.MustRegister(vectorDBReads, vectorDBWrites, vectorDBIOTime, vectorDBStorageBytes)
	prometheus.MustRegister(cpuUsage, memUsage)
}
