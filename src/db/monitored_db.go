package db

import (
	"github.com/prometheus/client_golang/prometheus"
	"time"
)

var (
	dbOperationDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "db_operation_duration_seconds",
			Help:    "Duration of database operations in seconds.",
			Buckets: prometheus.ExponentialBuckets(0.0001, 2, 20), // From 100us to ~100s
		},
		[]string{"db_type", "operation", "status"},
	)
	dbOperationTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "db_operations_total",
			Help: "Total number of database operations.",
		},
		[]string{"db_type", "operation", "status"},
	)
)

type MonitoredKvDb struct {
	KvDb
	dbType string
}

func NewMonitoredKvDb(dbType int, path string, bucket string) *MonitoredKvDb {
	kvDb, err := GetDb(dbType, path, bucket)
	if err != nil {
		panic(err)
	}
	var typeStr string
	switch dbType {
	case BOLT:
		typeStr = "BOLT"
	case BADGER:
		typeStr = "BADGER"
	case FILE:
		typeStr = "FILE"
	default:
		typeStr = "BOLT"
	}
	return &MonitoredKvDb{
		KvDb:   kvDb,
		dbType: typeStr,
	}
}

func (m *MonitoredKvDb) Open() error {
	start := time.Now()
	err := m.KvDb.Open()
	status := "success"
	if err != nil {
		status = "failure"
	}
	dbOperationDuration.WithLabelValues(m.dbType, "Open", status).Observe(time.Since(start).Seconds())
	dbOperationTotal.WithLabelValues(m.dbType, "Open", status).Inc()
	return err
}

func (m *MonitoredKvDb) Close() error {
	start := time.Now()
	err := m.KvDb.Close()
	status := "success"
	if err != nil {
		status = "failure"
	}
	dbOperationDuration.WithLabelValues(m.dbType, "Close", status).Observe(time.Since(start).Seconds())
	dbOperationTotal.WithLabelValues(m.dbType, "Close", status).Inc()
	return err
}

func (m *MonitoredKvDb) Set(key, values []byte) error {
	start := time.Now()
	err := m.KvDb.Set(key, values)
	status := "success"
	if err != nil {
		status = "failure"
	}
	dbOperationDuration.WithLabelValues(m.dbType, "Set", status).Observe(time.Since(start).Seconds())
	dbOperationTotal.WithLabelValues(m.dbType, "Set", status).Inc()
	return err
}

func (m *MonitoredKvDb) Get(key []byte) ([]byte, error) {
	start := time.Now()
	val, err := m.KvDb.Get(key)
	status := "success"
	if err != nil {
		status = "failure"
	}
	dbOperationDuration.WithLabelValues(m.dbType, "Get", status).Observe(time.Since(start).Seconds())
	dbOperationTotal.WithLabelValues(m.dbType, "Get", status).Inc()
	return val, err
}

func (m *MonitoredKvDb) BatchGet(keys [][]byte) ([][]byte, error) {
	start := time.Now()
	vals, err := m.KvDb.BatchGet(keys)
	status := "success"
	if err != nil {
		status = "failure"
	}
	dbOperationDuration.WithLabelValues(m.dbType, "BatchGet", status).Observe(time.Since(start).Seconds())
	dbOperationTotal.WithLabelValues(m.dbType, "BatchGet", status).Inc()
	return vals, err
}

func (m *MonitoredKvDb) Del(key []byte) error {
	start := time.Now()
	err := m.KvDb.Del(key)
	status := "success"
	if err != nil {
		status = "failure"
	}
	dbOperationDuration.WithLabelValues(m.dbType, "Del", status).Observe(time.Since(start).Seconds())
	dbOperationTotal.WithLabelValues(m.dbType, "Del", status).Inc()
	return err
}

func (m *MonitoredKvDb) BatchDel(keys [][]byte) error {
	start := time.Now()
	err := m.KvDb.BatchDel(keys)
	status := "success"
	if err != nil {
		status = "failure"
	}
	dbOperationDuration.WithLabelValues(m.dbType, "BatchDel", status).Observe(time.Since(start).Seconds())
	dbOperationTotal.WithLabelValues(m.dbType, "BatchDel", status).Inc()
	return err
}

func (m *MonitoredKvDb) Has(key []byte) bool {
	start := time.Now()
	exists := m.KvDb.Has(key)
	status := "success"
	// Method doesn't return an error, so we assume success for metric purposes
	dbOperationDuration.WithLabelValues(m.dbType, "Has", status).Observe(time.Since(start).Seconds())
	dbOperationTotal.WithLabelValues(m.dbType, "Has", status).Inc()
	return exists
}

func (m *MonitoredKvDb) TotalDb(f func(k, v []byte) error) (int64, error) {
	start := time.Now()
	count, err := m.KvDb.TotalDb(f)
	status := "success"
	if err != nil {
		status = "failure"
	}
	dbOperationDuration.WithLabelValues(m.dbType, "TotalDb", status).Observe(time.Since(start).Seconds())
	dbOperationTotal.WithLabelValues(m.dbType, "TotalDb", status).Inc()
	return count, err
}

func (m *MonitoredKvDb) TotalKey(f func(k []byte) error) (int64, error) {
	start := time.Now()
	count, err := m.KvDb.TotalKey(f)
	status := "success"
	if err != nil {
		status = "failure"
	}
	dbOperationDuration.WithLabelValues(m.dbType, "TotalKey", status).Observe(time.Since(start).Seconds())
	dbOperationTotal.WithLabelValues(m.dbType, "TotalKey", status).Inc()
	return count, err
}
