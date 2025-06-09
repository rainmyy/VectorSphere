package log

import (
	"VectorSphere/src/library/util"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"path/filepath"
	"sync"
	"time"
)

type Level int

const (
	FATAL Level = iota
	ERROR
	WARNING
	INFO
	TRACE
)

var levelStr = [...]string{"FATAL", "ERROR", "WARNING", "INFO", "TRACE"}

type Logger struct {
	mu       sync.Mutex
	level    Level
	logger   *log.Logger
	file     *os.File
	filePath string
	maxSize  int64 // 单位: 字节
	toStdout bool
}

var defaultLogger *Logger

// 默认初始化
func init() {
	basePath, err := util.GetProjectRoot()
	if err != nil {
		return
	}

	_ = InitLogger(INFO, path.Join(basePath, "log", "vector_sphere.log"), 10, true) // 默认INFO级别，输出到终端，最大10MB
}

// InitLogger 初始化日志，filePath为空则输出到终端，否则输出到文件
func InitLogger(level Level, filePath string, maxSizeMB int64, toStdout bool) error {
	var output io.Writer
	var file *os.File
	var err error
	if filePath != "" {
		dir := filepath.Dir(filePath)
		err = os.MkdirAll(dir, 0755)
		if err != nil {
			return err
		}
		file, err = os.OpenFile(filePath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		output = file
		if toStdout {
			output = io.MultiWriter(file, os.Stdout)
		}
	} else {
		output = os.Stdout
	}
	defaultLogger = &Logger{
		level:    level,
		logger:   log.New(output, "", log.Ldate|log.Ltime|log.Lshortfile),
		file:     file,
		filePath: filePath,
		maxSize:  maxSizeMB * 1024 * 1024,
		toStdout: toStdout,
	}
	return nil
}

func (l *Logger) rotateIfNeeded() {
	if l.file == nil || l.filePath == "" || l.maxSize <= 0 {
		return
	}
	info, err := l.file.Stat()
	if err != nil {
		return
	}
	if info.Size() < l.maxSize {
		return
	}
	err = l.file.Close()
	if err != nil {
		return
	}
	backupName := l.filePath + "." + time.Now().Format("20060102_150405")
	os.Rename(l.filePath, backupName)
	file, err := os.OpenFile(l.filePath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err == nil {
		l.file = file
		if l.toStdout {
			l.logger.SetOutput(io.MultiWriter(file, os.Stdout))
		} else {
			l.logger.SetOutput(file)
		}
	}
}

func (l *Logger) logf(level Level, format string, v ...interface{}) {
	if l == nil || level > l.level {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.rotateIfNeeded()
	prefix := "[" + levelStr[level] + "] "
	l.logger.SetPrefix(prefix)
	err := l.logger.Output(3, formatLog(format, v...))
	if err != nil {
		return
	}
	if level == FATAL {
		os.Exit(1)
	}
}

func formatLog(format string, v ...interface{}) string {
	if len(v) == 0 {
		return format
	}
	return fmt.Sprintf(format, v...)
}

// Fatal 对外接口
func Fatal(format string, v ...interface{})   { defaultLogger.logf(FATAL, format, v...) }
func Error(format string, v ...interface{})   { defaultLogger.logf(ERROR, format, v...) }
func Warning(format string, v ...interface{}) { defaultLogger.logf(WARNING, format, v...) }
func Info(format string, v ...interface{})    { defaultLogger.logf(INFO, format, v...) }
func Trace(format string, v ...interface{})   { defaultLogger.logf(TRACE, format, v...) }
