package scheduler

import (
	"seetaSearch/library/log"
	"time"
)

type MySampleTask struct {
	Name string
}

func (t *MySampleTask) Run() error {
	log.Info("任务 '%s' 正在运行... Timestamp: %s", t.Name, time.Now().String())
	// 你的任务逻辑
	return nil
}

func (t *MySampleTask) GetCronSpec() string {
	return "*/5 * * * * *" // 每5秒执行一次 (需要 cron.WithSeconds())
	// 或者 return "@every 5s"
	// 或者 return "0 */1 * * * *" // 每分钟的第0秒执行，即每分钟一次
}

func (t *MySampleTask) GetName() string {
	return t.Name
}

func (t *MySampleTask) Init() error {
	log.Info("任务 '%s' 正在初始化...", t.Name)
	return nil
}

func (t *MySampleTask) Stop() error {
	log.Info("任务 '%s' 正在停止...", t.Name)
	return nil
}
