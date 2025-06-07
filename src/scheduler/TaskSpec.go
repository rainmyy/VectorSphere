package scheduler

import "time"

type TaskSpec struct {
	Name        string
	RetryCount  int
	Timeout     time.Duration
	Description string
}
