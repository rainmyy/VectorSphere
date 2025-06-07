package strategy

import "math/rand"

func randomLevel() int {
	level := 1
	for rand.Float32() < p && level < maxLevel {
		level++
	}
	return level
}
