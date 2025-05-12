package server

type EndPoint struct {
	address         string
	weight          int64
	currentWeight   int64
	effectiveWeight int64
}
