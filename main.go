package main

import (
	"fmt"
	"seetaSearch/bootstrap"
)

/***
* julySearch 服务端入库
 */
func main() {
	bootstrap.GenInstance().Setup()
	bootstrap.GenInstance().Start()
	defer func() {
		if err := recover(); err != nil {
			fmt.Println(err)
		}
	}()
}
