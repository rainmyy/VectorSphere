package collect

import (
	"testing"
)

func TestAppend(t *testing.T) {
	listobj := ListInstance()
	var (
		testData = []interface{}{
			1, "a", []int{1, 3},
		}
	)
	for i := 0; i < len(testData); i++ {
		actual := listobj.Append(testData[i])
		if actual != true {
			t.Errorf("append is wrong")
		}
	}
}
