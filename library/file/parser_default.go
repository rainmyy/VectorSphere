package file

import (
	. "seetaSearch/library/strategy"
	"strings"
)

func ParserContent(data []byte) ([]*TreeStruct, error) {
	if len(data) == 0 {
		return nil, nil
	}
	lines := strings.Split(string(data), "\n")
	var trees []*TreeStruct
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// 假设格式为 key:value
		parts := strings.SplitN(line, ":", 2)
		if len(parts) == 2 {
			tree := TreeInstance()
			node := NodeInstance([]byte(strings.TrimSpace(parts[0])), []byte(strings.TrimSpace(parts[1])))
			tree.SetNode(node)
			trees = append(trees, tree)
		} else {
			// 只有key没有value
			tree := TreeInstance()
			node := NodeInstance([]byte(line), []byte{})
			tree.SetNode(node)
			trees = append(trees, tree)
		}
	}
	return trees, nil
}
