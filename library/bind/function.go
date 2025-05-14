package bind

import . "seetaSearch/library/common"

func formatBytes(bytes []byte) string {
	str := Bytes2str(bytes)
	if str == "" {
		return str
	}
	return "\"" + str + "\""
}
