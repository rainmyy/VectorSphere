package messages

import "VectorSphere/src/library/tree"

func (kw *KeyWord) ToString() string {
	if len(kw.Word) > 0 {
		return kw.Field + "\001" + kw.Word
	}

	return ""
}

// Less 实现 Key 接口
func (kw KeyWord) Less(other tree.Key) bool {
	otherKW, ok := other.(*KeyWord)
	if !ok {
		panic("other is not a KeyWord")
	}
	return kw.ToString() < otherKW.ToString()
}

func (kw KeyWord) Equal(other tree.Key) bool {
	otherKW, ok := other.(*KeyWord)
	if !ok {
		return false
	}
	return kw.ToString() == otherKW.ToString()
}
