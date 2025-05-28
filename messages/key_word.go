package messages

import bplus "seetaSearch/library/BPlus"

func (kw *KeyWord) ToString() string {
	if len(kw.Word) > 0 {
		return kw.Field + "\001" + kw.Word
	}

	return ""
}

// Less 实现 Key 接口
func (kw *KeyWord) Less(other bplus.Key) bool {
	otherKW, ok := other.(*KeyWord)
	if !ok {
		panic("other is not a KeyWord")
	}
	// 假设使用 ToString() 方法的结果进行比较
	return kw.ToString() < otherKW.ToString()
}

func (kw *KeyWord) Equal(other bplus.Key) bool {
	otherKW, ok := other.(*KeyWord)
	if !ok {
		return false
	}
	// 假设使用 ToString() 方法的结果进行比较
	return kw.ToString() == otherKW.ToString()
}
