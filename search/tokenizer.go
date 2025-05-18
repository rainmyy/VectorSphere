package search

import (
	"embed"
	"strings"
)
import "github.com/wangbin/jiebago"

var dicFs embed.FS

type Tokenizer struct {
	seg jiebago.Segmenter
}

func NewTokenizer(dicPath string) *Tokenizer {
	//file, err := dicFs.Open(dicPath)
	//if err != nil {
	//	panic(err)
	//}
	tokenizer := new(Tokenizer)
	err := tokenizer.seg.LoadDictionary(dicPath)
	if err != nil {
		panic(err)
	}

	return tokenizer
}

func (t *Tokenizer) Cut(text string) []string {
	text = strings.ToLower(text)
	text = strings.TrimSpace(text)
	var wordMap = make(map[string]struct{})
	resultChan := t.seg.CutForSearch(text, true)
	var wordList []string
	for v := range resultChan {
		if _, ok := wordMap[v]; !ok {
			wordMap[v] = struct{}{}
			wordList = append(wordList, v)
		}
	}

	return wordList
}
