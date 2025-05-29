package search

import (
	"github.com/huichen/sego"
)

type Segment struct {
	dict    string
	segment sego.Segmenter
}
type TermInfo struct {
	Term string
	Tf   int
}

func NewSegment(dict string) *Segment {
	var seg sego.Segmenter
	s := &Segment{dict, seg}
	s.segment.LoadDictionary(dict)
	return s
}

func (s *Segment) Segment(content string, searchMode bool) []string {
	text := []byte(content)
	segs := s.segment.Segment(text)
	res := sego.SegmentsToSlice(segs, searchMode)

	return res
}

func (s *Segment) SegmentSingle(content string) []string {
	restore := []rune(content)
	resMap := make(map[rune]bool)
	res := make([]string, 0)
	for _, r := range restore {
		resMap[r] = true
	}
	for k := range resMap {
		res = append(res, string(k))
	}

	return res
}

func (s *Segment) SegmentWithSingle(content string) ([]TermInfo, int) {
	r := []rune(content)
	resMap := make(map[rune]bool)
	for _, r := range r {
		resMap[r] = true
	}
	resTerms := make([]TermInfo, 0)
	for k := range resMap {
		resTerms = append(resTerms, TermInfo{string(k), 0})
	}

	return resTerms, len(resTerms)
}

func (s *Segment) SegmentWithTf(content string, searchModel bool) ([]TermInfo, int) {
	terms := s.Segment(content, searchModel)
	termMap := make(map[string]TermInfo)
	for _, term := range terms {
		if te, ok := termMap[term]; ok {
			tf := te.Tf
			termMap[term] = TermInfo{term, tf}
			continue
		}
		termMap[term] = TermInfo{term, -1}
	}
	resTerms := make([]TermInfo, len(termMap))
	idx := 0
	for _, v := range termMap {
		resTerms[idx] = v
		idx++
	}

	return resTerms, len(resTerms)
}
