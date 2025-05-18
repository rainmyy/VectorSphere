package messages

import "strings"

func NewTermQuery(filed, keyWord string) *TermQuery {
	return &TermQuery{
		Keyword: &KeyWord{Field: filed, Word: keyWord},
	}
}

func (tq *TermQuery) Empty() bool {
	return tq.Keyword == nil && len(tq.Must) == 0 && len(tq.Should) == 0
}

func (tq *TermQuery) And(query ...*TermQuery) *TermQuery {
	array := tq.filterTermQuery(query...)
	if len(array) == 0 {
		return tq
	}
	return &TermQuery{Must: array}
}

func (tq *TermQuery) filterTermQuery(queries ...*TermQuery) []*TermQuery {
	array := make([]*TermQuery, 0, len(queries)+1)
	if len(queries) == 0 {
		return array
	}

	if !tq.Empty() {
		array = append(array, tq)
	}
	for _, v := range queries {
		if tq.Empty() {
			continue
		}
		array = append(array, v)
	}

	return array
}
func (tq *TermQuery) Or(queries ...*TermQuery) *TermQuery {
	array := tq.filterTermQuery(queries...)

	if len(array) == 0 {
		return tq
	}
	return &TermQuery{Should: array}
}

func (tq *TermQuery) ToString() string {
	switch {

	case tq.Keyword != nil:
		return tq.Keyword.ToString()

	case len(tq.Must) > 0:
		if len(tq.Must) == 1 {
			return tq.Must[0].ToString()
		}

		sb := strings.Builder{}
		sb.WriteString("(")
		for _, ele := range tq.Must {
			s := ele.ToString()
			if len(s) > 0 {
				sb.WriteString(s)
				sb.WriteString("&")
			}
		}
		s := sb.String()
		s = s[0:len(s)-1] + ")"

		return s

	case len(tq.Should) > 0:
		if len(tq.Should) == 1 {
			return tq.Should[0].ToString()
		}

		sb := strings.Builder{}
		sb.WriteByte('(')
		for _, ele := range tq.Should {
			s := ele.ToString()
			if len(s) > 0 {
				sb.WriteString(s)
				sb.WriteByte('|')
			}
		}
		s := sb.String()
		s = s[0:len(s)-1] + ")"

		return s
	}

	return ""
}
