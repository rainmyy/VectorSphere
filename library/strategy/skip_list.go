package strategy

const (
	maxLevel int     = 32
	p        float32 = 0.25
)

type SkipList struct {
	header *Element
	len    int
	level  int
}

type Element struct {
	Score   float64
	Value   interface{}
	key     interface{}
	forward []*Element
}

func newElement(score float64, value interface{}, level int) *Element {
	return &Element{
		Score:   score,
		Value:   value,
		forward: make([]*Element, level),
	}
}

func (e *Element) Next() *Element {
	if e != nil {
		return e.forward[0]
	}
	return nil
}
func (e *Element) Key() interface{} {
	return e.key
}
func NewSkipList() *SkipList {
	return &SkipList{
		header: &Element{forward: make([]*Element, maxLevel)},
	}
}

func (sl *SkipList) Front() *Element {
	return sl.header.forward[0]
}

func (sl *SkipList) Search(score float64) (element *Element, ok bool) {
	x := sl.header
	for i := sl.level - 1; i >= 0; i-- {
		for x.forward[i] != nil && x.forward[i].Score < score {
			x = x.forward[i]
		}
	}

	x = x.forward[0]
	if x != nil && x.Score == score {
		return x, true
	}
	return nil, false
}

func (sl *SkipList) Insert(score float64, value interface{}) *Element {
	update := make([]*Element, maxLevel)
	x := sl.header
	for i := sl.level - 1; i >= 0; i-- {
		for x.forward[i] != nil && x.forward[i].Score < score {
			x = x.forward[i]
		}
		update[i] = x
	}
	x = x.forward[0]
	if x != nil && x.Score == score {
		x.Value = value
		return x
	}

	level := randomLevel()
	if level > sl.level {
		level = sl.level + 1
		update[sl.level] = sl.header
		sl.level = level
	}
	e := newElement(score, value, level)

	for j := 0; j < level; j++ {
		e.forward[j] = update[j].forward[j]
		update[j].forward[j] = e
	}
	sl.len++
	return e
}

func (sl *SkipList) Delete(score float64) *Element {
	update := make([]*Element, maxLevel)
	x := sl.header
	for i := sl.level - 1; i >= 0; i-- {
		for x.forward[i] != nil && x.forward[i].Score < score {
			x = x.forward[i]
		}
		update[i] = x
	}
	x = x.forward[0]
	if x != nil && x.Score == score {
		for j := 0; j < sl.level; j++ {
			if update[j].forward[j] != x {
				return nil
			}
			update[j].forward[j] = x.forward[j]
		}
		sl.len--
	}
	return x
}

func (sl *SkipList) Len() int {
	return sl.len
}
