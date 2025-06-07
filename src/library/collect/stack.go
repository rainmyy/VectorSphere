package collect

const treVal = 0.75

type Stack struct {
	list *ListObj
}

func StackInstance() *Stack {
	return &Stack{new(ListObj)}
}
func (s *Stack) Len() int {
	return int(s.list.length)
}
func (s *Stack) Peek() interface{} {
	if s.Len() == 0 {
		return nil
	}
	return s.list.head
}

func (s *Stack) Pop() interface{} {
	if s.Len() == 0 {
		return nil
	}
	value := s.list.head
	result := s.list.Delete(0)
	if !result {
		return nil
	}
	return value
}

func (s *Stack) Push(value interface{}) bool {
	if value == nil {
		return false
	}
	result := s.list.Insert(0, value)
	return result
}
