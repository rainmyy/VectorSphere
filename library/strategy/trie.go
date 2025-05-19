package strategy

type TriedNode struct {
	char     string
	isEnding bool
	children map[rune]*TriedNode
}

func NewTrieNode(char string) *TriedNode {
	return &TriedNode{
		char:     char,
		isEnding: false,
		children: make(map[rune]*TriedNode),
	}
}

type Trie struct {
	root *TriedNode
}

func NewTrie() *Trie {
	triedNode := NewTrieNode("/")
	return &Trie{triedNode}
}

func (t *Trie) Insert(word string) {
	node := t.root
	for _, code := range word {
		value, ok := node.children[code]
		if !ok {
			value = NewTrieNode(string(code))
			node.children[code] = value
		}
		node = value
	}
	node.isEnding = true
}

func (t *Trie) Find(word string) bool {
	node := t.root
	for _, code := range word {
		value, ok := node.children[code]
		if !ok {
			return false
		}
		node = value
	}
	if node.isEnding == false {
		return false
	}
	return true
}
