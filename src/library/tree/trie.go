package tree

import "strings"

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

func (t *Trie) InsertWords(sentence string) {
	words := strings.Fields(sentence) // 默认以空格切词
	for _, word := range words {
		t.Insert(word)
	}
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

func (t *Trie) FindWords(sentence string) map[string]bool {
	words := strings.Fields(sentence)
	result := make(map[string]bool)
	for _, word := range words {
		result[word] = t.Find(word)
	}
	return result
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
	return node.isEnding
}

func (t *Trie) StartsWith(prefix string) bool {
	node := t.root
	for _, code := range prefix {
		value, ok := node.children[code]
		if !ok {
			return false
		}
		node = value
	}
	return true
}
