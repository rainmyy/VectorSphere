package db

import (
	"math"
	"strings"
)

// DocumentVectorized 定义文档向量化函数的类型
type DocumentVectorized func(doc string) ([]float64, error)

// SimpleBagOfWordsVectorized 简单的词袋模型向量化函数
func SimpleBagOfWordsVectorized(vocab []string) DocumentVectorized {
	return func(doc string) ([]float64, error) {
		// 初始化向量
		vector := make([]float64, len(vocab))

		// 简单的分词，按空格分割
		words := strings.Fields(doc)

		// 统计词频
		for _, word := range words {
			for i, v := range vocab {
				if v == word {
					vector[i]++
				}
			}
		}

		return vector, nil
	}
}

// TFIDFVectorized TF-IDF 向量化函数
func TFIDFVectorized(vocab []string, docCount int, docFreq map[string]int) DocumentVectorized {
	return func(doc string) ([]float64, error) {
		// 初始化向量
		vector := make([]float64, len(vocab))

		// 简单的分词，按空格分割
		words := strings.Fields(doc)

		// 统计词频
		wordCount := make(map[string]int)
		for _, word := range words {
			wordCount[word]++
		}

		// 计算 TF-IDF
		for i, v := range vocab {
			tf := float64(wordCount[v]) / float64(len(words))
			idf := math.Log(float64(docCount) / float64(docFreq[v]+1))
			vector[i] = tf * idf
		}

		return vector, nil
	}
}
