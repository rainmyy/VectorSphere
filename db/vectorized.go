package db

import (
	"bufio"
	"fmt"
	"github.com/go-ego/gse"
	"math"
	"os"
	"strconv"
	"strings"
)

// DocumentVectorized 定义文档向量化函数的类型
type DocumentVectorized func(doc string) ([]float64, error)

// SimpleBagOfWordsVectorized 简单的词袋模型向量化函数
func SimpleBagOfWordsVectorized() DocumentVectorized {
	var globalVocab []string
	vocabSet := make(map[string]struct{})
	var seg gse.Segmenter
	err := seg.LoadDict()
	if err != nil {
		return nil
	}
	return func(doc string) ([]float64, error) {
		// 使用 gse 分词库进行分词
		words := seg.Cut(doc, true)

		// 更新全局词表
		for _, word := range words {
			if _, exists := vocabSet[word]; !exists {
				vocabSet[word] = struct{}{}
				globalVocab = append(globalVocab, word)
			}
		}

		// 初始化向量
		vector := make([]float64, len(globalVocab))

		// 统计词频
		wordCount := make(map[string]int)
		for _, word := range words {
			wordCount[word]++
		}

		// 填充向量
		for i, word := range globalVocab {
			vector[i] = float64(wordCount[word])
		}

		return vector, nil
	}
}

// TFIDFVectorized 返回一个实现TF-IDF向量化的函数
func TFIDFVectorized() DocumentVectorized {
	// 存储所有文档的词频统计
	var docFreq map[string]int
	var totalDocs int
	var seg gse.Segmenter
	err := seg.LoadDict()
	if err != nil {
		return nil
	}
	return func(doc string) ([]float64, error) {
		// 使用 gse 分词库进行分词
		words := seg.Cut(doc, true)
		// 统计当前文档的词频
		termFreq := make(map[string]int)
		for _, word := range words {
			termFreq[word]++
		}

		// 获取所有唯一词
		var vocab []string
		for word := range termFreq {
			vocab = append(vocab, word)
		}
		for word := range docFreq {
			found := false
			for _, v := range vocab {
				if v == word {
					found = true
					break
				}
			}
			if !found {
				vocab = append(vocab, word)
			}
		}

		// 计算 TF-IDF 向量
		vector := make([]float64, len(vocab))
		for i, word := range vocab {
			tf := float64(termFreq[word]) / float64(len(words))
			var idf float64
			if docFreq != nil && docFreq[word] > 0 {
				idf = math.Log(float64(totalDocs) / float64(docFreq[word]))
			} else {
				idf = math.Log(float64(totalDocs) / 1.0)
			}
			vector[i] = tf * idf
		}

		// 更新全局文档频率统计
		if docFreq == nil {
			docFreq = make(map[string]int)
		}
		for word := range termFreq {
			docFreq[word]++
		}
		totalDocs++

		return vector, nil
	}
}

// LoadWordEmbeddings 加载预训练的词向量文件
func LoadWordEmbeddings(filePath string) (map[string][]float64, error) {
	// 打开文件
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("无法打开文件: %w", err)
	}
	defer file.Close()
	var seg gse.Segmenter
	err = seg.LoadDict()
	if err != nil {
		return nil, err
	}
	// 读取文件内容
	scanner := bufio.NewScanner(file)
	embeddings := make(map[string][]float64)
	var expectedDim int
	var hasValidDim bool
	// 跳过第一行（如果第一行是元数据，如词的数量和向量维度）
	if scanner.Scan() {
		line := scanner.Text()
		parts := seg.Cut(line, true)
		if len(parts) == 2 {
			// 尝试解析元数据
			_, errNum := strconv.Atoi(parts[0])
			dim, errDim := strconv.Atoi(parts[1])
			if errNum == nil && errDim == nil {
				// 元数据解析成功，可以在后续使用 dim 来验证向量维度
				expectedDim = dim
				hasValidDim = true
			}
		}
	}

	for scanner.Scan() {
		line := scanner.Text()
		// 分割行内容为词和向量部分，使用标准的字符串分割而非分词库
		parts := seg.Cut(line, true)
		//parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		word := parts[0]
		vector := make([]float64, len(parts)-1)
		for i := 1; i < len(parts); i++ {
			val, err := strconv.ParseFloat(parts[i], 64)
			if err != nil {
				return nil, fmt.Errorf("解析向量值失败: %w", err)
			}
			vector[i-1] = val
		}

		// 验证向量维度
		if hasValidDim && len(vector) != expectedDim {
			return nil, fmt.Errorf("向量 '%s' 的维度 %d 与预期维度 %d 不匹配", word, len(vector), expectedDim)
		}

		embeddings[word] = vector
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("读取文件时出错: %w", err)
	}

	return embeddings, nil
}

// NewWordEmbeddingVectorized 创建一个基于词嵌入的文档向量化函数
func NewWordEmbeddingVectorized(embeddings map[string][]float64) DocumentVectorized {
	return func(doc string) ([]float64, error) {
		// 分词
		words := strings.Fields(doc)
		if len(words) == 0 {
			return nil, fmt.Errorf("文档为空")
		}

		var docVector []float64
		validCount := 0

		for _, word := range words {
			if vec, exists := embeddings[word]; exists {
				if docVector == nil {
					docVector = make([]float64, len(vec))
				}
				for i := range vec {
					docVector[i] += vec[i]
				}
				validCount++
			}
		}

		if validCount == 0 {
			return nil, fmt.Errorf("文档中没有有效的词向量")
		}

		// 求平均
		for i := range docVector {
			docVector[i] /= float64(validCount)
		}

		return docVector, nil
	}
}

// EnhancedWordEmbeddingVectorized 优化的词嵌入向量化函数
func EnhancedWordEmbeddingVectorized(embeddings map[string][]float64) DocumentVectorized {
	return func(doc string) ([]float64, error) {
		// 使用更高级的分词方法
		var seg gse.Segmenter
		err := seg.LoadDict()
		if err != nil {
			return nil, err
		}
		words := seg.Cut(doc, true)

		if len(words) == 0 {
			return nil, fmt.Errorf("文档为空")
		}

		// 使用TF-IDF加权词向量
		wordFreq := make(map[string]int)
		for _, word := range words {
			wordFreq[word]++
		}

		// 计算词频
		for word := range wordFreq {
			wordFreq[word] = wordFreq[word] / len(words)
		}

		// 初始化文档向量
		var docVector []float64
		validCount := 0

		// 加权求和
		for word, freq := range wordFreq {
			if vec, exists := embeddings[word]; exists {
				if docVector == nil {
					docVector = make([]float64, len(vec))
				}

				// 使用词频作为权重
				weight := float64(freq)
				for i := range vec {
					docVector[i] += vec[i] * weight
				}
				validCount++
			}
		}

		if validCount == 0 {
			return nil, fmt.Errorf("文档中没有有效的词向量")
		}

		// 归一化文档向量
		norm := 0.0
		for _, v := range docVector {
			norm += v * v
		}
		norm = math.Sqrt(norm)

		if norm > 0 {
			for i := range docVector {
				docVector[i] /= norm
			}
		}

		return docVector, nil
	}
}
