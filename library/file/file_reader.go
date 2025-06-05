package file

import (
	. "VectorSphere/library/common"
	"VectorSphere/library/parser"
	. "VectorSphere/library/tree"
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"sync"
)

// Parser 解析数据，将数据解析成树形结构进行存储
func (f *File) Parser() error {
	err := f.readFile()
	if err != nil {
		return err
	}
	return nil
}

// file size 1GB
var defaultSize int64 = 1 << 30

func (f *File) readFile() error {
	fileName := f.fileAbs
	fi, err := os.Open(fileName)
	defer fi.Close()
	if err != nil {
		return err
	}
	fileSize := f.size
	if fileSize == 0 {
		fiStat, err := fi.Stat()
		if err != nil {
			return err
		}
		fileSize = fiStat.Size()
	}
	// mmap缓存文件内容
	data, err := MmapFile(fileName, int(fileSize))
	if err != nil {
		return err
	}
	defer func() {
		if data != nil {
			MunmapFile(data)
		}
	}()
	// 大文件并发读取
	if fileSize > defaultSize && f.dataType == DataType {
		return f.readFileByConcurrentMmap(data)
	} else {
		return f.readFileByGeneralMmap(data)
	}
}

// 普通读取（mmap缓存）
func (f *File) readFileByGeneralMmap(data []byte) error {
	if data == nil {
		return fmt.Errorf("mmap data is nil")
	}
	tree, err := parserDataFunc(f, data)
	if err != nil {
		return err
	}
	f.content = tree
	return nil
}

// 并发读取（mmap缓存）
func (f *File) readFileByConcurrentMmap(data []byte) error {
	if data == nil {
		return fmt.Errorf("mmap data is nil")
	}
	lines := strings.Split(string(data), "\n")
	chunkSize := 10000 // 可根据实际情况调整
	var wg sync.WaitGroup
	treeChan := make(chan []*TreeStruct, len(lines)/chunkSize+1)
	for i := 0; i < len(lines); i += chunkSize {
		end := i + chunkSize
		if end > len(lines) {
			end = len(lines)
		}
		wg.Add(1)
		go func(chunk []string) {
			defer wg.Done()
			chunkData := []byte(strings.Join(chunk, "\n"))
			tree, _ := parserDataFunc(f, chunkData)
			if tree != nil {
				treeChan <- tree
			}
		}(lines[i:end])
	}
	wg.Wait()
	close(treeChan)
	var allTrees []*TreeStruct
	for t := range treeChan {
		allTrees = append(allTrees, t...)
	}
	f.content = allTrees
	return nil
}

func (f *File) readFileByGeneral(fileObj *os.File) error {
	if fileObj == nil {
		return fmt.Errorf("file is nil")
	}
	r := bufio.NewReader(fileObj)
	b := make([]byte, f.size)
	for {
		_, err := r.Read(b)
		if err != nil && err == io.EOF {
			break
		}
	}

	tree, err := parserDataFunc(f, b)
	if err != nil {
		return err
	}
	f.content = tree
	return nil
}

/**
* 并发读取,所有字符串按行分割， 暂不支持多行关联行数据
 */
func (f *File) readFileByConcurrent(fileObj *os.File) error {
	liensPool := sync.Pool{New: func() interface{} {
		lines := make([]byte, 500*1024)
		return lines
	}}
	stringPool := sync.Pool{New: func() interface{} {
		lines := ""
		return lines
	}}
	slicePool := sync.Pool{New: func() interface{} {
		lines := make([]string, 100)
		return lines
	}}
	r := bufio.NewReader(fileObj)
	var wg sync.WaitGroup
	for {
		buf := liensPool.Get().([]byte)
		n, err := r.Read(buf)
		if n == 0 {
			if err != nil {
				break
			}
			if err == io.EOF {
				break
			}
			return err
		}
		nextLine, err := r.ReadBytes('\n')
		if err != io.EOF {
			buf = append(buf, nextLine...)
		}
		wg.Add(1)
		go func() {
			ProcessChunk(buf, &liensPool, &stringPool, &slicePool)
			wg.Done()
		}()
	}

	wg.Wait()
	return nil
}

func ProcessChunk(chunk []byte, linesPool *sync.Pool, stringPool *sync.Pool, slicePool *sync.Pool) {
	var wg2 sync.WaitGroup
	logs := stringPool.Get().(string)
	logs = string(chunk)
	linesPool.Put(chunk)
	logsSlice := strings.Split(logs, "\n")
	stringPool.Put(logs)
	chunkSize := 100
	n := len(logsSlice)
	threadNo := n / chunkSize
	if n%chunkSize != 0 {
		threadNo++
	}
	length := len(logsSlice)
	for i := 0; i < length; i += chunkSize {
		wg2.Add(1)
		go func(s int, e int) {
			for i := s; i < e; i++ {
				text := logsSlice[i]
				if len(text) == 0 {
					continue
				}
				//tree, _ := parserDataFunc(f, []byte(text))
			}
			wg2.Done()
		}(i*chunkSize, int(math.Min(float64((i+1)*chunkSize), float64(len(logsSlice)))))
	}
}

/**
* 所有的数
 */
func parserDataFunc(file *File, data []byte) ([]*TreeStruct, error) {
	var objType = file.GetDataType()
	switch objType {
	case IniType:
		return parser.ParserIniContent(data)
	case YamlType:
		return parser.ParserYamlContent(data)
	case JsonType:
		return parser.ParserJsonContent(data)
	default:
		return parser.ParserContent(data)
	}
}
