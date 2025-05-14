package file

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	. "seetaSearch/library/common"
	. "seetaSearch/library/strategy"
	"time"
)

// File /**
type File struct {
	name     string
	filepath string
	size     int64
	content  []*TreeStruct
	mode     string
	isDir    bool
	/**
	* 文件数据类型有四种，ini型数据，yaml型数据，json型数据，data型数据，默认数据类型为data型
	 */
	dataType int
	modTime  time.Time
	fileAbs  string
}

func (f *File) fileInfo() (*File, error) {
	fi, err := os.Stat(f.fileAbs)
	if err != nil {
		return f, err
	} else if os.IsNotExist(err) {
		return f, err
	}
	f.size = fi.Size()
	f.mode = fi.Mode().String()
	f.modTime = fi.ModTime()
	f.isDir = fi.IsDir()
	f.dataType = f.SetDataType()
	return f, nil
}

func (f *File) getFilePath(fullname string) {
	fileAbs, err := filepath.Abs(fullname)
	if err != nil {
		return
	}
	fileName := path.Base(fullname)
	f.name = fileName
	f.fileAbs = fileAbs
}

func (f *File) SetDataType() (dataType int) {
	fileSuffix := path.Ext(f.name)
	switch fileSuffix {
	case IniSuffix:
		dataType = IniType
	case JsonSuffix:
		dataType = JsonType
	case YamlSuffix:
		dataType = YamlType
	default:
		dataType = DataType
	}
	return
}
func (f *File) GetDataType() int {
	return f.dataType
}
func (f *File) GetContent() []*TreeStruct {
	return f.content
}

func (f *File) createDir(fileName string) bool {
	if f.checkFileExist(fileName) {
		return true
	}
	paths, _ := filepath.Split(fileName)
	if f.checkFileExist(paths) {
		return true
	}
	err := os.MkdirAll(paths, os.ModePerm)
	if err != nil {
		fmt.Println("create file: " + fileName + " fail, msg:" + fmt.Sprintf("%s", err))
		return false
	}
	return true
}

func (f *File) checkFileExist(filename string) bool {
	var exist = true
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		exist = false
	}
	return exist
}

func Instance(name string) (*File, error) {
	fileObj := &File{}
	fileObj.getFilePath(name)
	fileObj, err := fileObj.fileInfo()

	if fileObj.isDir {
		return nil, fmt.Errorf("file is dir")
	}
	return fileObj, err
}
