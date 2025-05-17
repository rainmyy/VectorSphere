package server

import (
	"context"
	"errors"
	"fmt"
	"seetaSearch/index"
	"seetaSearch/library/common"
	"seetaSearch/library/log"
	"seetaSearch/messages"
	"strconv"
	"time"
)

type ServerInterface interface {
	DelDoc(id *messages.DocId) int
	AddDoc(document *messages.Document) (int, error)
	Search(query *messages.TermQuery, onFlag, offFlag uint64, orFlags []uint64) []*messages.Document
	Count() int
}

type IndexServer struct {
	Index     *index.Index
	hub       *EtcdServiceHub
	stop      bool
	localhost string
}

func (w *IndexServer) Init(docNumEstimate int, dbType int, DataDir string) error {
	w.Index = &index.Index{}
	return w.Index.NewIndexServer(docNumEstimate, dbType, "", DataDir)
}

func (w *IndexServer) RegisterService(servers []string, port int) error {
	if len(servers) == 0 {
		return errors.New("servers is empty")
	}
	if port <= 1024 || port > 65535 {
		return errors.New("port is out of range")
	}
	localIp, err := common.GetLocalHost()
	if err != nil {
		return fmt.Errorf("get local ip err: %v", err)
	}
	w.localhost = localIp + ":" + strconv.Itoa(port)
	var heartBeat int64 = 3
	hub := GetHub(servers, heartBeat)
	endPoint := &EndPoint{address: w.localhost}
	leasID, err := hub.RegisterService(IndexService, endPoint, 0)
	if err != nil {
		return fmt.Errorf("reigister fialed:%v", err)
	}
	w.hub = hub
	go func() {
		for !w.stop {
			_, err := hub.RegisterService(IndexService, endPoint, leasID)
			if err != nil {
				log.Logger.Printf("续约服务失败,租约ID:%d, 错误:%v", leasID, err)
			}
			time.Sleep(time.Duration(heartBeat)*time.Second - 100*time.Millisecond)
		}
	}()

	return nil
}

func (w *IndexServer) StopService() {
	w.stop = true
}

func (w *IndexServer) LoadFromIndexFile() (int, error) {
	data, err := w.Index.LoadIndex()
	if err != nil {
		return -1, err
	}
	return data, nil
}
func (w *IndexServer) DeleteDoc(ctx context.Context, docId *messages.DocId) (*messages.ReqCount, error) {
	// 调用Indexer的DeleteDoc方法删除文档，并返回影响的文档数量
	return &messages.ReqCount{
		Count: int32(w.Index.DelDoc(docId.Id)),
	}, nil
}

func (w *IndexServer) AddDoc(ctx context.Context, doc *messages.Document) (*messages.ReqCount, error) {
	// 调用Indexer的AddDoc方法添加文档，并返回影响的文档数量
	n, err := w.Index.AddDoc(*doc)
	return &messages.ReqCount{
		Count: int32(n),
	}, err
}

func (w *IndexServer) Search(ctx context.Context, request *messages.Request) (*messages.Result, error) {
	// 调用Indexer的Search方法进行检索，并返回检索结果
	result, err := w.Index.Search(request.Query, request.OnFlag, request.OffFlag, request.OrFlags)

	if err != nil {
		return nil, err
	}

	return &messages.Result{
		Results: result,
	}, nil
}

func (w *IndexServer) Count(ctx context.Context, request *messages.CountRequest) (*messages.ReqCount, error) {
	// 调用Indexer的Count方法获取文档数量，并返回结果
	return &messages.ReqCount{
		Count: int32(w.Index.Total()),
	}, nil
}
func (w *IndexServer) Close() error {
	if w.hub == nil {
		return w.Index.Close()
	}
	endPoint := &EndPoint{address: w.localhost}
	err := w.hub.UnRegisterService(IndexService, endPoint)
	if err != nil {
		log.Logger.Printf("注销服务失败，服务地址: %v, 错误: %v", w.localhost, err)
		return err
	}
	log.Logger.Printf("注销服务成功，服务地址: %v", w.localhost)
	return w.Index.Close()
}
