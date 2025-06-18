package backup

import (
	"VectorSphere/src/index"
	"VectorSphere/src/library/common"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/logger"
	"VectorSphere/src/proto/messages"
	"VectorSphere/src/proto/serverProto"
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"
)

//var _ IndexServiceServer = (*IndexServer)(nil)

type IndexServer struct {
	Index       *index.Index
	hub         *EtcdServiceHub
	localhost   string
	serviceName string
}

func (w *IndexServer) HealthCheck(ctx context.Context, request *serverProto.HealthCheckRequest) (*serverProto.HealthCheckResponse, error) {
	//TODO implement me
	panic("implement me")
}

func (w *IndexServer) ExecuteTask(ctx context.Context, request *serverProto.TaskRequest) (*serverProto.TaskResponse, error) {
	//TODO implement me
	panic("implement me")
}

func (w *IndexServer) ReportTaskResult(ctx context.Context, response *serverProto.TaskResponse) (*serverProto.ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (w *IndexServer) CreateTable(ctx context.Context, request *serverProto.CreateTableRequest) (*serverProto.ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (w *IndexServer) DeleteTable(ctx context.Context, request *serverProto.TableRequest) (*serverProto.ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (w *IndexServer) AddDocumentToTable(ctx context.Context, request *serverProto.AddDocumentRequest) (*serverProto.ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (w *IndexServer) DeleteDocumentFromTable(ctx context.Context, request *serverProto.DeleteDocumentRequest) (*serverProto.ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (w *IndexServer) SearchTable(ctx context.Context, request *serverProto.SearchRequest) (*serverProto.SearchResult, error) {
	//TODO implement me
	panic("implement me")
}

func (w *IndexServer) Init(docNumEstimate int, dbType int, DataDir string) error {
	w.Index = &index.Index{}
	return w.Index.NewIndexServer(docNumEstimate, dbType, "", DataDir)
}

func (w *IndexServer) RegisterService(servers []entity.EndPoint, port int, serviceName string) error {
	if len(servers) == 0 {
		return errors.New("servers is empty")
	}
	if port != 80 && (port <= 1024 || port > 65535) {
		return errors.New("port is out of range")
	}
	localIp, err := common.GetLocalHost()
	if err != nil {
		return fmt.Errorf("get local ip err: %v", err)
	}
	w.localhost = localIp + ":" + strconv.Itoa(port)
	var heartBeat int64 = 3
	err, hub := GetHub(servers, heartBeat, serviceName)
	if err != nil {
		return err
	}
	endPoint := &entity.EndPoint{Ip: w.localhost}
	leasID, err := hub.RegisterService(serviceName, endPoint, 0)
	if err != nil {
		return fmt.Errorf("reigister fialed:%v", err)
	}
	w.hub = hub
	w.serviceName = serviceName
	go func() {
		for !w.stop {
			_, err := hub.RegisterService(serviceName, endPoint, leasID)
			if err != nil {
				logger.Error("续约服务失败,租约ID:%d, 错误:%v", leasID, err)
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

func (w *IndexServer) DelDoc(ctx context.Context, docId *serverProto.DocId) (*serverProto.ResCount, error) {
	// 调用Indexer的DeleteDoc方法删除文档，并返回影响的文档数量
	return &serverProto.ResCount{
		Count: int32(w.Index.DelDoc(docId.Id)),
	}, nil
}

func (w *IndexServer) AddDoc(ctx context.Context, doc *messages.Document) (*serverProto.ResCount, error) {
	// 调用Indexer的AddDoc方法添加文档，并返回影响的文档数量
	n, err := w.Index.AddDoc(*doc)
	return &serverProto.ResCount{
		Count: int32(n),
	}, err
}

func (w *IndexServer) Search(ctx context.Context, request *serverProto.Request) (*serverProto.Result, error) {
	// 调用Indexer的Search方法进行检索，并返回检索结果
	result, err := w.Index.Search(request.Query, request.OnFlag, request.OffFlag, request.OrFlags)

	if err != nil {
		return nil, err
	}

	return &serverProto.Result{
		Results: result,
	}, nil
}

func (w *IndexServer) Count(ctx context.Context, request *serverProto.CountRequest) (*serverProto.ResCount, error) {
	// 调用Indexer的Count方法获取文档数量，并返回结果
	return &serverProto.ResCount{
		Count: int32(w.Index.Total()),
	}, nil
}
func (w *IndexServer) Close() error {
	if w.hub == nil {
		return w.Index.Close()
	}
	endPoint := &entity.EndPoint{Ip: w.localhost}
	err := w.hub.UnRegisterService(w.serviceName, endPoint)
	if err != nil {
		logger.Error("注销服务失败，服务地址: %v, 错误: %v", w.localhost, err)
		return err
	}
	logger.Info("注销服务成功，服务地址: %v", w.localhost)
	return w.Index.Close()
}
