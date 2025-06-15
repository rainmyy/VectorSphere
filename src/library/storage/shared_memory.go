package storage

// SharedMemory 是共享内存管理器的接口
// 具体实现在 shared_memory_linux.go 和 shared_memory_w.go 中
// 根据操作系统环境自动选择相应的实现

// 共享内存管理器提供以下功能：
// - 创建或打开共享内存区域
// - 在共享内存中存储和检索数据
// - 管理共享内存中的键值对
// - 清理和关闭共享内存

// 使用示例：
// sm, err := NewSharedMemory()
// if err != nil {
//     // 处理错误
// }
// defer sm.Close()
//
// // 存储数据
// err = sm.Put("key1", []string{"value1", "value2"}, time.Now().Unix())
//
// // 检索数据
// results, timestamp, exists := sm.Get("key1")
