from multiprocessing import Process, Manager

def process_data(data):
    # 模拟计算密集型任务
    result = data * 2
    return result

def process_function(data_chunk, result_list):
    for item in data_chunk:
        result = process_data(item)
        result_list.append(result)

def main():
    data = list(range(1000))  # 假设有一个需要处理的数据列表

    num_processes = 4
    chunk_size = len(data) // num_processes

    with Manager() as manager:
        result_list = manager.list()
        processes = []

        for i in range(num_processes):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            data_chunk = data[start_idx:end_idx]

            process = Process(target=process_function, args=(data_chunk, result_list))
            processes.append(process)

        # 启动所有进程
        for process in processes:
            process.start()

        # 等待所有进程完成
        for process in processes:
            process.join()

        print("Result List:", list(result_list))

if __name__ == "__main__":
    main()
