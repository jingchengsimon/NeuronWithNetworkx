import threading

def process_data(data):
    # 模拟计算密集型任务
    result = data * 2
    return result

def threaded_function(data_chunk, result_list):
    for item in data_chunk:
        result = process_data(item)
        result_list.append(result)

def main():
    data = list(range(1000))  # 假设有一个需要处理的数据列表

    num_threads = 4
    chunk_size = len(data) // num_threads

    result_list = []
    threads = []

    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        data_chunk = data[start_idx:end_idx]

        thread = threading.Thread(target=threaded_function, args=(data_chunk, result_list))
        threads.append(thread)

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    print("Result List:", result_list)

if __name__ == "__main__":
    main()
