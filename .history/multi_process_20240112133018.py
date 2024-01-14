import multiprocessing

def your_function(parameter):
    # 在这里编写您的代码，使用 parameter 参数

def run_processes():
    parameters_list = [param1, param2, param3]  # 以列表形式提供不同的参数

    processes = []
    for param in parameters_list:
        process = multiprocessing.Process(target=your_function, args=(param,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    run_processes()
