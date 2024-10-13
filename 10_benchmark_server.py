import subprocess
import time,os
from common import get_pid_by_port

def get_gpu_model_and_count():
    """
    获取显卡模型名称和数量。

    :return: (显卡模型名称, 显卡数量)
    """
    # 查询显卡模型名称
    result_model = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], stdout=subprocess.PIPE)
    gpu_model = result_model.stdout.decode('utf-8').split('\n')[0].strip().replace(" ", "")

    # 查询显卡数量
    result_count = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], stdout=subprocess.PIPE)
    gpu_count = len(result_count.stdout.decode('utf-8').strip().split('\n')) - 1
    print(gpu_model,gpu_count)
    return gpu_model, gpu_count


def run_llm_and_test(model_path, tensor_parallel_size, request_rate=10,backend="vllm"):
    """
    运行LLM并对其进行性能测试。

    :param model_path: LLM模型的路径。
    :param tensor_parallel_size: 并行处理的大小。
    :param request_rate: 请求速率。
    """
    gpu_model, _ = get_gpu_model_and_count()

    # 准备启动LLM的命令
    if backend=="vllm":
        cmd1 = f"python -m vllm.entrypoints.openai.api_server --model {model_path} --tensor-parallel-size {tensor_parallel_size}"
    elif backend=="lmdeploy":
        cmd1 = f"lmdeploy serve api_server {model_path}  --quant-policy 0 --server-name 0.0.0.0  --server-port 8000   --tp {tensor_parallel_size}"
    model_name = model_path.split("/")[-1]
    # 定义服务器日志文件的路径
    print("server cmd:",cmd1)

    server_log_filename = f"./logs/{backend}_server_{model_name}-{gpu_model}x{tensor_parallel_size}x{request_rate}.log"

    try:
        # 启动LLM服务器并保存输出到日志文件
        with open(server_log_filename, 'w') as server_log_file:
            server_process = subprocess.Popen(cmd1, shell=True, stdout=server_log_file, stderr=server_log_file, text=True, bufsize=1, universal_newlines=True)

            # 检查LLM服务器的输出，等待"http://localhost:8000"或其他错误信息
            while True:
                with open(server_log_filename, 'r') as f:
                    content = f.read()
                    if "http://0.0.0.0:8000" in content:
                        break
                    # 如果检测到错误信息，抛出异常
                    if "Error" in content or "Exception" in content:
                        raise Exception(f"Error detected in cmd1 output. Check {server_log_filename} for details.")
                time.sleep(1)  # 等待1秒后再次检查
            print("load done,begin infer")
            # 定义进行性能测试的命令
            filename = f"{backend}_result_{model_name}-{gpu_model}x{tensor_parallel_size}x{request_rate}"
            cmd2 = f"python3 scripts/vllm_benchmarks/benchmark_serving.py  --backend {backend} --model {model_path}  --dataset-path  /data/wjb/llm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --dataset-name sharegpt --num-prompts {num_prompts}  --request-rate {request_rate} --result-dir logs --result-filename {filename} --sharegpt-output-len 512" #--dataset ShareGPT_V3_unfiltered_cleaned_split.json 
            print("client cmd:",cmd2)
            # 执行性能测试命令并保存输出到日志文件
            log_filename = f"./logs/{backend}_client_{model_name}-{gpu_model}x{tensor_parallel_size}x{request_rate}.log"
            with open(log_filename, 'w') as log_file:
                subprocess.run(cmd2, shell=True, stdout=log_file, stderr=log_file)
            

    except Exception as e:
        # 输出错误信息并终止LLM服务器
        print(f"Error encountered with model: {model_path} and tensor_parallel_size: {tensor_parallel_size}. Error details: {e}")
        server_process.terminate()
        server_process.wait()
        return  

    server_process.terminate()
    server_process.wait()
    if backend=="vllm":
        os.system(" lsof -t -i :8000 | xargs  -i kill -9 {}")
        os.system(" nvidia-smi --query-compute-apps=pid --format=csv | grep -v pid |xargs -i kill -9 {}")
        cmd3 = f"python scripts/vllm_benchmarks/benchmark_throughput.py --backend vllm --model {model_path} --trust-remote-code  --num-prompts  {num_prompts}  --tensor-parallel-size {tensor_parallel_size}"
        print("cmd3:",cmd3)
        with open(log_filename, 'a') as log_file:
            subprocess.run(cmd3, shell=True, stdout=log_file, stderr=log_file)
if __name__ == "__main__":
    # 定义需要测试的LLM模型路径列表

    models = [
        # "/mlx/users/xingzheng.daniel/playground/model/chinese-alpaca-2-13b",
        # "/mlx/users/xingzheng.daniel/playground/model/chinese-alpaca-2-7b",
        # "/mlx/users/xingzheng.daniel/playground/model/finetuning-FreedomIntelligence-phoenix-inst-chat-7b-v4"
        "/data/wjb/llm/models/Meta-Llama-3-70B-Instruct"

        # "/data/wjb/llm/models/Meta-Llama-3-8B",
        # "/data/wjb/llm/models/Qwen2-7B-Instruct"
    ]

    _, gpu_count = get_gpu_model_and_count()
    # 根据显卡数量定义tensor_parallel_sizes
    tensor_parallel_sizes = [1,2,4,8]
    # tensor_parallel_sizes = [1]
    backends=["vllm","lmdeploy"]#,"lmdeploy"
    # backends=["lmdeploy"]#,"lmdeploy"
    backends=["vllm"]#,"lmdeploy"

    request_rates=[1,10]
    request_rates=[10]

    num_prompts=1000

    # 对每个LLM模型进行测试
    for model in models:
        for tensor_parallel_size in tensor_parallel_sizes:

            for backend in backends:
                if tensor_parallel_size==1 and "0B" in model:
                    continue
                for request_rate in request_rates:
                    os.system(" lsof -t -i :8000 | xargs  -i kill -9 {}")
                    os.system(" nvidia-smi --query-compute-apps=pid --format=csv | grep -v pid |xargs -i kill -9 {}")
                    print("*"*50+"start ")
                    try:
                        run_llm_and_test(model, tensor_parallel_size, request_rate=request_rate,backend=backend)
                    except Exception as e:
                        print(e)

# --base_url http://127.0.0.1:8000  --dataset-name random

              