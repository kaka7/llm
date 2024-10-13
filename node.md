## env
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
conda create --name wjbpy310 python=3.10
conda activate wjbpy310
pip install vllm==v0.5.3  --upgrade -i https://mirrors.aliyun.com/pypi/simple/
pip install timm==1.0.8 openai==1.40.3 lmdeploy[all]==0.5.3 --upgrade -i https://mirrors.aliyun.com/pypi/simple/
pip install 'ms-swift[all]' -U ms-swift==2.4.0.post1 -i https://mirrors.aliyun.com/pypi/simple/
pip install autoawq==0.2.6  --upgrade -i https://mirrors.aliyun.com/pypi/simple/

## profile
    # nsys profile  --stats=true --trace=cuda,cudnn,cublas,nvtx,osrt,oshmem python llm/vllm_test.py
    
## 文档

https://docs.vllm.ai/en/latest/dev/kernel/paged_attention.html
https://nvidia.github.io/TensorRT-LLM/performance/perf-best-practices.html
TensorRT-llm https://zhuanlan.zhihu.com/p/692445786

## vllm
### openai API 服务：

vllm serve /data/wjb/llm/models/Meta-Llama-3-8B

curl http://0.0.0.0:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "llama3-8b",
"prompt": "San Francisco is a",
"max_tokens": 64,
"temperature": 0
}'

### api_server
python -m vllm.entrypoints.api_server --model /data/wjb/llm/models/Meta-Llama-3-8B

#Query the model in shell:
curl http://0.0.0.0:8000/generate \
    -d '{
        "prompt": "Funniest joke ever:",
        "n": 1,
        "temperature": 0.95,
        "max_tokens": 200
    }'

## lmdeploy 

  --backend  pytorch,turbomind
  --tp 
  --dtype float16 

lmdeploy serve api_server \
     ${model_path} \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 8000 \
    --tp 1
    <!-- --model-format hf \ -->

python scripts/vllm_benchmarks/benchmark_serving.py --backend 'lmdeploy' --model ${model_path}  --dataset-path  /data/wjb/llm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --dataset-name sharegpt --num-prompts 100 --request-rate 10 --sharegpt-output-len 80
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  10.05     
Total input tokens:                      23304     
Total generated tokens:                  4523      
Request throughput (req/s):              9.95      
Input token throughput (tok/s):          2318.85   
Output token throughput (tok/s):         450.06    
---------------Time to First Token----------------
Mean TTFT (ms):                          23.74     
Median TTFT (ms):                        20.45     
P99 TTFT (ms):                           83.01     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.74     
Median TPOT (ms):                        13.37     
P99 TPOT (ms):                           21.22     
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.76     
Median ITL (ms):                         11.79     
P99 ITL (ms):                            59.68     
==================================================


lmdeploy serve api_server -h
lmdeploy serve api_client http://0.0.0.0:8000 cli client
 


## swift
--infer_backend --infer_backend`: 你可以选择'AUTO', 'vllm', 'pt'. 默认使用'AUTO
https://github.com/modelscope/ms-swift/blob/v2.4.0.post1/docs/source/Multi-Modal/vLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E6%96%87%E6%A1%A3.md vllm
https://github.com/modelscope/ms-swift/blob/v2.4.0.post1/docs/source/Multi-Modal/LmDeploy%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E6%96%87%E6%A1%A3.md lmdepoly
https://github.com/modelscope/ms-swift/blob/v2.4.0.post1/docs/source/LLM/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.md  swift帮助文档

export CUDA_VISIBLE_DEVICES=6,7 swift infer     --model_type deepseek-v2-chat     --model_id_or_path /data/models/DeepSeek-V2-Chat-0628    --quant_method bnb     --quantization_bit 4

CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_id_or_path /data/models/deepseek-coder-v2-instruct-awq --model_type deepseek-coder-v2-instruct --quant_method awq     --quantization_bit 8 --do_sample true
CUDA_VISIBLE_DEVICES=1,2 swift infer  --model_id_or_path  ${model_path}     --infer_backend lmdeploy --tp 2 --model_type llama3-8b
<<
CUDA_VISIBLE_DEVICES=1,2 swift deploy  --model_id_or_path  ${model_path}     --infer_backend lmdeploy --tp 2 --model_type llama3-8b
// openai




 CUDA_VISIBLE_DEVICES=1,2 swift deploy --model_id_or_path ${model_path} --infer_backend lmdeploy --tp 2 --model_type llama3-8b
curl http://0.0.0.0:8000/v1/completions -H "Content-Type: application/json" \
    -d '{
    "model": "llama3-8b",
    "prompt": "tell me a joke",
    "max_tokens": 32,
    "temperature":0.1,
    "logprobs":10,
    "seed": 42
    }'
error model 重复的key
python scripts/vllm_benchmarks/benchmark_serving.py --backend 'lmdeploy' --model  ${model_path}  --dataset-path  /data/wjb/llm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --dataset-name sharegpt --num-prompts 100 --request-rate 10 --sharegpt-output-len 80

model_path=/data/wjb/llm/models/Meta-Llama-3-8B

### 请求用时
python scripts/vllm_benchmarks/benchmark_latency.py --model ${model_path} --input-len 700 --output-len 80 --batch-size 10 --trust-remote-code --dtype 'bfloat16' --max-model-len 4096  # --kv-cache-dtype 'fp8'
Avg latency: 1.5671795869944618 seconds
10% percentile latency: 1.563765872339718 seconds
25% percentile latency: 1.5641336122353096 seconds
50% percentile latency: 1.5648210150538944 seconds
75% percentile latency: 1.5657598387042526 seconds
90% percentile latency: 1.5664633167209103 seconds
99% percentile latency: 1.614079956446076 seconds

python scripts/vllm_benchmarks/benchmark_latency.py --model ${model_path} --input-len 700 --output-len 80 --batch-size 10 --trust-remote-code --dtype 'bfloat16' --max-model-len 4096   --kv-cache-dtype 'fp8'
Avg latency: 2.3462807028670793 seconds
10% percentile latency: 2.3414108017808757 seconds
25% percentile latency: 2.3418073322391137 seconds
50% percentile latency: 2.3426827649818733 seconds
75% percentile latency: 2.3437288750428706 seconds
90% percentile latency: 2.3445479212561624 seconds
99% percentile latency: 2.4186123890418094 seconds

### test_throughput

python scripts/vllm_benchmarks/benchmark_throughput.py --backend vllm --model ${model_path} --trust-remote-code --output-len 80 --num-prompts 500
Throughput: 20.22 requests/s, 11970.13 tokens/s

--kv-cache-dtype fp8 --tensor-parallel-size 2 --request-rate 128
### request-rate 10
python -m vllm.entrypoints.openai.api_server --model ${model_path} --dtype bfloat16 --trust-remote-code --max-model-len 4096
python scripts/vllm_benchmarks/benchmark_serving.py --backend 'vllm' --model ${model_path}  --dataset-path  /data/wjb/llm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --dataset-name sharegpt --num-prompts 100 --request-rate 10 --sharegpt-output-len 80
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  102.35    
Total input tokens:                      224330    
Total generated tokens:                  44277     
Request throughput (req/s):              9.77      
Input token throughput (tok/s):          2191.73   
Output token throughput (tok/s):         432.59    
---------------Time to First Token----------------
Mean TTFT (ms):                          28.53     
Median TTFT (ms):                        28.65     
P99 TTFT (ms):                           122.05    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          26.83     
Median TPOT (ms):                        25.86     
P99 TPOT (ms):                           51.33     
---------------Inter-token Latency----------------
Mean ITL (ms):                           26.28     
Median ITL (ms):                         19.21     
P99 ITL (ms):                            125.61    
==================================================
python scripts/vllm_benchmarks/benchmark_serving.py --backend 'lmdeploy' --model ${model_path}  --dataset-path  /data/wjb/llm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --dataset-name sharegpt --num-prompts 100 --request-rate 10 --sharegpt-output-len 80


============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  102.29    
Total input tokens:                      224330    
Total generated tokens:                  44380     
Request throughput (req/s):              9.78      
Input token throughput (tok/s):          2193.11   
Output token throughput (tok/s):         433.87    
---------------Time to First Token----------------
Mean TTFT (ms):                          28.00     
Median TTFT (ms):                        28.26     
P99 TTFT (ms):                           108.88    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          26.80     
Median TPOT (ms):                        25.58     
P99 TPOT (ms):                           50.81     
---------------Inter-token Latency----------------
Mean ITL (ms):                           26.32     
Median ITL (ms):                         19.24     
P99 ITL (ms):                            122.39    
==================================================



sft 参数 微调
pt 参数 PT参数继承了sft参数，并修改了部分默认值
rlhf 参数 RLHF参数继承了sft参数, 除此之外增加了以下参数:
infer merge-lora 参数
    lmdeploy 参数
        参考文档: https://lmdeploy.readthedocs.io/en/latest/api/pipeline.html#turbomindengineconfig
        --tp: tensor并行, 用于初始化lmdeploy引擎的参数, 默认值为1.
        --cache_max_entry_count: 初始化lmdeploy引擎的参数, 默认值为0.8.
        --quant_policy: Key-Value Cache量化, 初始化lmdeploy引擎的参数, 默认值为0, 你可以设置为4, 8.
        --vision_batch_size: 初始化lmdeploy引擎的参数, 默认值为1. 该参数只有在使用多模态模型时生效.
    vLLM 参数
        参考文档: https://docs.vllm.ai/en/latest/models/engine_args.html
        vllm 支持的模型 https://github.com/modelscope/ms-swift/blob/v2.4.0.post1/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md#%E6%A8%A1%E5%9E%8B
        --gpu_memory_utilization: 初始化vllm引擎EngineArgs的参数, 默认为0.9. 该参数只有在使用vllm时才生效. VLLM推理加速和部署可以查看VLLM推理加速与部署.
        --tensor_parallel_size: 初始化vllm引擎EngineArgs的参数, 默认为1. 该参数只有在使用vllm时才生效.
        --max_num_seqs: 初始化vllm引擎EngineArgs的参数, 默认为256. 该参数只有在使用vllm时才生效.
        --max_model_len: 覆盖模型的max_model_len, 默认为None. 该参数只有在使用vllm时才生效.
        --disable_custom_all_reduce: 是否禁用自定义的all-reduce kernel, 而回退到NCCL. 默认为True, 这与vLLM的默认值不同.
        --enforce_eager: vllm使用pytorch eager模式还是建立cuda graph. 默认为False. 设置为True可以节约显存, 但会影响效率.
        --vllm_enable_lora: 默认为False. 是否开启vllm对lora的支持. 具体可以查看VLLM & LoRA.
        --vllm_max_lora_rank: 默认为16. vllm对于lora支持的参数.
        --lora_modules: 已介绍.
export 参数 
    支持ollama ，量化
    export参数继承了infer参数, 除此之外增加了以下参数:



deploy 参数
    deploy参数继承了infer参数, 除此之外增加了以下参数:
    --host: 默认为'127.0.0.1. 要使其在非本机上可访问, 可设置为'0.0.0.0'.
    --port: 默认为8000.
    --api_key: 默认为None, 即不对请求进行api_key验证.
    --ssl_keyfile: 默认为None.
    --ssl_certfile: 默认为None.
    --verbose: 是否对请求内容进行打印, 默认为True.
    --log_interval: 对统计信息进行打印的间隔, 单位为秒. 默认为10. 如果设置为0, 表示不打印统计信息.

eval参数 
    eval参数继承了infer参数，除此之外增加了以下参数：（注意: infer中的generation_config参数将失效, 由evalscope控制.）
    评测的官方数据集

app-ui 参数
    app-ui参数继承了infer参数, 除此之外增加了以下参数:

    --host: 默认为'127.0.0.1'. 传递给gradio的demo.queue().launch(...)函数.
    --port: 默认为7860. 传递给gradio的demo.queue().launch(...)函数.
    --share: 默认为False. 传递给gradio的demo.queue().launch(...)函数.


https://github.com/modelscope/ms-swift?tab=readme-ov-file
https://github.com/modelscope/ms-swift/blob/v2.4.0.post1/docs/source/LLM/LLM%E6%8E%A8%E7%90%86%E6%96%87%E6%A1%A3.md llm推理
https://github.com/modelscope/ms-swift/blob/v2.4.0.post1/docs/source/LLM/LLM%E8%AF%84%E6%B5%8B%E6%96%87%E6%A1%A3.md llm 效果评测
https://github.com/modelscope/ms-swift/blob/v2.4.0.post1/docs/source/LLM/LLM%E9%87%8F%E5%8C%96%E6%96%87%E6%A1%A3.md llm量化

https://github.com/modelscope/ms-swift/blob/v2.4.0.post1/docs/source/LLM/%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%8E%E6%8B%93%E5%B1%95.md 自定义和拓展


## 部署
评测：https://zhuanlan.zhihu.com/p/703474709  https://github.com/vllm-project/vllm/tree/main/benchmarks

## 深入

    Throughput is calculated as output tokens per second per gpu. out_tps=output_seqlen*batch_size/total_latency/tp
    优化参数：max_batch_size, max_seq_len and max_num_tokens

    encoder-only：BERT
    encoder-decoder：T5, GLM-130B, UL2
    decoder-only：GPT系列, LLaMA, OPT, PaLM,BLOOM
    了解典型 Decoder-only 语言模型的基础结构和简单原理。


    掌握 Continue Pre-train、Fine-tuning 已有开源模型的能力；
    掌握 Lora、QLora 等最小化资源进行高效模型训练的PEFT技术；
    掌握强化学习基础；
    Alignment与RLHF；
    数据处理技术；
    压缩模型、推理加速技术；
    分布式训练并行技术；
    分布式网络通信技术；
    生产环境部署大模型的相关技术。

## MLC LLM

## vllm
    最牛的serving 吞吐量
    PagedAttention对kv cache的有效管理
    传入请求的continus batching，而不是static batching
    高性能CUDA kernel
    流行的HuggingFace模型无缝集成
    有各种decoder算法的高吞吐量服务，包括parallel sampling和beam search等
    tensor parallel
    兼容OpenAI的API服务器

    SamplingParams 重要推理超参数
    do_sample：是否使用随机采样方式运行推理，如果设置为False，则使用beam_search方式
    temperature：大于等于零的浮点数。公式为：
    取值为0，则效果类似argmax，此时推理几乎没有随机性；取值为正无穷时接近于取平均。一般temperature取值介于[0, 1]之间。取值越高输出效果越随机。
    如果该问答只存在确定性答案，则T值设置为0。反之设置为大于0。

    top_k：大于0的正整数。从k个概率最大的结果中进行采样。k越大多样性越强，越小确定性越强。
    top_p：大于0的浮点数。使所有被考虑的结果的概率和大于p值，p值越大多样性越强，越小确定性越强。一般设置0.7~0.95之间。实际实验中可以先从0.95开始降低，直到效果达到最佳。
    top_p比top_k更有效，应优先调节这个参数。
    repetition_penalty： 大于等于1.0的浮点数。如何惩罚重复token，默认1.0代表没有惩罚。




Ollama，作为Llama.cpp和Llamafile的升级之选

## 量化
AWQ: Activation-aware Weight Quantization,
GPTQ: Accurate Post-Training Quantization
https://zhuanlan.zhihu.com/p/666569378
https://blog.csdn.net/qq_48191249/article/details/140358123 awq
https://zhuanlan.zhihu.com/p/689358526 awq
vllm 支持awq https://github.com/vllm-project/vllm/pull/1714/files https://github.com/vllm-project/vllm/pull/926/files




I recommend using https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1 or https://huggingface.co/datasets/teknium/OpenHermes-2.5.

## build vllm
(wjbpy310_b) root@bm-2204ns8:/data/wjb/vllm# pip install -e . -v -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com



Make sure you have already know the theory and goal of TP.Usually, TP is use to solve the bottleneck of memory, for small size model, there is no need to use TP, multi-instances is better than use TP. Because when you use TP to a small model, you will meet the computing bottleneck of the GPU card itself. Considering of the communication cost, maybe you can only get 40-50% improvement when you use 2-GPUs(comparing to 1-GPU).tensor_parallel_size=2 means spliting the model into two GPUs rather than running two full models. 层数是tensor_parallel_size整数倍
小模型适合多个port
for X in 0..8
CUDA_VISIBLE_DEVICES=X python -m vllm.entrypoints.openai.api_server \
    --model /data/models/Qwen2-7B-Instruct/ \
    --served-model-name aaa-X \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --port 800X


python -m ktransformers.local_chat --model_path  /data/wjb/llm/../../models/deepseek-coder-v2-instruct  --gguf_path /data/wjb/llm/../../models/bartowski/DeepSeek-V2.5-GGUF/DeepSeek-V2.5-Q4_K_M/

 python -m ktransformers.local_chat --model_path  /data/wjb/llm/dsv25  --gguf_path /data/models/bartowski/DeepSeek-V2.5-GGUF/DeepSeek-V2.5-Q4_K_M/ --optimize_rule_path /data/wjb/llm/DeepSeek-V2-Chat-new.yaml

 https://www.runoob.com/regexp/regexp-syntax.html
 https://www.jyshare.com/front-end/854/

 nohup huggingface-cli download --resume-download   bartowski/DeepSeek-V2.5-GGUF --include "*Q4_K_M-00004*"  --token $hf_token --repo-type "model" --local-dir . &
