model_path=/data/wjb/llm/models/Meta-Llama-3-8B
# python scripts/vllm_benchmarks/benchmark_throughput.py --model ${model_path} --num-prompts=1024  --output-len 256 --input-len 128  --backend vllm #  --tensor-parallel-size=4 
# python scripts/vllm_benchmarks/benchmark_throughput.py --model ${model_path} --num-prompts=1024  --output-len 256 --input-len 128  --backend hf --hf-max-batch-size 128

# python scripts/vllm_benchmarks/benchmark_throughput.py --model ${model_path} --num-prompts=1024  --output-len 1024 --input-len 1024  --backend vllm #  --tensor-parallel-size=4 
python scripts/vllm_benchmarks/benchmark_throughput.py --model ${model_path} --num-prompts=256  --output-len 1024 --input-len 1024  --backend hf --hf-max-batch-size 16

model_path=/data/wjb/llm/models/Qwen2-7B-Instruct
# python scripts/vllm_benchmarks/benchmark_throughput.py --model ${model_path} --num-prompts=1024  --output-len 256 --input-len 128  --backend vllm #  --tensor-parallel-size=4 
# python scripts/vllm_benchmarks/benchmark_throughput.py --model ${model_path} --num-prompts=1024  --output-len 256 --input-len 128  --backend hf --hf-max-batch-size 128

# python scripts/vllm_benchmarks/benchmark_throughput.py --model ${model_path} --num-prompts=1024  --output-len 1024 --input-len 1024  --backend vllm #  --tensor-parallel-size=4 
python scripts/vllm_benchmarks/benchmark_throughput.py --model ${model_path} --num-prompts=256  --output-len 1024 --input-len 1024  --backend hf --hf-max-batch-size 16

#   llama3-70b hf跑不起来
