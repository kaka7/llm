#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

# models=("meta-llama/Meta-Llama-3.1-70B-Instruct")
#  "meta-llama/Meta-Llama-3.1-8B-Instruct"

download_dir="/data/wjb/llm/models"

hf_token="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"
# model="meta-llama/Meta-Llama-3.1-70B-Instruct"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
model="mistralai/Mixtral-8x7B-Instruct-v0.1"

# for model in "${models[@]}"; do

  save_path="$download_dir/$model"

  echo "Downloading $model..."
  until huggingface-cli download --token $hf_token --local-dir-use-symlinks "False" --resume-download $model --repo-type "model" --local-dir $save_path; do 
    echo "Download failed, retrying..."
  done
  echo "$model downloaded and saved."

  echo "success: $model"
# done

echo "All Models downloaded successfully."

#nohup ./batch_download_models.sh > log.txt 2>&1 &

# python -m vllm.entrypoints.openai.api_server --model "/data/models/CSG-Wukong-Chinese-Llama-3.1-405B-instruct" --served-model-name "405B" 