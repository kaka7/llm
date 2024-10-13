import os,sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"]="/data/wjb/llm/models" 
quant_base_dir="/data/wjb/llm/models"
# os.system("/root/miniconda3/condabin/conda init && /root/miniconda3/condabin/conda activate wjbpy310")


import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from transformers import BertModel

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from modelscope.msdatasets import MsDataset
import os,shutil,datasets
from vllm import LLM, SamplingParams
from datasets import load_dataset
import time
from tqdm import tqdm
from transformers import AutoTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
print("use device:",device)
print("use device count:",torch.cuda.device_count())
def get_calib_data(task_name="",num=-1):
    if task_name=="code" or task_name=="":
        calib_data = datasets.load_dataset("json",data_files="./models/nickrosh/Evol-Instruct-Code-80k-v1/EvolInstruct-Code-80k.json")["train"]
        calib_data=[it["instruction"] for it in calib_data][:num]
    
    return calib_data

# 加载数据
def load_wikitext():
    data = MsDataset.load('wikitext', subset_name='wikitext-2-v1', split='train')
    return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20]

def load_openhermes_coding():
    data = load_dataset("alvarobartt/openhermes-preferences-coding", split="train")

    samples = []
    for sample in data:
        responses = [f'{response["role"]}: {response["content"]}' for response in sample["chosen"]]
        samples.append("\n".join(responses))

    return samples

import subprocess
 
def get_pid_by_port(port):
    # 在Linux系统中，可以使用netstat命令配合grep来查找端口对应的进程号
    # 这里使用subprocess.Popen来执行命令并获取输出
    netstat_cmd = "netstat -tulnp | grep :{}".format(port)
    # 使用shell=True是为了能够直接运行一个命令字符串，而不是传递一个参数列表
    process = subprocess.Popen(netstat_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # 解析netstat的输出结果，提取进程号
    if stdout:
        lines = stdout.decode().split('\n')
        for line in lines:
            if 'LISTEN' in line and str(port) in line:
                parts = line.split()
                # 假设进程号是第七个字段（这在不同的系统中可能会有所不同）
                return parts[6]
    return None