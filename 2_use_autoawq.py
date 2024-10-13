from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from modelscope.msdatasets import MsDataset
import os,shutil,datasets
# data = MsDataset.load('OpenHermes-2.5', subset_name='wikitext-2-v1', split='train')


data = datasets.load_dataset('nickrosh/Evol-Instruct-Code-80k-v1', split='train')
train_dataset = data.train_test_split(test_size=0.1)["train"]
# eval_dataset = data.train_test_split(test_size=0.1)["test"]

train_dataset=[it["instruction"] for it in train_dataset]


## 模型量化参数
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
quant_str=""
for k,v in quant_config.items():
    quant_str=quant_str+"-".join([str(k),str(v)])

model_path="/data/models/deepseek-coder-v2-instruct"
model_path="/data/models/DeepSeek-V2-Lite-Chat"

quant_path = "models/DeepSeek-V2-Lite-Chat"
quant_path=os.path.join("models",model_path.split("/")[-1],quant_str)
if not os.path.exists(quant_path):
    os.mkdir(quant_path)

# if os.path.exists(quant_path):
#     shutil.
    

# 模型、tokenizer加载
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
print(model)
# exit
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载数据
def load_wikitext():
    data = MsDataset.load('wikitext', subset_name='wikitext-2-v1', split='train')
    return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20]

# 量化
# model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext())
model.quantize(tokenizer, quant_config=quant_config, calib_data=train_dataset)

# 模型保存
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
