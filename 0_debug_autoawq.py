from common import *

hf_model="bert-base-cased"
hf_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
resaved_dir="./models/TinyLlama-1.1B-Chat-v1.0/"

model_path="/data1/wjb/llm/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(hf_model)
# tokenizer.save_pretrained(resaved_dir)

dummy_model_input = tokenizer("This is a sample", return_tensors="pt")#["input_ids"]#token_type_ids,attention_mask

# sequences = ["Hello!", "Cool.", "Nice!"]
# encoded_sequences = [
#     [101, 7592, 999, 102],
#     [101, 4658, 1012, 102],
#     [101, 3835, 999, 102],
# ]
# dummy_model_input = torch.tensor(encoded_sequences)
# out=model(dummy_model_input)
# print("demo out:\n",out)


## 模型量化参数
quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM" }
quant_str=""
for k,v in quant_config.items():
    quant_str=quant_str+str(k)+"_"+str(v)


quant_path=os.path.join(quant_base_dir,model_path.split("/")[-1],quant_str)
if not os.path.exists(quant_path):
    os.system("mkdir -p "+quant_path)
calib_data=get_calib_data()
print("len calib data len:",len(calib_data))
print("begin load")
# # 模型、tokenizer加载
model = AutoAWQForCausalLM.from_pretrained(
    hf_model, **{"low_cpu_mem_usage": True, "use_cache": False}
).cuda()
model.eval()
tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
# print(model)
# for i,module in enumerate(model.modules()):
#     print("\n",i)
#     print(module)
# exit(0)
print("begin quant")
# 量化
# model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext())
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

# 模型保存
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')