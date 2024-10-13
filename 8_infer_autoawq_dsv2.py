from awq import AutoAWQForCausalLM
from awq.utils.utils import get_best_device
from transformers import AutoTokenizer, TextStreamer
import os,sys,torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

quant_path = "/data1/wjb/llm/models/DeepSeek-Coder-V2-Lite-Base/zero_point_Trueq_group_size_64w_bit_4version_GEMM"
quant_path="/data1/wjb/llm/models/DeepSeek-Coder-V2-Lite-Base/zero_point_Trueq_group_size_64w_bit_4version_GEMM_0"
quant_path="/data1/wjb/llm/models/DeepSeek-Coder-V2-Instruct/zero_point_Trueq_group_size_64w_bit_4version_GEMM"
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

# tokenizer = AutoTokenizer.from_pretrained(quant_path)
tokenizer = AutoTokenizer.from_pretrained("/data1/wjb/llm/models/deepseek-ai/DeepSeek-Coder-V2-Instruct")



input_text = "#write a quick sort algorithm"
# inputs = tokenizer(input_text, return_tensors="pt").cuda()
# outputs = model.generate(**inputs, max_length=128)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


streamer = TextStreamer(tokenizer,  skip_special_tokens=True)#skip_prompt=True,
inputs = tokenizer(input_text, return_tensors='pt').to("cuda")#.input_ids
# tokens = torch.randint(0, tokenizer.vocab_size, (1,64)).cuda()


model = AutoAWQForCausalLM.from_quantized(
    quant_path,
    trust_remote_code=True,
)#.cuda()
model.eval()


# Generate output
generation_output = model.generate(
    **inputs,
    streamer=streamer,
    max_new_tokens=512,
)



exit(0)
# 加载模型
if get_best_device() == "cpu":
    print("use cpu")
    model = AutoAWQForCausalLM.from_quantized(quant_path, use_qbits=True, fuse_layers=False)
else:
    print("not use cpu")
    model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True,max_seq_len=64)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"

chat = [
    {"role": "system", "content": "You are a concise assistant that helps answer questions."},
    {"role": "user", "content": prompt},
]

chat=[{"role":"user","content":"Write a piece of quicksort code in C++."}]
terminators = [
    tokenizer.eos_token_id,
    # tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


tokens = tokenizer.apply_chat_template(
    chat,
    return_tensors="pt"
)
tokens = tokens.to(get_best_device())

# 输出
generation_output = model.generate(
    tokens,
    streamer=streamer,
    max_new_tokens=64,
    eos_token_id=terminators
)