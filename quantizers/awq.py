from transformers import AutoTokenizer
import torch
from awq import AutoAWQForCausalLM


def train_model_and_tokenizer():

    model = "meta-llama/Llama-2-13b-chat-hf"
    quant = ".weights/4b-13B-awq"

    tokenizer = AutoTokenizer.from_pretrained(model)

    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    model = AutoAWQForCausalLM.from_pretrained(
        model, 
        device_map="auto",
        low_cpu_mem_usage=True, torch_dtype=torch.float16
    )
    
    model.quantize(tokenizer, quant_config=quant_config)
    
    model.save_quantized(quant)
    tokenizer.save_pretrained(quant)