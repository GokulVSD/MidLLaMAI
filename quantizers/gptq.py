from transformers import LlamaTokenizer
import transformers
import json
import os
import torch
from optimum.gptq import GPTQQuantizer


def train_model_and_tokenizer():
    quantizer = GPTQQuantizer(bits=4, dataset="wikitext2", model_seqlen=4096)
    quantizer.quant_method = "gptq"

    model = "meta-llama/Llama-2-13b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model, 
        device_map="auto",
        low_cpu_mem_usage=True, torch_dtype=torch.float16
    )
    
    quantized_model = quantizer.quantize_model(model, tokenizer)
    quantized_model.save_pretrained(".weights/4b-13B-gptq", safe_serialization=True)
    tokenizer.save_pretrained(".weights/4b-13B-gptq")

    with open(os.path.join(".weights/4b-13B-gptq", "quantize_config.json"), "w", encoding="utf-8") as f:
        quantizer.disable_exllama = False
        json.dump(quantizer.to_dict(), f, indent=2)