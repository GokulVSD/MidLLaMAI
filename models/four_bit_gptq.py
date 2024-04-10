from optimum.gptq import GPTQQuantizer
from transformers import LlamaTokenizer
import transformers
import json
import os
import torch


def get_model_and_tokenizer():
    """
    To save quantization time, we use a model that has already been quantized
    using the same approach as the function `train_model_and_tokenizer`.
    """
    
    model = "TheBloke/Llama-2-7B-chat-GPTQ"

    tokenizer = LlamaTokenizer.from_pretrained(model)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model, 
        device_map={"": 0},
    )

    model.eval()

    return model, tokenizer    


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
    quantized_model.save_pretrained(".weights/4b-gptq", safe_serialization=True)
    tokenizer.save_pretrained(".weights/4b-gptq")

    with open(os.path.join(".weights/4b-gptq", "quantize_config.json"), "w", encoding="utf-8") as f:
        quantizer.disable_exllama = False
        json.dump(quantizer.to_dict(), f, indent=2)