from transformers import LlamaTokenizer
import transformers
from torch import float16

def get_model_and_tokenizer():
    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = LlamaTokenizer.from_pretrained(model)

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=float16
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model, 
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer