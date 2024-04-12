from transformers import LlamaTokenizer
import transformers
from torch import float16

"""
bitsandbytes is the easiest option for quantizing a model to 4-bit. 
4-bit quantization multiplies outliers in fp16 with non-outliers in int4, 
converts the non-outlier values back to fp16, and then adds them together 
to return the weights in fp16. This reduces the degradative effect outlier 
values have on a model's performance.
"""

"""
On Disk Usage: 13GB
GPU VRAM Usage (Baseline after batch size 1 Perplexity on WikiText2): 4530MB
Perplexity (WikiText2) (limit=): 97.15
Perplexity Time taken (batch size 1 on above limit):
MMLU (limit=):
MMLU Time taken (batch size 1 on above limit):
BBH (limit=):
BBH Time taken (batch size 1 on above limit):
"""

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
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.eval()

    return model, tokenizer