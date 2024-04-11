from transformers import AutoTokenizer
import transformers
from torch import float16

"""
On Disk Usage: 13GB
GPU VRAM Usage (Baseline after running Perplexity on WikiText2):
Perplexity (WikiText2):
MMLU:
MMLU Time taken:
BBH (limit=):

Good accuracy, no upfront quantization required, large on-disk space
requirement, slow inference.
"""

def get_model_and_tokenizer():
    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=float16
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model, 
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.eval()

    return model, tokenizer