from transformers import AutoTokenizer
import transformers

"""
On Disk Usage: 6.8GB
GPU VRAM Usage (Baseline after running Perplexity on WikiText2):
Perplexity (WikiText2):
MMLU:
MMLU Time taken:
BBH (limit=):

Not so good accuracy, upfront quantization required, but super low
space required on disk, faster inference.
"""


def get_model_and_tokenizer():
    """
    To save quantization time, we use a model that has already been quantized
    using the same approach as `quantizers/gptq.py`.
    """
    
    model = "TheBloke/Llama-2-13B-chat-GPTQ"

    tokenizer = AutoTokenizer.from_pretrained(model)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model, 
        device_map={"": 0},
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.eval()

    return model, tokenizer