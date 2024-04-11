from transformers import AutoTokenizer
import transformers

"""
"""

"""
On Disk Usage: 
GPU VRAM Usage (Baseline after batch size 1 Perplexity on WikiText2):
Perplexity (WikiText2):
MMLU (limit=):
MMLU Time taken (batch size 1 on above limit):
BBH (limit=):

??
"""


def get_model_and_tokenizer():
    """
    ??
    """
    
    model = "neuralmagic/Llama2-7b-chat-pruned50-quant-ds"

    tokenizer = AutoTokenizer.from_pretrained(model)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model, 
        device_map={"": 0},
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.eval()

    return model, tokenizer