from transformers import LlamaTokenizer
import transformers

"""
Gradient-based Language Model Pruner (GBLM-Pruner) is a sparsity-centric pruning 
method for pretrained LLMs. GBLM-Pruner leverages the first-order term of the 
Taylor expansion, operating in a training-free manner by harnessing properly 
normalized gradients from a few calibration samples to determine the pruning 
metric.

We combine this pruning technique with 8 bit quantization.
"""

"""
On Disk Usage: 13GB
GPU VRAM Usage (Baseline after batch size 1 Perplexity on WikiText2):
Perplexity (WikiText2) (limit=):
Perplexity Time taken (batch size 1 on above limit):
MMLU (limit=):
MMLU Time taken (batch size 1 on above limit):
BBH (limit=):
BBH Time taken (batch size 1 on above limit):
"""


def get_model_and_tokenizer():
    """
    Pruned using https://github.com/VILA-Lab/GBLM-Pruner
    """
    
    model = "MBZUAI-LLM/GBLM-Pruner-LLaMA-2-7B-chat"

    tokenizer = LlamaTokenizer.from_pretrained(model)

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model, 
        device_map={"": 0},
        quantization_config=bnb_config,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.eval()

    return model, tokenizer