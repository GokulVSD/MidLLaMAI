from transformers import LlamaTokenizer
import transformers

"""
GPTQ is a post-training quantization technique where each row of the weight matrix 
is quantized independently to find a version of the weights that minimizes the error. 
These weights  are quantized to int4, but they're restored to fp16 on the fly during 
inference. This can save your memory-usage by 4x because the int4 weights are 
dequantized in a fused kernel rather than a GPU's global memory, and you can also 
expect a speedup in inference  because using a lower bitwidth takes less time to 
communicate.
"""

"""
On Disk Usage: 6.8GB
GPU VRAM Usage (Baseline after batch size 1 Perplexity on WikiText2):
Perplexity (WikiText2) (limit=):
MMLU (limit=):
MMLU Time taken (batch size 1 on above limit):
BBH (limit=):
BBH Time taken (batch size 1 on above limit):
"""


def get_model_and_tokenizer():
    """
    To save quantization time, we use a model that has already been quantized
    using the same approach as `quantizers/gptq.py`.
    """
    
    model = "TheBloke/Llama-2-13B-chat-GPTQ"

    tokenizer = LlamaTokenizer.from_pretrained(model)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model, 
        device_map={"": 0},
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.eval()

    return model, tokenizer