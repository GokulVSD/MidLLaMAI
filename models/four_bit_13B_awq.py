from transformers import LlamaTokenizer
import transformers

"""
Activation-aware Weight Quantization (AWQ) doesn't quantize all the weights in a
model, and instead, it preserves a small percentage of weights that are important 
for LLM performance. This significantly reduces quantization loss such that you 
can run models in 4-bit precision without experiencing any performance degradation.
"""

"""
On Disk Usage: 6.8GB
GPU VRAM Usage (Baseline after batch size 1 Perplexity on WikiText2): 7772MB
Perplexity (WikiText2) (limit=None): 61.01
Perplexity Time taken (batch size 1 on above limit): 2790.1s
MMLU (limit=50): 0.529
MMLU Time taken (batch size 1 on above limit): 634.73s
BBH (limit=10): 0.0 (??? Weird)
BBH Time taken (batch size 1 on above limit): 12494.52s
"""


def get_model_and_tokenizer():
    """
    To save quantization time, we use a model that has already been quantized
    using the same approach as `quantizers/awq.py`.
    """
    
    model = "TheBloke/Llama-2-13B-chat-AWQ"

    tokenizer = LlamaTokenizer.from_pretrained(model)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model, 
        device_map={"": 0},
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.eval()

    return model, tokenizer