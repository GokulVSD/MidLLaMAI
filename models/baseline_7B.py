from transformers import LlamaTokenizer
import transformers

token = "<HUGGINGFACE-TOKEN>"

"""
On Disk Usage: 
GPU VRAM Usage (Baseline after batch size 1 Perplexity on WikiText2): 
Perplexity (WikiText2) (limit=None): 95.74
Perplexity Time taken (batch size 1 on above limit): 1606.26s
MMLU (limit=50): 0.3428
MMLU Time taken (batch size 1 on above limit): 394.53s
BBH (limit=10):  0.3741
BBH Time taken (batch size 1 on above limit): 1736.50s
"""
def get_model_and_tokenizer():
    
    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = LlamaTokenizer.from_pretrained(model, token=token)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model, 
        device_map={"": 0},
        token=token
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.eval()

    return model, tokenizer
