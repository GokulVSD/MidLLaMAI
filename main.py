from transformers import AutoTokenizer
import transformers

from torch import bfloat16

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model, 
    quantization_config=bnb_config,
    low_cpu_mem_usage=True
)

model.config.use_cache = False
model.config.pretraining_tp = 1

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

sequences = pipeline(
    'Hello!',
    do_sample=True
)
print(sequences[0].get("generated_text"))