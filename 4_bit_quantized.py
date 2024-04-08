from transformers import AutoTokenizer
import transformers
from torch import float16

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

pipe = transformers.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1000)

while True:
    prompt = input("\nConverse with LLaMA2: ")
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])