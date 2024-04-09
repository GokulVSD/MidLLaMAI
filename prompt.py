import transformers
from utils import select_model_tokenizer_name

model, tokenizer, name = select_model_tokenizer_name()

pipe = transformers.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1000)

while True:
    prompt = input(f"\nConverse with LLaMA2 {name}: ")
    result = pipe(f"<s>[INST] {prompt} [/INST]")[0]['generated_text']
    print(result[len(prompt):])