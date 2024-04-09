from utils import select_model_tokenizer_name
from metrics import perplexity

model, tokenizer, name = select_model_tokenizer_name()

def run_benchmark():
    while True:
        print("\nSelect a metric:")
        print("1. Perlexity using WikiText2.")

        ch = int(input("Choice: "))

        if ch == 1:
            print(f'{name} LLaMA2 perplexity using WikiText2: {perplexity.get_wikitext2_perplexity(model, tokenizer)['mean_perplexity']}')
        else:
            print("Invalid choice, try again.")

run_benchmark()