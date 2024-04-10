from utils import select_model_tokenizer_name
from metrics import perplexity, lm_eval_harness

def run_benchmark():

    model, tokenizer, name = select_model_tokenizer_name()

    while True:
        print("\nSelect a metric:")
        print("1. Perlexity using WikiText2.")
        print("2. MMLU - Massive Multitask Language Understanding.")
        print("3. BBH - Big-Bench Hard.")

        ch = int(input("Choice: "))

        if ch == 1:
            print(f'{name} LLaMA2 perplexity using WikiText2: ' + str(perplexity.get_wikitext2_perplexity(model, tokenizer)['mean_perplexity']))
        elif ch == 2:
            print(lm_eval_harness.do_lm_eval_task(model, tokenizer, 'mmlu', limit=20))
        elif ch == 3:
            print(lm_eval_harness.do_lm_eval_task(model, tokenizer, 'bbh', limit=5))
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    run_benchmark()