from utils import select_model_tokenizer_name
from metrics import perplexity, lm_eval_harness
import time

MMLU_LIMIT_RATIO = 0.05
BBH_LIMIT_RATIO = 0.05


def run_benchmark():

    model, tokenizer, name = select_model_tokenizer_name()

    while True:
        print("\nSelect a metric:")
        print("1. Perlexity using WikiText2.")
        print("2. MMLU - Massive Multitask Language Understanding.")
        print("3. BBH - Big-Bench Hard.")

        ch = int(input("Choice: "))

        begin = time.time()

        if ch == 1:
            print(f'{name} Perplexity using WikiText2: ' + str(perplexity.get_wikitext2_perplexity(model, tokenizer)['mean_perplexity']))
        elif ch == 2:
            print(f'{name} MMLU Score:')
            print(lm_eval_harness.do_lm_eval_task(model, tokenizer, 'mmlu', limit=MMLU_LIMIT_RATIO))
        elif ch == 3:
            print(f'{name} BBH Score:')
            print(lm_eval_harness.do_lm_eval_task(model, tokenizer, 'bbh', limit=BBH_LIMIT_RATIO))
        else:
            print("Invalid choice, try again.")
        
        end = time.time()
        print("Took time: " + str(end - begin) + " seconds")

if __name__ == "__main__":
    run_benchmark()