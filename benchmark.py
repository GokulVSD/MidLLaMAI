from utils import select_model_tokenizer_name
from metrics import perplexity, lm_eval_harness
import time
import click

PERPLEXITY_LIMIT = None
MMLU_LIMIT_RATIO = 50
BBH_LIMIT_RATIO = 10

@click.command()
@click.option("-b", "--benchmark", type=click.Choice(["wiki", "mmlu", "bbh"]), help="The benchmark to run", required=True)
@click.option("-m", "--model-name", type=click.Choice(["baseline", "bnb", "gblm", "gptq-7b", "gptq-13b", "awq"]), help="The model to benchmark", required=True)
def run_benchmark(benchmark, model_name):
    model, tokenizer, name = select_model_tokenizer_name(model_name)

    begin = time.time()

    if benchmark == "wiki":
        print(f'{name} Perplexity using WikiText2: ', perplexity.get_wikitext2_perplexity(model, tokenizer, limit=PERPLEXITY_LIMIT))
    elif benchmark == "mmlu":
        print(f'{name} MMLU Score:')
        print(lm_eval_harness.do_lm_eval_task(model, tokenizer, 'mmlu', limit=MMLU_LIMIT_RATIO))
    elif benchmark == "bbh":
        print(f'{name} BBH Score:')
        print(lm_eval_harness.do_lm_eval_task(model, tokenizer, 'bbh', limit=BBH_LIMIT_RATIO))
    else:
        print("Invalid choice.")
    
    end = time.time()
    print("Took time: " + str(end - begin) + " seconds")

if __name__ == "__main__":
    run_benchmark()