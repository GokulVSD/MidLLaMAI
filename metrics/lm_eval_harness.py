import lm_eval
from lm_eval.models.huggingface import HFLM

def do_lm_eval_task(model, tokenizer, task, limit=None):

    model_wrapper = HFLM(pretrained=model, tokenizer=tokenizer)

    results = lm_eval.evaluator.simple_evaluate(
        model=model_wrapper, 
        tasks=[task], 
        log_samples=False, 
        limit=limit, 
        batch_size=1, 
        max_batch_size=1
    )
    
    return results['results'][task]