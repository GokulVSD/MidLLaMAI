from models import four_bit_7B_bnb, four_bit_7B_gptq, four_bit_13B_gptq, four_bit_13B_awq, eight_bit_7B_gblm_pruned, baseline_7B

def select_model_tokenizer_name(model_name):
    if model_name == "bnb":
        return *four_bit_7B_bnb.get_model_and_tokenizer(), "4-bit 7B BitsAndBytes"
    elif model_name == "gptq-7b":
        return *four_bit_7B_gptq.get_model_and_tokenizer(), '4-bit 7B GPTQ'
    elif model_name == "gptq-13b":
        return *four_bit_13B_gptq.get_model_and_tokenizer(), '4-bit 13B GPTQ'
    elif model_name == "awq":
        return *four_bit_13B_awq.get_model_and_tokenizer(), '4-bit 13B AWQ'
    elif model_name == "gblm":
        return *eight_bit_7B_gblm_pruned.get_model_and_tokenizer(), '8-bit 7B GBLM Pruned'
    elif model_name == "baseline":
        return *baseline_7B.get_model_and_tokenizer(), '7B Baseline'
    else:
        print("Invalid choice.")

    