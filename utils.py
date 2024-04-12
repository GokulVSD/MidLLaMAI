from models import four_bit_7B_bnb, four_bit_7B_gptq, four_bit_13B_gptq, four_bit_13B_awq, eight_bit_7B_gblm_pruned

def select_model_tokenizer_name():
    while True:
        print("Select a model:")
        print("1. 4-bit 7B BitsAndBytes quantized LLaMA2.")
        print("2. 4-bit 7B GPTQ quantized LLaMA2.")
        print("3. 4-bit 13B GPTQ quantized LLaMA2.")
        print("4. 4-bit 13B AWQ quantized LLaMA2.")
        print("5. 8-bit 7B GBLM Pruned LLaMA2.")

        ch = int(input("Choice: "))
        if ch == 1:
            return *four_bit_7B_bnb.get_model_and_tokenizer(), "4-bit 7B BitsAndBytes"
        elif ch == 2:
            return *four_bit_7B_gptq.get_model_and_tokenizer(), '4-bit 7B GPTQ'
        elif ch == 3:
            return *four_bit_13B_gptq.get_model_and_tokenizer(), '4-bit 13B GPTQ'
        elif ch == 4:
            return *four_bit_13B_awq.get_model_and_tokenizer(), '4-bit 13B AWQ'
        elif ch == 5:
            return *eight_bit_7B_gblm_pruned.get_model_and_tokenizer(), '8-bit 7B GBLM Pruned'
        else:
            print("Invalid choice, try again.")

    