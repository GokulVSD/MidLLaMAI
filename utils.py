from models import four_bit_bnb, four_bit_gptq

def select_model_tokenizer_name():
    while True:
        print("Select a model:")
        print("1. 4-bit BitsAndBytes quantized LLaMA2.")
        print("2. 4-bit GPTQ quantized LLaMA2.")

        ch = int(input("Choice: "))
        if ch == 1:
            return *four_bit_bnb.get_model_and_tokenizer(), "4-bit BitsAndBytes"
        elif ch == 2:
            return *four_bit_gptq.get_model_and_tokenizer(), '4-bit GPTQ'
        else:
            print("Invalid choice, try again.")

    