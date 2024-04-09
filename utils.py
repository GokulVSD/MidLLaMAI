from models import four_bit_quantized

def select_model_tokenizer_name():
    while True:
        print("Select a model:")
        print("1. 4-bit quantized LLaMA2.")

        ch = int(input("Choice: "))
        if ch == 1:
            return *four_bit_quantized.get_model_and_tokenizer(), "4-bit quantized"
        else:
            print("Invalid choice, try again.")

    