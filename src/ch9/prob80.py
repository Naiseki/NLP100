from transformers import AutoTokenizer

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "The movie was full of incomprehensibilities."

    tokens = tokenizer.tokenize(text)

    print(f"元の文: {text}")
    print(f"トークン列: {tokens}")

if __name__ == "__main__":
    main()
