import random

def typoglycemia(text):
    words = text.split()
    for i in range(len(words)):
        if len(words[i]) > 4:
            chars = list(words[i])
            first = chars.pop(0)
            last = chars.pop()
            random.shuffle(chars)
            words[i] = first + "".join(chars) + last
    return " ".join(words)

def main():
    text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
    print(typoglycemia(text))

if __name__ == "__main__":
    main()
