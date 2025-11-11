def ngram(target, n):
    return [target[i:i+n] for i in range(len(target) - n + 1)]

def main():
    text = "I am an NLPer"
    
    char_trigrams = ngram(text, 3)
    print("文字tri-gram:", char_trigrams)

    words = text.split()
    word_bigrams = ngram(words, 2)
    print("単語bi-gram:", word_bigrams)

if __name__ == "__main__":
    main()
