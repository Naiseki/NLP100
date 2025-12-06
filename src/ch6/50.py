from gensim.models import KeyedVectors

def main():
    # Google Newsベクトルをロード
    model = KeyedVectors.load_word2vec_format("input/GoogleNews-vectors-negative300.bin.gz", binary=True)
    vector = model["United_States"]
    print(vector)


if __name__ == "__main__":
    main()
