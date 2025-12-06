from gensim.models import KeyedVectors

def main():
    # Google Newsベクトルをロード
    model = KeyedVectors.load_word2vec_format("input/GoogleNews-vectors-negative300.bin.gz", binary=True)
    vector = model["United_States"]

    # "United_States"と"U.S."のコサイン類似度を計算
    similarity = model.similarity("United_States", "U.S.")
    print(f"United_StatesとU.S.のコサイン類似度: {similarity}")


if __name__ == "__main__":
    main()
