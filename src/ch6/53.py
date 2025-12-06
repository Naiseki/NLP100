from gensim.models import KeyedVectors

def main():
    # Google Newsベクトルをロード
    model = KeyedVectors.load_word2vec_format("input/GoogleNews-vectors-negative300.bin.gz", binary=True)
    vector = model["United_States"]

    # Spain - Madrid + Athens
    spain = model["Spain"]
    madrid = model["Madrid"]
    athens = model["Athens"]
    result_vector = spain - madrid + athens
    # 結果ベクトルに最も類似した10語と類似度を取得
    similar_words = model.most_similar(positive=[result_vector], topn=10)
    for word, similarity in similar_words:
        print(f"{word}: {similarity}")


if __name__ == "__main__":
    main()
