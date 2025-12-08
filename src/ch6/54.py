from gensim.models import KeyedVectors
import pandas as pd

def main():
    # Google Newsベクトルをロード
    model = KeyedVectors.load_word2vec_format("input/GoogleNews-vectors-negative300.bin.gz", binary=True)

    words_lists = []
    start_reading = False
    with open("input/questions-words.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(": capital-common-countries"):
                start_reading = True
                continue
            if line.startswith(":") and start_reading:
                break  # 他のセクションに入ったら終了
            if start_reading:
                # 各行の単語をリストとして追加
                words_lists.append(line.split())
    
    # vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算した辞書
    word_vector_dict = {f"{words[1]}-{words[0]}+{words[2]}": model[words[1]] - model[words[0]] + model[words[2]] for words in words_lists}

    for words, vec in word_vector_dict.items():
        try:
            # 最も類似した単語を取得
            similar_word, similarity = model.most_similar(positive=[vec], topn=1)[0]
            print(f"{words} -> 単語: {similar_word} 類似度: {similarity}")
        except KeyError as e:
            print(f"{words} -> エラー: {e}")

if __name__ == "__main__":
    main()
