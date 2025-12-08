from gensim.models import KeyedVectors
import pandas as pd
from scipy.stats import spearmanr

def main():
    # Google Newsベクトルをロード
    model = KeyedVectors.load_word2vec_format("input/GoogleNews-vectors-negative300.bin.gz", binary=True)
    
    # wordsim353_combined.csvを読み込み
    df = pd.read_csv("input/wordsim353_combined.csv")
    
    # ベクトル類似度を計算
    vector_similarities = []
    human_similarities = []
    for word1, word2, human_sim in df.itertuples(index=False):
        vec_sim = model.similarity(word1, word2)
        vector_similarities.append(vec_sim)
        human_similarities.append(human_sim)
    
    # スピアマン相関係数を計算
    if vector_similarities and human_similarities:
        corr, _ = spearmanr(human_similarities, vector_similarities)
        print(f"スピアマン相関係数: {corr:.4f}")
    else:
        print("類似度の計算に失敗しました。")

if __name__ == "__main__":
    main()
