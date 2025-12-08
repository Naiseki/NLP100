from gensim.models import KeyedVectors
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def main():
    # Google Newsベクトルをロード
    model = KeyedVectors.load_word2vec_format("input/GoogleNews-vectors-negative300.bin.gz", binary=True)

    # questions-words.txtから国名を抽出
    countries = set()
    with open("input/questions-words.txt", "r", encoding="utf-8") as f:
        start = False
        for line in f:
            line = line.strip()
            if line == ": capital-common-countries":
                start = True
                continue
            elif line.startswith(":") and start:
                break
            if start and line:
                parts = line.split()
                if len(parts) == 4:
                    countries.add(parts[3])  # 国名（4列目）
    
    # 国名ベクトルを抽出
    country_vectors = []
    valid_countries = []
    for country in countries:
        try:
            vec = model[country]
            country_vectors.append(vec)
            valid_countries.append(country)
        except KeyError:
            pass
    
    if not country_vectors:
        print("国名のベクトルが見つかりませんでした。")
        return
    
    # numpy arrayに変換
    X = np.array(country_vectors)
    
    # t-SNEで2次元に削減
    tsne = TSNE(n_components=2, random_state=0, perplexity=20)
    X_tsne = tsne.fit_transform(X)
    
    # 可視化
    plt.figure(figsize=(10, 7))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    for i, country in enumerate(valid_countries):
        plt.annotate(country, (X_tsne[i, 0], X_tsne[i, 1]))
    plt.title('t-SNE Visualization of Country Vectors')
    plt.savefig("output/ch6/59_tsne.png")
    print("t-SNE可視化を output/ch6/59_tsne.png に保存しました。")

if __name__ == "__main__":
    main()
