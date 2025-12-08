from gensim.models import KeyedVectors
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

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
    
    # Ward法による階層型クラスタリング
    clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
    labels = clustering.fit_predict(X)
    
    # デンドログラムを生成
    linked = linkage(X, 'ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=valid_countries, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram for Country Vectors (Ward Method)')
    plt.savefig("output/ch6/58_dendrogram.png")
    print("デンドログラムを output/ch6/58_dendrogram.png に保存しました。")
    
    # クラスタごとに国名を出力（オプション）
    clusters = {i: [] for i in range(5)}
    for country, label in zip(valid_countries, labels):
        clusters[label].append(country)
    
    for cluster_id, countries_in_cluster in clusters.items():
        print(f"クラスタ {cluster_id}: {', '.join(countries_in_cluster)}")

if __name__ == "__main__":
    main()
