from gensim.models import KeyedVectors
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
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
    
    # k-meansクラスタリング (k=5)
    kmeans = KMeans(n_clusters=5, random_state=0)
    labels = kmeans.fit_predict(X)
    
    # クラスタごとに国名を出力
    clusters = {i: [] for i in range(5)}
    for country, label in zip(valid_countries, labels):
        clusters[label].append(country)
    
    for cluster_id, countries_in_cluster in clusters.items():
        print(f"クラスタ {cluster_id}: {', '.join(countries_in_cluster)}")

if __name__ == "__main__":
    main()
