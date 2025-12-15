import pandas as pd
from collections import Counter

def count_labels(file_path):
    df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
    labels = df["label"].astype(int)
    return labels.value_counts().to_dict()


def main():
    train_path = "input/SST-2/train.tsv"
    dev_path = "input/SST-2/dev.tsv"
    
    train_counts = count_labels(train_path)
    dev_counts = count_labels(dev_path)
    
    print("訓練データ:")
    print(f"ネガティブ (0): {train_counts.get(0, 0)}")
    print(f"ポジティブ (1): {train_counts.get(1, 0)}")
    
    print("検証データ:")
    print(f"ネガティブ (0): {dev_counts.get(0, 0)}")
    print(f"ポジティブ (1): {dev_counts.get(1, 0)}")

if __name__ == "__main__":
    main()
