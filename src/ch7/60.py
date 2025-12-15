import csv
from collections import Counter

def main():
    def count_labels(file_path):
        labels = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # ヘッダーをスキップ
            for row in reader:
                if len(row) >= 2:
                    # 2列目のラベルを取得
                    labels.append(int(row[1])) 
                else:
                    raise ValueError("行の列数が不足しています。")
        return Counter(labels)
    
    train_path = "input/SST-2/train.tsv"
    dev_path = "input/SST-2/dev.tsv"
    
    train_counts = count_labels(train_path)
    dev_counts = count_labels(dev_path)
    
    print("訓練データ:")
    print(f"ネガティブ (0): {train_counts[0]}")
    print(f"ポジティブ (1): {train_counts[1]}")
    
    print("検証データ:")
    print(f"ネガティブ (0): {dev_counts[0]}")
    print(f"ポジティブ (1): {dev_counts[1]}")

if __name__ == "__main__":
    main()
