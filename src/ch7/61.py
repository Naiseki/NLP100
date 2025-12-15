import pandas as pd
from collections import Counter

def text_to_bow(text: str) -> dict:
    tokens = text.split()
    bow = {}
    for t in tokens:
        bow[t] = bow.get(t, 0) + 1
    return bow

def load_dataset_bow(file_path: str) -> list:
    # ヘッダ有無に対応して読み込む（文字列として扱う）
    df = pd.read_csv(file_path, sep="\t", encoding="utf-8", dtype={"sentence": str, "label": str})
    dataset = []
    for text, label in df.itertuples(index=False):
        feature = text_to_bow(text)
        dataset.append({"text": text, "label": label, "feature": feature})
    return dataset

def main():
    train_path = "input/SST-2/train.tsv"
    dev_path = "input/SST-2/dev.tsv"

    # BoW に変換してリスト化
    train_list = load_dataset_bow(train_path)
    dev_list = load_dataset_bow(dev_path)
    print(f"\n学習データ件数: {len(train_list)}, 検証データ件数: {len(dev_list)}")

    # 学習データの最初の事例を表示して確認
    if train_list:
        first = train_list[0]
        print("\n学習データの最初の事例:")
        print(f"テキスト: {first['text']}")
        print(f"ラベル: {first['label']}")
        print(f"特徴ベクトル: {first['feature']}")

    with open("output/ch7/61_train_bow.txt", "w", encoding="utf-8") as f:
        for item in train_list:
            f.write(f"{item}\n")
    with open("output/ch7/61_dev_bow.txt", "w", encoding="utf-8") as f:
        for item in dev_list:
            f.write(f"{item}\n")

if __name__ == "__main__":
    main()
