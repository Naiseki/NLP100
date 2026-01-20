import pandas as pd
import torch
import os
from typing import List, Dict, Any

def load_word2id(path: str) -> Dict[str, int]:
    df = pd.read_csv(path, names=["id", "word"], index_col="word")
    return df["id"].to_dict()

def process_dataset(file_path: str, word2id: Dict[str, int]) -> List[Dict[str, Any]]:
    # SST-2は通常 header=0 (sentence, label)
    df = pd.read_csv(file_path, sep="\t")
    processed_data = []

    for _, row in df.iterrows():
        text = row["sentence"]
        label = float(row["label"])
        
        # 単語埋め込みの語彙に含まれる単語のみIDに変換
        input_ids = [word2id[word] for word in text.split() if word in word2id]
        
        # 空のトークン列となる事例は削除
        if len(input_ids) > 0:
            processed_data.append({
                'text': text,
                'label': torch.tensor([label]),
                'input_ids': torch.tensor(input_ids)
            })
            
    return processed_data

def main() -> None:
    words_path = "output/ch8/words.csv"
    train_path = "input/SST-2/train.tsv"
    dev_path = "input/SST-2/dev.tsv"

    if not os.path.exists(words_path):
        print("エラー: words.csv が見つかりません。")
        return

    word2id = load_word2id(words_path)
    
    print("データセットを処理中...")
    train_data = process_dataset(train_path, word2id)
    dev_data = process_dataset(dev_path, word2id)

    print(f"訓練セット: {len(train_data)} 件")
    print(f"検証セット: {len(dev_data)} 件")

    # 保存先ディレクトリを作成して、辞書データと処理済みデータを保存
    os.makedirs("output/ch8", exist_ok=True)

    pd.to_pickle(train_data, "output/ch8/train.pkl")
    pd.to_pickle(dev_data, "output/ch8/dev.pkl")

    print("処理済みデータを output/ch8/train.pkl と output/ch8/dev.pkl に保存しました。")

    # 例を表示
    if train_data:
        print("\n訓練セットのサンプル:")
        print(train_data[0])

if __name__ == "__main__":
    main()
