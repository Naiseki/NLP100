import pandas as pd
import torch
import pickle
from transformers import AutoTokenizer
import os

def process_and_save(file_path, tokenizer, save_name):
    df = pd.read_csv(file_path, sep='\t')
    texts = df['sentence'].tolist()
    labels = df['label'].tolist()
    
    # トークン化
    encoded_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # データをひとまとめにする
    data_to_save = {
        'input_ids': encoded_inputs['input_ids'],
        'attention_mask': encoded_inputs['attention_mask'],
        'labels': torch.tensor(labels)
    }
    
    # 保存先ディレクトリ作成
    os.makedirs("output", exist_ok=True)
    save_path = f"output/ch9/{save_name}.pth"
    
    # PyTorchの保存関数を使用
    torch.save(data_to_save, save_path)
    print(f"Saved: {save_path}")

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 訓練・開発セットそれぞれ保存
    process_and_save("input/SST-2/train.tsv", tokenizer, "train_data")
    process_and_save("input/SST-2/dev.tsv", tokenizer, "dev_data")

if __name__ == "__main__":
    main()
