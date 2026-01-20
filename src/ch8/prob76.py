import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# 前の問題で作成したモデルをインポート
from prob72 import LogisticRegressionCBoW

class SST2Dataset(Dataset):
    """SST2用のDataset。input_idsとlabelを辞書形式で返す。"""
    def __init__(self, dataset_df: pd.DataFrame):
        self.labels = dataset_df["label"].values
        self.features = dataset_df["input_ids"].values

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # torch.as_tensor を使って、不要なコピーや警告を避ける
        return {
            "input_ids": torch.as_tensor(self.features[idx], dtype=torch.long),
            "label": torch.as_tensor(self.labels[idx], dtype=torch.float32)
        }

def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """ミニバッチ作成時にパディングとソートを行う関数。"""
    batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
    
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["label"] for item in batch]
    
    # 動的パディング
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    
    # ここでは単純に重ねるだけにする（形状は [batch_size] もしくは [batch_size, 1]）
    labels_tensor = torch.stack(labels_list)
    
    return {"input_ids": input_ids_padded, "label": labels_tensor}

def calculate_accuracy(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    """正解率を計算する。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            # 比較のために形状を合わせる
            preds = (outputs > 0.5).float()
            correct += (preds.view_as(labels) == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train() -> None:
    embed_npy = "output/ch8/E.npy"
    train_pkl = "output/ch8/train.pkl"
    dev_pkl = "output/ch8/dev.pkl"
    
    batch_size = 64
    epochs = 10
    lr = 0.01
    device = "cpu"

    print("リソースの読み込みを開始します...")
    E = torch.from_numpy(np.load(embed_npy)).float()
    
    train_df = pd.DataFrame(pd.read_pickle(train_pkl))
    dev_df = pd.DataFrame(pd.read_pickle(dev_pkl))
    
    train_dataset = SST2Dataset(train_df)
    dev_dataset = SST2Dataset(dev_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = LogisticRegressionCBoW(pretrained_weight=E)
    model.embedding.weight.requires_grad = False
    model.to(device)

    optimizer = optim.Adam(model.linear.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"訓練を開始します（デバイス: {device}）")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch["input_ids"].to(device)
            # 一旦転送
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # ここで outputs ([64, 1]) と labels の形状を確実に合わせる
            labels = labels.view_as(outputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        train_acc = calculate_accuracy(model, train_loader, device)
        dev_acc = calculate_accuracy(model, dev_loader, device)
        print(f"エポック {epoch}: 損失 = {running_loss/len(train_loader):.4f}, 訓練正解率 = {train_acc:.4f}, 開発正解率 = {dev_acc:.4f}")

    print("訓練が完了しました。")

if __name__ == "__main__":
    train()
