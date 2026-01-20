import os
import re
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from prob72 import LogisticRegressionCBoW


class SST2Dataset(torch.utils.data.Dataset):
    """SST2用のカスタムDataset"""
    def __init__(self, dataset_df: pd.DataFrame, max_len: int = 20):
        self.labels = dataset_df["label"].values
        self.features = dataset_df["input_ids"].values
        self.max_len = max_len # 最大の長さを指定

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.features[idx]
        
        # リストをTensorに変換
        if isinstance(ids, list):
            ids = torch.tensor(ids, dtype=torch.long)
        
        # パディング処理：指定したmax_lenに合わせる
        curr_len = ids.size(0)
        if curr_len < self.max_len:
            # 足りない分を0で埋める
            padding = torch.zeros(self.max_len - curr_len, dtype=torch.long)
            ids = torch.cat([ids, padding])
        else:
            # 長すぎる場合は切り捨てる
            ids = ids[:self.max_len]
            
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)
        return ids, label


def load_resources(embed_npy: str, train_pkl: str) -> tuple[torch.Tensor, pd.DataFrame]:
    """埋め込みとデータセット読み込み"""
    E_np = np.load(embed_npy)
    E_tensor = torch.from_numpy(E_np).float()
    # 読み込んだデータ（リスト）を DataFrame に変換する
    data = pd.read_pickle(train_pkl)
    if isinstance(data, list):
        train_df = pd.DataFrame(data)
    else:
        train_df = data
    return E_tensor, train_df


def train(
    embed_npy: str = "output/ch8/E.npy",
    train_pkl: str = "output/ch8/train.pkl",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.01,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("読み込み: 埋め込みとデータセット")
    E, train_df = load_resources(embed_npy, train_pkl)

    print(f"埋め込み行列の形状: {tuple(E.shape)} (語彙数: {E.shape[0]}, 次元数: {E.shape[1]})")

    dataset = SST2Dataset(train_df)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LogisticRegressionCBoW(pretrained_weight=E)

    # 埋め込みを固定 
    model.embedding.weight.requires_grad = False
    model.to(device)

    optimizer = optim.Adam(list(model.linear.parameters()), lr=lr)
    criterion = nn.BCELoss()

    print(f"訓練開始: device={device}, epochs={epochs}, batch_size={batch_size}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (input_ids, label) in enumerate(dataloader, 1):
            input_ids = input_ids.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)

            # view_asでoutputsとshapeを合わせる
            label = label.to(device).view_as(outputs)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                avg = running_loss / 100
                print(f"Epoch {epoch} batch {i:4d} loss avg = {avg:.6f}")
                running_loss = 0.0

        # epoch summary
        print(f"Epoch {epoch} 終了.")

    print("訓練完了。線形層の重みを保存します: output/ch8/w_linear.pt")
    os.makedirs("output/ch8", exist_ok=True)
    torch.save(model.linear.state_dict(), "output/ch8/w_linear.pt")


if __name__ == "__main__":
    train()
