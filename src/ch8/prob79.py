import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from prob78 import SST2Dataset
from prob78 import collate_fn


# RNNモデルの定義
class RNNModel(nn.Module):
    """RNNを用いた感情分類モデル。"""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, pretrained_weight: torch.Tensor):
        super().__init__()
        # 単語埋め込み層（ファインチューニングを有効に設定）
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_weight, 
            freeze=False, 
            padding_idx=0
        )
        # RNN層 (batch_first=True で (batch, seq, feature) の入力を受け取る)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        # 出力層（RNNの最後の隠れ状態を入力にする）
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # RNNの出力: 
        # out: 各時刻の隠れ状態 (batch, seq_len, hidden_dim)
        # h_n: 最終時刻の隠れ状態 (1, batch, hidden_dim)
        out, h_n = self.rnn(embedded)
        
        # 最終時刻の隠れ状態を使って二値分類
        # h_n は (num_layers, batch, hidden_dim) なので、squeezeして (batch, hidden_dim) にする
        logits = self.linear(h_n.squeeze(0))
        return self.sigmoid(logits)


# 評価関数の定義
def calculate_accuracy(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            correct += (preds.view_as(labels) == labels).sum().item()
            total += labels.size(0)
    return correct / total

# 学習メイン処理
def train() -> None:
    embed_npy = "output/ch8/E.npy"
    train_pkl = "output/ch8/train.pkl"
    dev_pkl = "output/ch8/dev.pkl"
    
    # ハイパーパラメータ
    hidden_dim = 128
    batch_size = 64
    epochs = 10
    lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("リソース読み込み中...")
    E = torch.from_numpy(np.load(embed_npy)).float()
    train_df = pd.DataFrame(pd.read_pickle(train_pkl))
    dev_df = pd.DataFrame(pd.read_pickle(dev_pkl))
    
    train_loader = DataLoader(SST2Dataset(train_df), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(SST2Dataset(dev_df), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = RNNModel(E.shape[0], E.shape[1], hidden_dim, E).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"RNNでの学習を開始します（デバイス: {device}）")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].to(device).view_as(model(inputs)) # 安全に形状合わせ

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        dev_acc = calculate_accuracy(model, dev_loader, device)
        print(f"エポック {epoch}: 損失 = {running_loss/len(train_loader):.4f}, 開発正解率 = {dev_acc:.4f}")

if __name__ == "__main__":
    train()
