import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from prob72 import LogisticRegressionCBoW
from prob73 import SST2Dataset

def calculate_accuracy(model: torch.nn.Module, dataloader: DataLoader, device: str) -> float:
    """
    データローダー内のデータに対する正解率を計算する。
    """
    model.eval()  # 評価モードに設定
    correct = 0
    total = 0
    
    # 勾配の計算を停止（メモリ節約と高速化）
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # 推論
            outputs = model(input_ids)
            
            # 出力は確率（0~1）なので、0.5を閾値として二値分類
            # outputsの形状が [batch, 1] なら、予測値 preds もそれに合わせる
            preds = (outputs > 0.5).float()
            
            # 正解数をカウント（形状を揃えて比較）
            correct += (preds.view_as(labels) == labels).sum().item()
            total += labels.size(0)
            
    return correct / total

def evaluate():
    # パス設定
    dev_pkl = "output/ch8/dev.pkl" # あらかじめ作成してある想定
    embed_npy = "output/ch8/E.npy"
    model_path = "output/ch8/w_linear.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("リソースを読み込み中...")
    # 1. 埋め込み行列の読み込みとモデルの初期化
    E = torch.from_numpy(np.load(embed_npy)).float()
    model = LogisticRegressionCBoW(pretrained_weight=E)
    
    # 2. 学習済み重み（線形層のみ）をロード
    model.linear.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # 3. 開発セットの読み込み
    dev_df = pd.DataFrame(pd.read_pickle(dev_pkl))
    dev_dataset = SST2Dataset(dev_df, max_len=20)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
    
    # 4. 正解率の計算
    print("開発セットで評価中...")
    acc = calculate_accuracy(model, dev_loader, device)
    
    print(f"開発セットの正解率: {acc:.4f}")

if __name__ == "__main__":
    evaluate()
