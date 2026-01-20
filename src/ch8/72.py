import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class LogisticRegressionCBoW(nn.Module):
    """
    単語埋め込みの平均ベクトルを入力とし、ポジティブ/ネガティブを分類する
    ロジスティック回帰モデル。
    """
    def __init__(self, pretrained_weight: torch.Tensor) -> None:
        super(LogisticRegressionCBoW, self).__init__()
        
        # 保存された行列から埋め込み層を初期化
        # padding_idx=0 を指定し、ID:0 (<PAD>) をゼロベクトルとして扱う
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_weight, 
            freeze=False, 
            padding_idx=0
        )
        
        # 埋め込み次元数を取得し、線形層を定義
        embedding_dim = pretrained_weight.size(1)
        self.linear = nn.Linear(embedding_dim, 1)
        
        # 二値分類のためシグモイド関数を適用
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理。
        Args:
            input_ids (torch.Tensor): 単語IDのシーケンス (batch_size, seq_len)
        Returns:
            torch.Tensor: 各サンプルのポジティブ確率 (batch_size, 1)
        """
        # 1. 単語IDを埋め込みベクトルに変換: (batch_size, seq_len, d_emb)
        embedded = self.embedding(input_ids)
        
        # 2. 文章内の全単語ベクトルの平均を計算（CBoW表現）: (batch_size, d_emb)
        # dim=1 (seq_len方向) で平均をとる
        pooled = torch.mean(embedded, dim=1)
        
        # 3. 線形層を通過させ、シグモイドで確率に変換
        logits = self.linear(pooled)
        probs = self.sigmoid(logits)
        
        return probs

def load_embedding(npy_path: str) -> torch.Tensor:
    """
    保存済みの埋め込み行列(Numpy)と単語辞書(CSV)を読み込む。
    """
    # 埋め込み行列の読み込みとTensor変換
    E_np = np.load(npy_path)
    E_tensor = torch.from_numpy(E_np).float()
    
    return E_tensor

def main() -> None:
    # ファイルパスの指定
    npy_path: str = "output/ch8/E.npy"

    try:
        # 1. データのロード
        print("リソースを読み込み中...")
        E = load_embedding(npy_path)
        print(f"埋め込み行列を読み込みました。形状: {E.shape}")

        # 2. モデルの構築
        model = LogisticRegressionCBoW(pretrained_weight=E)
        print("モデルの初期化が完了しました。")

        # 3. テスト用の入力データ作成 (batch_size=2, seq_len=4)
        # 例として、words.csvのIDに基づいた単語列を想定
        # [5, 4, 3, 0] は "is that for <PAD>" に相当
        sample_ids = torch.tensor([
            [5, 4, 3, 0],
            [2, 3, 4, 5]
        ], dtype=torch.long)

        # 4. 推論の実行
        model.eval()
        with torch.no_grad():
            outputs = model(sample_ids)
        
        print("\n--- 推論結果 (ポジティブである確率) ---")
        for i, prob in enumerate(outputs):
            print(f"サンプル {i+1}: {prob.item():.4f}")

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。パスを確認してください。\n{e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
