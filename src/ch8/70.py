import numpy as np
import pandas as pd
import os
from gensim.models import KeyedVectors
from typing import Dict, Tuple, List

def build_embedding_matrix(
    pretrained_path: str, 
    limit: int
) -> Tuple[np.ndarray, List[str]]:
    """
    事前学習済みモデルから埋め込み行列と単語辞書を構築する。

    Args:
        pretrained_path (str): 事前学習済みバイナリファイルのパス。
        limit (int): 読み込む単語の最大数（メモリ節約のため）。

    Returns:
        Tuple[np.ndarray, List[str]]: 
            - 埋め込み行列 E (|V| x d_emb)
            - 単語リスト (words)
    """

    # 事前学習済みモデルの読み込み（Google News形式のバイナリ）
    model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True, limit=limit)
    
    d_emb = model.vector_size
    vocab_size = len(model.index_to_key) + 1  # <PAD>の分を追加

    # 全てをゼロで初期化
    embedding_matrix = np.zeros((vocab_size, d_emb), dtype=np.float32)

    words = ["<PAD>"]

    # 1行目以降に単語ベクトルを格納していく
    for i, word in enumerate(model.index_to_key):
        target_index = i + 1
        embedding_matrix[target_index] = model[word]
        words.append(word)

    return embedding_matrix, words

def main() -> None:
    file_path: str = "input/GoogleNews-vectors-negative300.bin.gz"
    output_dir: str = "output/ch8"
    
    # 語彙数制限
    limit_count: int = 500000

    try:
        print("行列と辞書を構築中...")
        E, words = build_embedding_matrix(file_path, limit=limit_count)

        # 保存先のディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)

        # 行列の保存
        np.save(os.path.join(output_dir, "E.npy"), E)

        df = pd.DataFrame(words, columns=["word"])
        df.to_csv(os.path.join(output_dir, "words.csv"), index=True, lineterminator="\n", header=False)

        print(f"行列と辞書を {output_dir}/ に保存しました。")
        
        # 結果の確認
        print(f"行列のshape: {E.shape}")  # (|V|, d_emb)
    except FileNotFoundError:
        print(f"エラー: {file_path} が見つかりません。")

if __name__ == "__main__":
    main()
