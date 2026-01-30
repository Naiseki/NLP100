import torch
from torch.utils.data import TensorDataset, DataLoader

def main():
    # 1. 保存したデータの読み込み
    # 85番で保存した .pth ファイルを指定
    data_path = "output/ch9/train_data.pth"
    try:
        data = torch.load(data_path)
    except FileNotFoundError:
        print(f"ファイル{data_path}が見つかりません。先に85番の保存処理を実行してください。")
        return

    # 2. 冒頭4事例を抽出
    # dataは辞書形式 {'input_ids': tensor, 'attention_mask': tensor, 'labels': tensor}
    batch_size = 4
    subset_data = {key: val[:batch_size] for key, val in data.items()}

    # 3. TensorDatasetの作成
    # テンソルをまとめて一つのデータセットオブジェクトにする
    dataset = TensorDataset(
        subset_data['input_ids'],
        subset_data['attention_mask'],
        subset_data['labels']
    )

    # 4. DataLoaderの作成
    # これが「ミニバッチを構成する」ための標準的なインターフェース
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 5. ミニバッチの確認
    # loaderから1回分取り出す
    batch = next(iter(loader))
    input_ids, attention_mask, labels = batch

    print(f"ミニバッチの構成 (Batch Size: {batch_size})")
    print("-" * 40)
    print(f"input_ids shape:      {input_ids.shape}")
    print(f"attention_mask shape: {attention_mask.shape}")
    print(f"labels shape:         {labels.shape}")
    
    print("\n--- 最初の事例のトークンID列 (一部) ---")
    print(input_ids[0][:20]) # 冒頭20トークン分を表示

if __name__ == "__main__":
    main()
