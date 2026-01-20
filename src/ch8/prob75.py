import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch: list[dict[str, any]]) -> dict[str, torch.Tensor]:
    """
    複数の事例をまとめ、パディングと並び替えを行う関数。
    
    Args:
        batch (list[dict[str, any]]): Datasetから得られる辞書のリスト。
        
    Returns:
        dict[str, torch.Tensor]: パディング済みinput_idsとlabelsを含む辞書。
    """
    
    # 1. トークン列の長さ（input_idsの要素数）が長い順に事例をソート
    # reverse=True で降順に並び替える
    batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
    
    # 2. 各事例から input_ids と label を抽出してリスト化
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["label"] for item in batch]
    
    # 3. input_ids のパディング処理
    # batch_first=True で (batch_size, max_len) の形状にする
    # padding_value=0 で 0番のトークンIDでパディングする
    input_ids_padded = pad_sequence(
        input_ids_list, 
        batch_first=True, 
        padding_value=0
    )
    
    # 4. label を一つのテンソルにまとめる (batch_size, 1)
    labels_tensor = torch.stack(labels_list)
    
    return {
        "input_ids": input_ids_padded,
        "label": labels_tensor
    }

def main() -> None:
    # テスト用データ（問題文の例）
    test_batch = [
        {
            'text': 'hide new secretions from the parental units',
            'label': torch.tensor([0.]),
            'input_ids': torch.tensor([5785, 66, 113845, 18, 12, 15095, 1594])
        },
        {
            'text': 'contains no wit , only labored gags',
            'label': torch.tensor([0.]),
            'input_ids': torch.tensor([3475, 87, 15888, 90, 27695, 42637])
        },
        {
            'text': 'that loves its characters and communicates something rather beautiful about human nature',
            'label': torch.tensor([1.]),
            'input_ids': torch.tensor([4, 5053, 45, 3305, 31647, 348, 904, 2815, 47, 1276, 1964])
        },
        {
            'text': 'remains utterly satisfied to remain the same throughout',
            'label': torch.tensor([0.]),
            'input_ids': torch.tensor([987, 14528, 4941, 873, 12, 208, 898])
        }
    ]

    # 関数の実行
    result = collate_fn(test_batch)

    # 結果の表示
    print("--- input_ids ---")
    print(result["input_ids"])
    print(f"形状: {result['input_ids'].shape}")

    print("\n--- label ---")
    print(result["label"])
    print(f"形状: {result['label'].shape}")

if __name__ == "__main__":
    main()
