"""
90. 次単語予測
GPT型（Transformerのデコーダ型）の事前学習済みモデルを利用して、
"The movie was full of"に続くトークンとして適切なもの上位10個と、
その確率（尤度）を求める。
また、プロンプトがどのようなトークン列に変換されたか確認する。
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def predict_next_token(prompt, top_k=10):
    """
    GPT-2モデルを使用して、プロンプトに続く次のトークンを予測する。
    
    Args:
        prompt (str): 入力プロンプト
        top_k (int): 上位何個のトークンを表示するか
    
    Returns:
        tuple: (トークンID列, トップkのトークンと確率のリスト)
    """
    # モデルとトークナイザーの読み込み
    print("モデルとトークナイザーを読み込んでいます...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # プロンプトをトークン化
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    print(f"\n入力プロンプト: '{prompt}'")
    print(f"\nトークン列に変換:")
    print(f"Token IDs: {input_ids[0].tolist()}")
    print(f"Tokens: {[tokenizer.decode([token_id]) for token_id in input_ids[0]]}")
    
    # モデルで予測
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        # 最後のトークンの次のトークンの確率分布を取得
        next_token_logits = outputs.logits[0, -1, :]
        
        # 確率に変換（softmax）
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        
        # 上位k個のトークンとその確率を取得
        top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k)
    
    # 結果をリストに格納
    results = []
    print(f"\n次のトークンとして適切なもの上位{top_k}個:")
    print("-" * 60)
    for i, (prob, token_id) in enumerate(zip(top_k_probs, top_k_indices), 1):
        token = tokenizer.decode([token_id.item()])
        results.append((token, prob.item()))
        print(f"{i:2d}. Token ID: {token_id.item():5d} | "
              f"Token: {repr(token):15s} | Probability: {prob.item():.6f}")
    
    return input_ids[0].tolist(), results


def main():
    # 問題90: 次単語予測
    prompt = "The movie was full of"
    token_ids, top_tokens = predict_next_token(prompt, top_k=10)
    
    print("\n" + "=" * 60)
    print("まとめ:")
    print("=" * 60)
    print(f"入力: '{prompt}'")
    print(f"トークン数: {len(token_ids)}")
    print(f"\n上位3つの次のトークン:")
    for i, (token, prob) in enumerate(top_tokens[:3], 1):
        print(f"  {i}. {repr(token)} (確率: {prob:.4f})")


if __name__ == "__main__":
    main()
