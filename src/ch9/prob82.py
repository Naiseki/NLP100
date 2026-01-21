from transformers import pipeline

def main():
    # マスク予測用のパイプラインをロード（上位10個を取得するように設定）
    unmasker = pipeline("fill-mask", model="bert-base-uncased", top_k=10)

    # 対象の文
    text = "The movie was full of [MASK]."

    # 予測の実行
    results = unmasker(text)

    # 結果の表示
    print(f"対象の文: {text}\n")
    print(f"{'Rank':<5} | {'Token':<15} | {'Probability':<12}")
    print("-" * 35)
    
    for i, res in enumerate(results, 1):
        token_str = res['token_str']
        score = res['score']
        print(f"{i:<5} | {token_str:<15} | {score:.4f}")

if __name__ == "__main__":
    main()
