from transformers import pipeline

def main():
    # マスク予測用のパイプラインをロード
    unmasker = pipeline("fill-mask", model="bert-base-uncased")

    # 対象の文（[MASK] を含む）
    text = "The movie was full of [MASK]."

    # 予測の実行
    results = unmasker(text)

    # 結果の表示（上位5件）
    print(f"元の文: {text}\n")
    print(f"{'Token':<15} | {'Score':<10} | {'Full Sentence'}")
    print("-" * 50)
    
    for res in results:
        token_str = res['token_str']
        score = res['score']
        sequence = res['sequence']
        print(f"{token_str:<15} | {score:.4f}     | {sequence}")

if __name__ == "__main__":
    main()
