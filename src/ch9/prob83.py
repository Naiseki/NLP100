import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

def get_cls_embeddings(sentences, tokenizer, model):
    # トークナイズ（PyTorchテンソルとして返す）
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    # 推論
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 最終層の出力を取得
    # [バッチサイズ, トークン数, 隠れ層の次元] という形
    last_hidden_state = outputs.last_hidden_state
    
    # [CLS]トークン（各文の0番目のトークン）のベクトルを抽出
    cls_embeddings = last_hidden_state[:, 0, :]
    return cls_embeddings

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    sentences = [
        "The movie was full of fun.",
        "The movie was full of excitement.",
        "The movie was full of crap.",
        "The movie was full of rubbish."
    ]

    # ベクトルの取得
    embeddings = get_cls_embeddings(sentences, tokenizer, model)

    # 全ての組み合わせのコサイン類似度を計算
    # cosine_similarityは [n_samples, n_samples] の行列を返す
    sim_matrix = cosine_similarity(embeddings)

    # 結果の表示
    print(f"{'Sentence A':<35} | {'Sentence B':<35} | {'Similarity'}")
    print("-" * 85)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            print(f"{sentences[i]:<35} | {sentences[j]:<35} | {sim_matrix[i][j]:.4f}")

if __name__ == "__main__":
    main()
