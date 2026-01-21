import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

def get_mean_embeddings(sentences, tokenizer, model):
    # トークナイズ
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    # 推論
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 最終層の出力を取得 [batch_size, seq_len, hidden_dim]
    last_hidden_state = outputs.last_hidden_state
    
    # パディング部分を除外するためのAttention Maskを考慮
    mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings

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

    # 平均ベクトルの取得
    embeddings = get_mean_embeddings(sentences, tokenizer, model)

    # コサイン類似度の計算
    sim_matrix = cosine_similarity(embeddings)

    # 結果の表示
    print(f"{'Sentence A':<35} | {'Sentence B':<35} | {'Similarity'}")
    print("-" * 85)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            print(f"{sentences[i]:<35} | {sentences[j]:<35} | {sim_matrix[i][j]:.4f}")

if __name__ == "__main__":
    main()
