"""
93. パープレキシティ

事前学習済み言語モデル（GPT-2）を用いて、指定された4文のパープレキシティを測定する。
文法的な正しさがパープレキシティの数値にどのように影響するかを観察する。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math

def calculate_perplexity(sentence, model, tokenizer, device):
    """文のパープレキシティを計算する
    
    PPL(X) = exp( -1/N * sum(log P(x_i | x_{<i})) )
    これは CrossEntropyLoss の exponent (exp(loss)) に相当する。
    """
    # 入力をトークン化
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Causal LMでは、labelsにinput_idsを渡すと内部でシフトして計算してくれる
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # 平均負の対数尤度
        perplexity = math.exp(loss.item())
        
    return perplexity

def main():
    # モデルとトークナイザーの準備
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # 評価する文のリスト
    sentences = [
        "The movie was full of surprises",
        "The movies were full of surprises",
        "The movie were full of surprises",  # 文法間違い
        "The movies was full of surprises"   # 文法間違い
    ]

    print(f"{'Sentence':<40} | {'Perplexity':>12}")
    print("-" * 55)

    for i, sentence in enumerate(sentences):
        ppl = calculate_perplexity(sentence, model, tokenizer, device)
        label = ""
        print(f"{sentence + label:<40} | {ppl:12.4f}")

if __name__ == "__main__":
    main()
