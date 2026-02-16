"""
92. 予測されたテキストの確率を計算

"The movie was full of"に続くテキストを予測し、
生成された各単語の尤度（確率）を表示する。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

def main():
    prompt = "The movie was full of"
    model_name = "gpt2"
    
    # トークナイザーとモデルのロード
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 入力をトークン化
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # テキスト生成 (Greedy Decodingで短めに生成)
    max_length = 15
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )
    
    # 生成されたシーケンス
    generated_ids = output_ids.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"Full Text: {generated_text}")
    print("-" * 50)
    print(f"{'Token':<15} | {'Logit':>10} | {'Probability':>12}")
    print("-" * 50)
    
    # プロンプト部分は含まず、生成された部分の確率を計算
    # model.generateのscoresは生成ステップごとのロジット
    # input_idsの長さから開始する
    input_len = input_ids.shape[1]
    
    for i, score in enumerate(output_ids.scores):
        # 個々のトークンのID
        token_id = generated_ids[input_len + i]
        token_str = tokenizer.decode(token_id)
        
        # ロジットから確率を計算
        probs = F.softmax(score, dim=-1)
        token_prob = probs[0, token_id].item()
        token_logit = score[0, token_id].item()
        
        print(f"{token_str:<15} | {token_logit:10.4f} | {token_prob:12.4f}")

if __name__ == "__main__":
    main()
