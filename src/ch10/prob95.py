"""
95. マルチターンのチャット

問題94の応答に続き、"Please give me the plural form of the word with its spelling in reverse order."
という問いかけを行い、マルチターンの対話を実現する。
その際のプロンプトと応答を確認する。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    # 問題94と同じモデルを使用
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 1. 最初のターン
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What do you call a sweet eaten after dinner?"}
    ]
    
    # 最初の応答を生成
    prompt1 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs1 = tokenizer(prompt1, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs1 = model.generate(
            **inputs1,
            max_new_tokens=50,
            do_sample=False, # 再現性のためにGreedy
            pad_token_id=tokenizer.eos_token_id
        )
    
    response1 = tokenizer.decode(outputs1[0][inputs1.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # 2. 2番目のターン（追加の問いかけ）
    messages.append({"role": "assistant", "content": response1})
    messages.append({"role": "user", "content": "Please give me the plural form of the word with its spelling in reverse order."})
    
    # チャットテンプレートを適用してマルチターンのプロンプトを作成
    prompt2 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("=" * 80)
    print("Multi-turn Prompt:")
    print("-" * 80)
    print(prompt2)
    print("=" * 80)
    
    # 2番目の応答を生成
    inputs2 = tokenizer(prompt2, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs2 = model.generate(
            **inputs2,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response2 = tokenizer.decode(outputs2[0][inputs2.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    print("\nModel Response 1:")
    print(response1)
    print("\nModel Response 2:")
    print("-" * 80)
    print(response2)
    print("=" * 80)

if __name__ == "__main__":
    main()
