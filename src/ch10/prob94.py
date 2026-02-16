"""
94. チャットテンプレート

"What do you call a sweet eaten after dinner?" という問いかけに対し、
チャットテンプレートを適用してプロンプトを作成し、応答を生成する。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    # 軽量なインストラクションモデルを使用
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 1. チャット形式のメッセージを定義
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What do you call a sweet eaten after dinner?"}
    ]
    
    # 2. チャットテンプレートを適用してプロンプトを作成
    # tokenize=False にすることでテキストとして取得
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("=" * 80)
    print("Generated Prompt (with Chat Template):")
    print("-" * 80)
    print(prompt)
    print("=" * 80)
    
    # 3. 応答の生成
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 生成された部分のみを取り出す（入力プロンプトの長さを除去）
    input_length = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print("\nModel Response:")
    print("-" * 80)
    print(response.strip())
    print("=" * 80)

if __name__ == "__main__":
    main()
