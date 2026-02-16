"""
91. 続きのテキストの予測

"The movie was full of"に続くテキストを複数予測する。
デコーディングの方法や温度パラメータ（temperature）を変えながら、
予測される複数のテキストの変化を観察する。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_text(prompt, method="greedy", **kwargs):
    """テキスト生成関数
    
    Args:
        prompt: 入力プロンプト
        method: デコーディング方法
        **kwargs: 生成パラメータ
    """
    # GPT-2モデルとトークナイザーをロード
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 入力をトークン化
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # テキスト生成
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )
    
    # デコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    prompt = "The movie was full of"
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print("=" * 80)
    
    # 1. 貪欲法 (デフォルト)
    print("\n1. 貪欲法:")
    print("-" * 80)
    result = generate_text(prompt, method="greedy")
    print(result)
    
    # 2. Beam Search (num_beams=5)
    print("\n2. Beam Search (num_beams=5):")
    print("-" * 80)
    result = generate_text(prompt, method="beam_search", num_beams=5, early_stopping=True)
    print(result)
    
    # 3. サンプリング (Temperature = 0.5)
    print("\n3. サンプリング (Temperature = 0.5):")
    print("-" * 80)
    result = generate_text(prompt, method="sampling", do_sample=True, temperature=0.5)
    print(result)
    
    # 4. サンプリング (Temperature = 1.0)
    print("\n4. サンプリング (Temperature = 1.0):")
    print("-" * 80)
    result = generate_text(prompt, method="sampling", do_sample=True, temperature=1.0)
    print(result)
    
    # 5. サンプリング (Temperature = 1.5)
    print("\n5. サンプリング (Temperature = 1.5):")
    print("-" * 80)
    result = generate_text(prompt, method="sampling", do_sample=True, temperature=1.5)
    print(result)
    
    # 6. Top-k サンプリング (k=50)
    print("\n6. Top-k サンプリング (k=50):")
    print("-" * 80)
    result = generate_text(prompt, method="top_k", do_sample=True, top_k=50, temperature=1.0)
    print(result)
    
    # 7. Top-p (Nucleus) サンプリング (p=0.95)
    print("\n7. Top-p (Nucleus) サンプリング (p=0.95):")
    print("-" * 80)
    result = generate_text(prompt, method="top_p", do_sample=True, top_p=0.95, temperature=1.0)
    print(result)
    
    # 8. Top-k + Top-p サンプリング
    print("\n8. Top-k + Top-p サンプリング (k=50, p=0.95):")
    print("-" * 80)
    result = generate_text(prompt, method="combined", do_sample=True, top_k=50, top_p=0.95, temperature=1.0)
    print(result)

if __name__ == "__main__":
    main()
