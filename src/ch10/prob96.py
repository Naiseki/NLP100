"""
96. プロンプトによる感情分析

事前学習済み言語モデルで感情分析を行う。
テキストを含むプロンプトを事前学習済み言語モデルに与え、
（ファインチューニングは行わずに）テキストのポジネガを予測し、
SST-2の開発データにおける正解率を測定する。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
from tqdm import tqdm

def load_sst2_dev_data(filepath):
    """SST-2の開発データを読み込む"""
    sentences = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)  # ヘッダーをスキップ
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    sentences.append(parts[0])
                    labels.append(int(parts[1]))
    
    return sentences, labels

def predict_sentiment(text, model, tokenizer, device):
    """プロンプトベースで感情を予測"""
    # プロンプトを作成（ゼロショット）
    messages = [
        {"role": "system", "content": "You are a sentiment analysis assistant."},
        {"role": "user", "content": f"Classify the sentiment of the following text as either 'positive' or 'negative'. Only respond with one word: 'positive' or 'negative'.\n\nText: {text}\n\nSentiment:"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,  # 決定的な出力
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    
    # 応答から感情を抽出
    if 'positive' in response:
        return 1
    elif 'negative' in response:
        return 0
    else:
        # デフォルトはnegative（慎重な予測）
        return 0

def main():
    # モデルとトークナイザーの読み込み
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    # データの読み込み
    data_path = Path("/net/nas4/data/home/kitahara/workspace/NLP100/input/SST-2/dev.tsv")
    print(f"\nLoading data from: {data_path}")
    sentences, labels = load_sst2_dev_data(data_path)
    print(f"Total samples: {len(sentences)}")
    
    # 予測
    correct = 0
    total = len(sentences)
    
    print("\nPredicting sentiments...")
    for i, (sentence, true_label) in enumerate(tqdm(zip(sentences, labels), total=total)):
        pred_label = predict_sentiment(sentence, model, tokenizer, device)
        if pred_label == true_label:
            correct += 1
        
        # 最初の数サンプルを表示
        if i < 5:
            print(f"\nSample {i+1}:")
            print(f"  Text: {sentence}")
            print(f"  True: {true_label}, Predicted: {pred_label}")
    
    # 正解率の計算
    accuracy = correct / total
    print(f"\n{'='*80}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
