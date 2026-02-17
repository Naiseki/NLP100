"""
98. ファインチューニング

prob96.pyのプロンプトに対して、正解の感情ラベルをテキストの応答として返すように
事前学習済みモデルをファインチューニングする。
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from pathlib import Path
from tqdm import tqdm
import numpy as np

class SST2FineTuneDataset(Dataset):
    """ファインチューニング用のSST-2データセット"""
    
    def __init__(self, filepath, tokenizer, max_length=512, max_samples=None):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # データ読み込み
        with open(filepath, 'r', encoding='utf-8') as f:
            next(f)  # ヘッダーをスキップ
            for line in f:
                if max_samples and len(self.data) >= max_samples:
                    break
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        text = parts[0]
                        label = int(parts[1])
                        label_text = "positive" if label == 1 else "negative"
                        
                        # prob96.pyと同じプロンプト形式
                        messages = [
                            {"role": "system", "content": "You are a sentiment analysis assistant."},
                            {"role": "user", "content": f"Classify the sentiment of the following text as either 'positive' or 'negative'. Only respond with one word: 'positive' or 'negative'.\n\nText: {text}\n\nSentiment:"},
                            {"role": "assistant", "content": label_text}
                        ]
                        
                        self.data.append(messages)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        messages = self.data[idx]
        
        # チャット形式でトークン化
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # トークン化
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # ラベルはinput_idsと同じ（言語モデリング）
        labels = input_ids.clone()
        
        # プロンプト部分（assistant応答以外）はlossを計算しない
        # assistant応答部分のみlossを計算するため、それ以外を-100にマスク
        # これは簡易実装として、全体を学習対象とする
        # より精緻にはassistant応答部分のみをラベルとするべき
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def predict_sentiment(text, model, tokenizer, device):
    """ファインチューニング後のモデルで感情を予測"""
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
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    
    # 応答から感情を抽出
    if 'positive' in response:
        return 1
    elif 'negative' in response:
        return 0
    else:
        return 0

def evaluate_model(model, tokenizer, dev_path, device):
    """開発データで評価"""
    sentences = []
    labels = []
    
    with open(dev_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    sentences.append(parts[0])
                    labels.append(int(parts[1]))
    
    correct = 0
    total = len(sentences)
    
    print("\nEvaluating on dev set...")
    for sentence, true_label in tqdm(zip(sentences, labels), total=total):
        pred_label = predict_sentiment(sentence, model, tokenizer, device)
        if pred_label == true_label:
            correct += 1
    
    accuracy = correct / total
    return accuracy

def main():
    # ハイパーパラメータ
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"  # 小さいモデルで高速に
    MAX_LENGTH = 256  # 短くしてメモリ削減
    BATCH_SIZE = 4  # メモリ不足対策で削減
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    OUTPUT_DIR = "./finetuned_sentiment_model"
    MAX_TRAIN_SAMPLES = 1000  # 訓練データを1000サンプルに制限（高速化）
    
    # データパス
    TRAIN_PATH = Path("input/SST-2/train.tsv")
    DEV_PATH = Path("input/SST-2/dev.tsv")
    
    # デバイス設定（単一GPUに制限）
    import os
    if torch.cuda.is_available():
        # 最初のGPUのみを使用
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # トークナイザーとモデルの初期化
    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # パディングトークンの設定（必要な場合）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="auto")
    
    # メモリ削減のためgradient checkpointingを有効化
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    model = model.to(device)
    
    # データセットの作成
    print("\nLoading datasets...")
    train_dataset = SST2FineTuneDataset(TRAIN_PATH, tokenizer, max_length=MAX_LENGTH, max_samples=MAX_TRAIN_SAMPLES)
    dev_dataset = SST2FineTuneDataset(DEV_PATH, tokenizer, max_length=MAX_LENGTH)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")
    
    # ファインチューニング前の評価
    print(f"\n{'='*80}")
    print("Evaluating BEFORE fine-tuning...")
    print(f"{'='*80}")
    before_accuracy = evaluate_model(model, tokenizer, DEV_PATH, device)
    print(f"Accuracy before fine-tuning: {before_accuracy:.4f}")
    
    # トレーニング設定
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        gradient_accumulation_steps=4,  # 実効バッチサイズ=4*4=16
        bf16=torch.cuda.is_available(),  # FP16の代わりにBF16を使用
        gradient_checkpointing=True,  # メモリ削減
        max_grad_norm=1.0,  # 勾配クリッピング
    )
    
    # Trainerの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    
    # ファインチューニング
    print(f"\n{'='*80}")
    print(f"Starting fine-tuning for {NUM_EPOCHS} epochs")
    print(f"{'='*80}\n")
    
    trainer.train()
    
    # ファインチューニング後の評価
    print(f"\n{'='*80}")
    print("Evaluating AFTER fine-tuning...")
    print(f"{'='*80}")
    after_accuracy = evaluate_model(model, tokenizer, DEV_PATH, device)
    print(f"Accuracy after fine-tuning: {after_accuracy:.4f}")
    
    # 結果のサマリー
    print(f"\n{'='*80}")
    print(f"Fine-tuning completed!")
    print(f"Before: {before_accuracy:.4f}")
    print(f"After:  {after_accuracy:.4f}")
    print(f"Improvement: {after_accuracy - before_accuracy:+.4f}")
    print(f"{'='*80}")
    
    # モデルの保存
    print(f"\nSaving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 結果の保存
    output_file = Path("output/ch10/out98.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("SFT (Supervised Fine-Tuning) Training Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Dev samples: {len(dev_dataset)}\n\n")
        f.write(f"Accuracy before SFT: {before_accuracy:.4f}\n")
        f.write(f"Accuracy after SFT: {after_accuracy:.4f}\n")
        f.write(f"Improvement: {after_accuracy - before_accuracy:+.4f}\n\n")
        f.write(f"Model saved to: {OUTPUT_DIR}\n")
    
    print(f"\nResults saved to: {output_file}")
    print("\nDone!")

if __name__ == "__main__":
    main()
