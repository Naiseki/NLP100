"""
99. 選好チューニング

prob96.pyのプロンプトに対して、正解の感情ラベルを含むテキストを望ましい応答、
間違った感情ラベルを含むテキストを望ましくない応答として、
事前学習済み言語モデルを選好チューニング (preference tuning) を実施する。
DPO (Direct Preference Optimization) を利用する。
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig
import torch
from pathlib import Path
from datasets import Dataset
import pandas as pd
import os

def load_sst2_data(filepath):
    """SST-2のデータを読み込む"""
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

def create_dpo_dataset(sentences, labels, tokenizer):
    """DPO用のデータセットを作成
    
    各サンプルに対して：
    - prompt: prob96と同じプロンプト
    - chosen: 正解の感情ラベル
    - rejected: 間違った感情ラベル
    """
    data = []
    
    for sentence, label in zip(sentences, labels):
        # prob96と同じプロンプト形式
        messages = [
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {"role": "user", "content": f"Classify the sentiment of the following text as either 'positive' or 'negative'. Only respond with one word: 'positive' or 'negative'.\n\nText: {sentence}\n\nSentiment:"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # ラベルに応じた応答
        if label == 1:  # positive
            chosen = "positive"
            rejected = "negative"
        else:  # negative
            chosen = "negative"
            rejected = "positive"
        
        data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    
    return Dataset.from_list(data)

def evaluate_model(model, tokenizer, eval_sentences, eval_labels, device):
    """モデルの評価"""
    model.eval()
    correct = 0
    total = len(eval_sentences)
    
    for sentence, true_label in zip(eval_sentences, eval_labels):
        messages = [
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {"role": "user", "content": f"Classify the sentiment of the following text as either 'positive' or 'negative'. Only respond with one word: 'positive' or 'negative'.\n\nText: {sentence}\n\nSentiment:"}
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
            pred_label = 1
        elif 'negative' in response:
            pred_label = 0
        else:
            pred_label = 0
        
        if pred_label == true_label:
            correct += 1
    
    accuracy = correct / total
    return accuracy

def main():
    # ハイパーパラメータ
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
    MAX_TRAIN_SAMPLES = 1000  # 訓練データを制限
    
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    if torch.cuda.is_available():
        # 最初のGPUのみを使用
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # データの読み込み
    train_path = Path("input/SST-2/train.tsv")
    dev_path = Path("input/SST-2/dev.tsv")
    
    print(f"\nLoading training data from: {train_path}")
    train_sentences, train_labels = load_sst2_data(train_path)
    
    # データを制限
    if MAX_TRAIN_SAMPLES:
        train_sentences = train_sentences[:MAX_TRAIN_SAMPLES]
        train_labels = train_labels[:MAX_TRAIN_SAMPLES]
    
    print(f"Training samples: {len(train_sentences)}")
    
    print(f"Loading dev data from: {dev_path}")
    dev_sentences, dev_labels = load_sst2_data(dev_path)
    print(f"Dev samples: {len(dev_sentences)}")
    
    # DPO用のデータセット作成
    print("\nCreating DPO dataset...")
    train_dataset = create_dpo_dataset(train_sentences, train_labels, tokenizer)
    eval_dataset = create_dpo_dataset(dev_sentences, dev_labels, tokenizer)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # サンプル表示
    print("\nExample DPO data:")
    print(f"Prompt (first 200 chars): {train_dataset[0]['prompt'][:200]}...")
    print(f"Chosen: {train_dataset[0]['chosen']}")
    print(f"Rejected: {train_dataset[0]['rejected']}")
    
    # 参照モデル（元のモデル）の読み込み
    print("\nLoading reference model...")
    model_ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # DPO設定
    training_args = DPOConfig(
        output_dir="output/ch10/dpo_sentiment_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-7,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True if torch.cuda.is_available() else False,
        remove_unused_columns=False,
        beta=0.1,  # DPOのハイパーパラメータ
    )
    
    # DPOトレーナーの初期化
    print("\nInitializing DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # トレーニング前の評価
    print("\n" + "="*80)
    print("Evaluating before DPO training...")
    print("="*80)
    model.to(device)
    accuracy_before = evaluate_model(model, tokenizer, dev_sentences[:100], dev_labels[:100], device)
    print(f"Accuracy before DPO: {accuracy_before:.4f} (on 100 dev samples)")
    
    # トレーニング実行
    print("\n" + "="*80)
    print("Starting DPO training...")
    print("="*80)
    dpo_trainer.train()
    
    # トレーニング後の評価
    print("\n" + "="*80)
    print("Evaluating after DPO training...")
    print("="*80)
    model.to(device)
    accuracy_after = evaluate_model(model, tokenizer, dev_sentences[:100], dev_labels[:100], device)
    print(f"Accuracy after DPO: {accuracy_after:.4f} (on 100 dev samples)")
    print(f"Improvement: {accuracy_after - accuracy_before:.4f}")
    
    # モデルの保存
    print("\nSaving DPO-tuned model...")
    output_dir = Path("dpo_sentiment_model")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")
    
    # 結果の保存
    output_file = Path("output/ch10/out99.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DPO (Direct Preference Optimization) Training Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Training samples: {len(train_sentences)}\n")
        f.write(f"Dev samples: {len(dev_sentences)}\n\n")
        f.write(f"Accuracy before DPO: {accuracy_before:.4f}\n")
        f.write(f"Accuracy after DPO: {accuracy_after:.4f}\n")
        f.write(f"Improvement: {accuracy_after - accuracy_before:.4f}\n\n")
        f.write(f"Model saved to: {output_dir}\n")
    
    print(f"\nResults saved to: {output_file}")
    print("\nDone!")

if __name__ == "__main__":
    main()
