"""
97. 埋め込みに基づく感情分析

事前学習済み言語モデル（GPT系）でテキストをベクトルで表現（エンコード）し、
そのベクトルにフィードフォワード層を通すことで極性ラベルを予測するモデルを学習する。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
import os

class SST2Dataset(Dataset):
    """SST-2データセット用のDatasetクラス"""
    
    def __init__(self, filepath, tokenizer, max_length=128, max_samples=None):
        self.sentences = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # データ読み込み
        with open(filepath, 'r', encoding='utf-8') as f:
            next(f)  # ヘッダーをスキップ
            for line in f:
                if max_samples and len(self.sentences) >= max_samples:
                    break
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        self.sentences.append(parts[0])
                        self.labels.append(int(parts[1]))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        # トークン化
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier(nn.Module):
    """埋め込みベースの感情分類モデル（GPT系モデル使用）"""
    
    def __init__(self, model_name, num_labels=2, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        # 事前学習済みGPTモデル
        self.gpt = AutoModelForCausalLM.from_pretrained(model_name)
        
        # GPTモデルの次元数を取得
        self.hidden_size = self.gpt.config.hidden_size
        self.num_labels = num_labels
        
        # フィードフォワード層（分類ヘッド）
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        # GPTでテキストをベクトル化
        outputs = self.gpt.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 最後のトークンの隠れ状態を使用
        # GPTではシーケンスの最後のトークンが文脈を集約している
        hidden_states = outputs.hidden_states[-1]  # 最終層の隠れ状態
        
        # 各サンプルの最後の有効トークンの位置を取得
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        
        # 最後のトークンの埋め込みを抽出
        pooled_output = hidden_states[range(batch_size), sequence_lengths, :]
        
        # 分類層を通して予測
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return (loss, logits) if loss is not None else logits

def compute_metrics(eval_pred):
    """評価メトリクスの計算"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def main():
    # ハイパーパラメータ
    MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"  # GPT系モデル（Qwen）
    MAX_LENGTH = 128
    BATCH_SIZE = 16  # GPTモデルは大きいのでバッチサイズを減らす
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    HIDDEN_DIM = 256
    DROPOUT = 0.1
    MAX_TRAIN_SAMPLES = 1000  # 訓練データを制限
    
    # データパス
    TRAIN_PATH = Path("input/SST-2/train.tsv")
    DEV_PATH = Path("input/SST-2/dev.tsv")

    # デバイス設定
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
    
    # パディングトークンの設定（GPT系モデルには通常必要）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = SentimentClassifier(
        model_name=MODEL_NAME,
        num_labels=2,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    )
    
    # GPTモデルをフリーズ（勾配計算を停止）
    for param in model.gpt.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    
    # データセットの作成
    print("\nLoading datasets...")
    train_dataset = SST2Dataset(TRAIN_PATH, tokenizer, max_length=MAX_LENGTH, max_samples=MAX_TRAIN_SAMPLES)
    dev_dataset = SST2Dataset(DEV_PATH, tokenizer, max_length=MAX_LENGTH)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")
    
    # トレーニング設定
    training_args = TrainingArguments(
        output_dir="output/ch10/sentiment_classifier",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,  # カスタムモデルの場合はTrueにすると必要な引数が消える場合がある
        bf16=torch.cuda.is_available(),  # FP16の代わりにBF16を使用
        save_safetensors=False,  # 共有テンソルによる保存エラーを回避
    )
    
    # Trainerの初期化
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 学習実行
    print(f"\n{'='*80}")
    print(f"Starting training for {NUM_EPOCHS} epochs")
    print(f"{'='*80}\n")
    
    trainer.train()
    
    # 最終評価
    eval_results = trainer.evaluate()
    best_dev_accuracy = eval_results["eval_accuracy"]
    
    # 最終結果
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best Dev Accuracy: {best_dev_accuracy:.4f}")
    print(f"{'='*80}")

    # 結果の保存
    output_file = Path("output/ch10/out97.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Sentiment Analysis based on Embeddings Results (using Trainer)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Dev samples: {len(dev_dataset)}\n\n")
        f.write(f"Best Dev Accuracy: {best_dev_accuracy:.4f}\n")
        f.write(f"Model saved to: {training_args.output_dir}\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
