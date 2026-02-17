"""
97. 埋め込みに基づく感情分析

事前学習済み言語モデル（GPT系）でテキストをベクトルで表現（エンコード）し、
そのベクトルにフィードフォワード層を通すことで極性ラベルを予測するモデルを学習する。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

class SST2Dataset(Dataset):
    """SST-2データセット用のDatasetクラス"""
    
    def __init__(self, filepath, tokenizer, max_length=128):
        self.sentences = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # データ読み込み
        with open(filepath, 'r', encoding='utf-8') as f:
            next(f)  # ヘッダーをスキップ
            for line in f:
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
            'label': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier(nn.Module):
    """埋め込みベースの感情分類モデル（GPT系モデル使用）"""
    
    def __init__(self, model_name, num_labels=2, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        # 事前学習済みGPTモデル
        self.gpt = AutoModelForCausalLM.from_pretrained(model_name)
        
        # GPTモデルの次元数を取得
        self.hidden_size = self.gpt.config.hidden_size
        
        # フィードフォワード層（分類ヘッド）
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
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
        
        return logits

def train_epoch(model, dataloader, optimizer, criterion, device):
    """1エポックの学習"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 勾配をゼロに
        optimizer.zero_grad()
        
        # 順伝播
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # 逆伝播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 予測と正解ラベルを記録
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """評価"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 順伝播
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # 予測と正解ラベルを記録
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def main():
    # ハイパーパラメータ
    MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"  # GPT系モデル（Qwen）
    MAX_LENGTH = 128
    BATCH_SIZE = 16  # GPTモデルは大きいのでバッチサイズを減らす
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    HIDDEN_DIM = 256
    DROPOUT = 0.1
    
    # データパス
    TRAIN_PATH = Path("input/SST-2/train.tsv")
    DEV_PATH = Path("input/SST-2/dev.tsv")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # トークナイザーとモデルの初期化
    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
    
    # データセットとデータローダーの作成
    print("\nLoading datasets...")
    train_dataset = SST2Dataset(TRAIN_PATH, tokenizer, max_length=MAX_LENGTH)
    dev_dataset = SST2Dataset(DEV_PATH, tokenizer, max_length=MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")
    
    # 損失関数とオプティマイザー（分類層のみを学習）
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)
    
    # 学習ループ
    print(f"\n{'='*80}")
    print(f"Starting training for {NUM_EPOCHS} epochs")
    print(f"{'='*80}\n")
    
    best_dev_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 80)
        
        # 学習
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # 評価
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
        print(f"Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_acc:.4f}")
        
        # ベストモデルの保存
        if dev_acc > best_dev_accuracy:
            best_dev_accuracy = dev_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Best model saved! (Dev Accuracy: {dev_acc:.4f})")
    
    # 最終結果
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best Dev Accuracy: {best_dev_accuracy:.4f}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
