import torch
import torch.nn as nn
import lightning as L
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from torchmetrics import Accuracy
from prob87 import SST2Dataset
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torchmetrics import Accuracy


class CustomBERTClassifier(L.LightningModule):
    def __init__(self, model_name="bert-base-uncased", lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. BERTのベースモデル（分類ヘッドなし）をロード
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 2. 分類用の線形層を自分で定義
        # BERTの隠れ層の次元（通常768）からクラス数（2）へ
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        
        # 3. 損失関数とメトリクス
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)

    def forward(self, input_ids, attention_mask):
        # BERTに通す
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 最終層の出力 (last_hidden_state): [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs.last_hidden_state
        
        # [CLS]トークン（0番目のトークン）のベクトルを抽出
        # cls_token_embeddings: [batch_size, hidden_size]
        cls_token_embeddings = last_hidden_state[:, 0, :]
        
        # 自前の分類層に通す
        logits = self.classifier(cls_token_embeddings)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.criterion(logits, batch['labels'])
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, batch['labels'])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.criterion(logits, batch['labels'])
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, batch['labels'])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)

def main():
    trrain_set = SST2Dataset("output/ch9/train_data.pth")
    dev_set = SST2Dataset("output/ch9/dev_data.pth")

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=3)
    dev_loader = DataLoader(dev_set, batch_size=64, num_workers=3)


# 1. データセットの定義
class SST2Dataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = data['labels']

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx].long()
        }

    def __len__(self):
        return len(self.labels)

# 3. メイン処理
def main():
    # データの準備
    train_set = SST2Dataset("output/ch9/train_data.pth")
    dev_set = SST2Dataset("output/ch9/dev_data.pth")
    
    # TITAN RTX 向けに安定した設定
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=3)
    dev_loader = DataLoader(dev_set, batch_size=64, num_workers=3)

    # モデルの初期化
    model = CustomBERTClassifier()

    # Trainer の設定（ハイパラ記録とGPU使用）
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=1,
        precision="32", # セグフォ回避のため16-mixedではなく32
        log_every_n_steps=10,
        default_root_dir="./output/ch9/lightning89/" # ここにログやチェックポイントが保存される
    )

    # 学習開始
    trainer.fit(model, train_loader, dev_loader)

if __name__ == "__main__":
    main()
