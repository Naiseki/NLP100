import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import lightning as L
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torchmetrics import Accuracy

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

# 2. LightningModule の定義
class BERTClassifier(L.LightningModule):
    def __init__(self, model_name="bert-base-uncased", lr=2e-5):
        super().__init__()
        self.save_hyperparameters() # ハイパラを自動記録
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        
        # ログ記録
        self.train_acc(preds, batch['labels'])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        
        self.val_acc(preds, batch['labels'])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)

# 3. メイン処理
def main():
    # データの準備
    train_set = SST2Dataset("output/ch9/train_data.pth")
    dev_set = SST2Dataset("output/ch9/dev_data.pth")
    
    # TITAN RTX 向けに安定した設定
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=3)
    dev_loader = DataLoader(dev_set, batch_size=64, num_workers=3)

    # モデルの初期化
    model = BERTClassifier()

    # Trainer の設定（ハイパラ記録とGPU使用）
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=1,
        precision="32", # セグフォ回避のため16-mixedではなく32
        log_every_n_steps=10,
        default_root_dir="./output/ch9/lightning87/" # ここにログやチェックポイントが保存される
    )

    # 学習開始
    trainer.fit(model, train_loader, dev_loader)

if __name__ == "__main__":
    main()