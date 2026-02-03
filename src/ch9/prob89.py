import torch
import torch.nn as nn
import lightning as L
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torchmetrics import Accuracy


class CustomBERTClassifier(L.LightningModule):
    def __init__(self, model_name="bert-base-uncased", lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # BERTのベースモデル（分類ヘッドなし）をロード
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 分類用の線形層（特徴次元数→2）
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        
        # 損失関数, 評価指標
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)

    def forward(self, input_ids, attention_mask):
        # BERTに予測させる
        # outputsから隠れ層の状態やアテンションなどが得られる
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 最終層 (last_hidden_state): [バッチサイズ, トークン数, 隠れ層次元数]
        last_hidden_state = outputs.last_hidden_state
        
        # [CLS]トークン（0番目のトークン）のベクトルを抽出
        # cls_token_embeddings: [バッチサイズ, 隠れ層次元数]
        cls_token_embeddings = last_hidden_state[:, 0, :]
        
        # 自前の分類層に通す
        logits = self.classifier(cls_token_embeddings)
        return logits

    def training_step(self, batch, batch_idx):
        # モデル実行をして、ロジット（ロジスティック関数の逆関数の値）を得る
        logits = self(batch['input_ids'], batch['attention_mask'])
        # 損失を計算
        loss = self.criterion(logits, batch['labels'])
        
        # ロジットが最大のクラスを予測とする
        # logits: [バッチサイズ, クラス数]なので、dim=1で最大値のインデックスを取得
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


# データセットの定義
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

def main():
    # データの準備
    train_set = SST2Dataset("output/ch9/train_data.pth")
    dev_set = SST2Dataset("output/ch9/dev_data.pth")
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=3)
    dev_loader = DataLoader(dev_set, batch_size=64, num_workers=3)

    # モデルの初期化
    model = CustomBERTClassifier()

    # Trainer の設定（ハイパラ記録とGPU使用）
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=1,
        precision="32", 
        log_every_n_steps=10,
        default_root_dir="./output/ch9/lightning89/" # ログやチェックポイントの保存先
    )

    # 学習開始
    trainer.fit(model, train_loader, dev_loader)

if __name__ == "__main__":
    main()
