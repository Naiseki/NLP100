import torch
from transformers import AutoTokenizer
from prob87 import BERTClassifier

def predict_sentiment(text, model, tokenizer, device):
    # 1. 推論モードに設定
    model.eval()
    model.to(device)

    # 2. 入力文のトークン化
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    ).to(device)

    # 3. 勾配計算なしで推論
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Softmaxで確率に変換（必須ではないけど、確信度がわかるよ）
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()

    # ラベルの変換 (SST-2: 0=Negative, 1=Positive)
    label = "Positive" if prediction == 1 else "Negative"
    confidence = probs[0][prediction].item()
    
    return label, confidence

def main():
    # 1. パスの設定
    checkpoint_path = "output/ch9/lightning87/lightning_logs/version_1/checkpoints/epoch=2-step=3159.ckpt"
    model_name = "bert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. モデルとトークナイザのロード
    print(f"Loading checkpoint: {checkpoint_path}")
    # LightningModuleとしてロード
    lightning_model = BERTClassifier.load_from_checkpoint(checkpoint_path)
    # 推論用にモデル本体を抽出してGPUへ
    model = lightning_model.model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. 予測対象の文
    test_sentences = [
        "The movie was full of incomprehensibilities.",
        "The movie was full of fun.",
        "The movie was full of excitement.",
        "The movie was full of crap.",
        "The movie was full of rubbish."
    ]

    print(f"\n{'Sentence':<45} | {'Prediction':<10} | {'Confidence'}")
    print("-" * 75)

    # 4. 推論ループ
    for text in test_sentences:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()

        label = "Positive" if prediction == 1 else "Negative"
        conf = probs[0][prediction].item()
        print(f"{text:<45} | {label:<10} | {conf:.4f}")

if __name__ == "__main__":
    main()
