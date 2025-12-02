import os
import sys
from typing import Optional
import re
import unicodedata
import pandas as pd
import time
from transformers import pipeline, Pipeline
import torch
# 注意: 事前に pip install transformers torch accelerate が必要です
from huggingface_hub import login as hf_login  # optional: login for gated models
from datasets import Dataset  # 追加: バッチ処理用


def call_local_model(prompt: str, llm: Pipeline, max_tokens: int = 64) -> Optional[str]:
    """
    transformers pipelineを使ってプロンプトを投げ、文字列出力を返す。
    """
    try:
        outputs = llm(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,  # 更新: サンプリング有効化
            temperature=0.2,  # 更新: 温度設定
            return_full_text=False
        )
        if isinstance(outputs, list) and len(outputs) > 0:
            out = outputs[0]
            text = out.get("generated_text") or out.get("text") or str(out)
            return text.strip() if text else None
        return None
    except Exception as e:
        print("ローカルモデル呼び出しエラー:", e, file=sys.stderr)
        return None


def extract_choice_from_text(text: str) -> Optional[str]:
    """
    モデルの出力から選択肢 (A/B/C/D) を抽出する。
    全角→半角正規化を行い、大文字化して抽出する。
    """
    if not text:
        return None
    s = unicodedata.normalize("NFKC", text).upper()
    # A,B,C,D の最初に現れる文字を取得
    m = re.search(r"\b([A-D])\b", s)
    if m:
        return m.group(1)
    # 拡張: "A." や "A)" などのケースも考慮
    m2 = re.search(r"([A-D])[\.\)\s:、。]?", s)
    if m2:
        return m2.group(1)
    return None

def main():
    csv_path = "input/college_computer_science.csv"
    # 温度を削除: 無効なパラメータのため
    model_id = "google/gemma-3-4b-it"
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("環境変数 HF_TOKEN に Hugging Face のアクセストークンを設定してください。")

    try:
        device_map = "auto" if torch.cuda.is_available() else None
        client = pipeline("text-generation", model=model_id, device_map=device_map)
    except Exception as e:
        print("transformers pipeline の生成に失敗しました:", e, file=sys.stderr)
        if ("403" in str(e)) or ("Access to model" in str(e)) or ("not found" in str(e).lower()):
            print("モデルが gated の場合は、環境変数 HF_TOKEN にアクセストークンを設定してください。", file=sys.stderr)
        print("必要なパッケージがインストールされているか、モデルが transformers 互換か確認してください。", file=sys.stderr)
        return None

    # CSV を読み込む
    if not os.path.exists(csv_path):
        print(f"CSV ファイルが見つかりません: {csv_path}", file=sys.stderr)
        return

    columns = ["question", "choice_a", "choice_b", "choice_c", "choice_d", "answer"]
    df = pd.read_csv(csv_path, names=columns, header=None, encoding="utf-8-sig")

    total_count = len(df)

    # プロンプトリストを作成
    prompts = [
        f"問題: {row.question}\nA: {row.choice_a}\nB: {row.choice_b}\nC: {row.choice_c}\nD: {row.choice_d}\n正解の選択肢をA,B,C,Dで答えてください。"
        for row in df.itertuples()
    ]

    # Dataset作成とバッチ生成
    dataset = Dataset.from_dict({"prompt": prompts})
    outputs = client(
        list(dataset["prompt"]),  # Columnをリストに変換
        max_new_tokens=64,
        do_sample=True,  # サンプリング有効化
        temperature=0.2,  # 温度設定
        return_full_text=False,
        batch_size=8  # バッチサイズを調整可能
    )

    # 出力から予測を抽出
    predicted_list = [extract_choice_from_text(out[0]["generated_text"]) if out else None for out in outputs]

    df["predicted"] = predicted_list

    # 正解率を計算して表示
    correct_count = (df["predicted"] == df["answer"]).sum()
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"正解率: {accuracy:.2%} ({correct_count}/{total_count})")

    for i, row in enumerate(df.itertuples(), 1):
        print(f"Q{i}: {row.question}")
        print(f"予測: {row.predicted}, 正解: {row.answer}")
        print("-----")

    # 結果をテキストファイルに出力
    output_path = "output/ch5/43_out.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"正解率: {accuracy:.2%} ({correct_count}/{total_count})\n")
        for i, row in enumerate(df.itertuples(), 1):
            f.write(f"Q{i}: {row.question}\n")
            f.write(f"予測: {row.predicted}, 正解: {row.answer}\n")
            f.write("-----\n")
    print(f"結果を {output_path} に出力しました。")

    return


if __name__ == "__main__":
    main()
