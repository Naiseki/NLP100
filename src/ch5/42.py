import os
import sys
from typing import Optional, List, Tuple
import re
import unicodedata
import pandas as pd
import time
from  google import genai
from dataclasses import dataclass


def call_gemini(prompt: str, client: genai.Client, model_id: str, temperature: float) -> Optional[str]:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[prompt],
                config=genai.types.GenerateContentConfig(
                    temperature=temperature
                )
            )
            # response.text があるケースは返し、なければ response を文字列化して返す
            if hasattr(response, "text") and response.text:
                return response.text
            return str(response)
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < max_retries - 1:
                    print(f"API過負荷、再試行 {attempt + 1}/{max_retries}", file=sys.stderr)
                    time.sleep(2 ** attempt)  # 指数バックオフ
                    continue
            print("Gemini API 呼び出しエラー:", e, file=sys.stderr)
            return None
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

def process_row(row, client: genai.Client, model_id: str, temperature: float) -> Optional[str]:
    prompt = f"問題: {row.question}\nA: {row.choice_a}\nB: {row.choice_b}\nC: {row.choice_c}\nD: {row.choice_d}\n正解の選択肢をA,B,C,Dで答えてください。"
    response = call_gemini(prompt, client, model_id, temperature)
    return extract_choice_from_text(response) if response else None

def main():
    csv_path = "input/college_computer_science.csv"
    model_id = "gemini-2.5-flash-lite"
    temperature = 0.2
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("環境変数 GEMINI_API_KEY に API キーを設定してください。", file=sys.stderr)
        return None
    client = genai.Client(api_key=api_key)

    # CSV を読み込む
    if not os.path.exists(csv_path):
        print(f"CSV ファイルが見つかりません: {csv_path}", file=sys.stderr)
        return

    columns = ["question", "choice_a", "choice_b", "choice_c", "choice_d", "answer"]
    df = pd.read_csv(csv_path, names=columns, header=None, encoding="utf-8-sig")

    total_count = len(df)

    # df.itertuplesを使って各行を処理し、predicted列を追加
    predicted_list = []
    for i, row in enumerate(df.itertuples(), 1):
        print(f"処理中: {i}/{total_count}", end='\r')
        predicted = process_row(row, client, model_id, temperature)
        predicted_list.append(predicted)
        time.sleep(5)  # レートリミット: 1分あたり12リクエスト (60/12=5秒)
    df["predicted"] = predicted_list
    print()  # 改行して進捗表示をクリア

    # 正解率を計
    correct_count = (df["predicted"] == df["answer"]).sum()
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"正解率: {accuracy:.2%} ({correct_count}/{total_count})")

    for i, row in enumerate(df.itertuples(), 1):
        print(f"Q{i}: {row.question}")
        print(f"予測: {row.predicted}, 正解: {row.answer}")
        print("-----")

    # 結果をテキストファイルに出力
    output_path = "output/ch5/42_out.txt"
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
