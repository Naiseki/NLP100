import os
import sys
from typing import Optional
import json
from  google import genai
import re
import time
import statistics
import csv


def call_gemini(prompt: str, model_id: str, temperature: float) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("環境変数 GEMINI_API_KEY に API キーを設定してください。", file=sys.stderr)
        return None
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                temperature=temperature
            )
        )
        return response.text
    except Exception as e:
        print("Gemini API 呼び出しエラー:", e, file=sys.stderr)
        return None


# モデル出力から1〜10のスコアを抽出する
def parse_scores_from_text(text: str, n_items: int = 10):
    scores = [None] * n_items
    if not text:
        return scores
    lines = text.splitlines()
    for line in lines:
        # 先頭に番号がある行 (例: "1. 7", "1) 8/10", "1. 7点 理由...")
        m = re.match(r'\s*(\d{1,2})[\.\)\:]\s*(?:[:\-]?\s*)?(\d{1,2})(?:\s*\/\s*10|点)?', line)
        if m:
            idx = int(m.group(1))
            val = int(m.group(2))
            if 1 <= idx <= n_items and 1 <= val <= 10:
                scores[idx - 1] = val
            continue
        # または行中に "1 7" や "1: 7/10" のような形式
        m2 = re.match(r'\s*(\d{1,2})[^\d\n]*?(\d{1,2})(?:\s*\/\s*10|点)?', line)
        if m2:
            idx = int(m2.group(1))
            val = int(m2.group(2))
            if 1 <= idx <= n_items and 1 <= val <= 10:
                scores[idx - 1] = val
            continue
    # テキスト中の "x/10" を順に拾って埋める（番号が無い場合の救済）
    if any(s is None for s in scores):
        found = re.findall(r'(\d{1,2})\s*\/\s*10', text)
        i = 0
        for f in found:
            if i >= n_items:
                break
            try:
                v = int(f)
            except:
                continue
            if 1 <= v <= 10 and scores[i] is None:
                scores[i] = v
                i += 1
    return scores


def main():
    prompt = (
        """
        以下の10句のすだちに関する川柳の面白さをそれぞれ10段階 (1〜10) で評価し，その理由を簡潔に述べてください。
        1. 絞りすぎ 汗が目にしむ 徳島かな
        2. 焼き魚 添えて笑顔が ほころんだ
        3. 薬味か 主役か 悩む すだちかな
        4. 徳島から 届いた便り 緑の香
        5. 貧乏ゆすり 止まらぬ指に すだちかな
        6. 刺身には 欠かせぬ相棒 すだちだよ
        7. 焼酎に 浮かべりゃ気分 上々だ
        8. 料理長 腕を振るうも すだち頼み
        9. 季節感 ぎゅっと詰まった 小さな実
        10. 忘れ物 探すうちに 香り立つ

        各行は「番号. スコア（1-10）」のように出力し、後に簡潔な理由を追加してください。スコアは必ず1〜10の整数でお願いします。
        """
    )

    model_id = "gemini-2.5-flash-lite"
    temperature = 0.5
    trials = 10  # 繰り返し回数

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("環境変数 GEMINI_API_KEY に API キーを設定してください。", file=sys.stderr)
        return
    client = genai.Client(api_key=api_key)

    all_scores = []  # 各試行ごとのスコアリストを格納
    successful_trials = 0
    for t in range(1, trials + 1):
        print(f"処理中: 試行 {t}/{trials}", end='\r')
        result = call_gemini(prompt, model_id=model_id, temperature=temperature)
        if result is None:
            print(f"\n試行 {t} で応答が得られませんでした。")
        else:
            scores = parse_scores_from_text(result, n_items=10)
            all_scores.append(scores)
            successful_trials += 1
    print()  # 改行

    if successful_trials == 0:
        print("有効な応答が得られませんでした。", file=sys.stderr)
        return

    # 各項目ごとに有効なスコアを集めて統計量を計算
    per_item_stats = []
    n_items = 10
    for i in range(n_items):
        vals = [s[i] for s in all_scores if s[i] is not None]
        count = len(vals)
        if count == 0:
            mean = None
            var = None
            stdev = None
        else:
            mean = statistics.mean(vals)
            var = statistics.pvariance(vals) if count >= 1 else None
            stdev = statistics.sqrt(var) if var is not None else None
        per_item_stats.append({
            "index": i + 1,
            "mean": mean,
            "variance": var,
            "stddev": stdev,
            "samples": vals,
            "count": count
        })

    # 出力ディレクトリ作成
    out_dir = "output/ch5"
    os.makedirs(out_dir, exist_ok=True)

    # CSV 出力
    csv_path = os.path.join(out_dir, "48_variance.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["index", "mean", "variance", "stddev", "count", "samples"])
        for st in per_item_stats:
            writer.writerow([st["index"], st["mean"], st["variance"], st["stddev"], st["count"], json.dumps(st["samples"], ensure_ascii=False)])
    print(f"集計結果を CSV に出力しました: {csv_path}")

    # テキスト出力
    txt_path = os.path.join(out_dir, "48_variance.txt")
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(f"試行回数: {trials}, 有効応答数: {successful_trials}\n")
        tf.write("\n")
        for st in per_item_stats:
            tf.write(f"{st['index']}. 平均: {st['mean']}, 分散: {st['variance']}, 標準偏差: {st['stddev']}, サンプル数: {st['count']}\n")
            tf.write(f"   サンプル値: {st['samples']}\n")
        tf.write("\n生データ（各試行ごとのスコア: None は抽出不可）:\n")
        for ti, s in enumerate(all_scores, 1):
            tf.write(f"試行{ti}: {s}\n")
    print(f"可読結果をテキストで出力しました: {txt_path}")


if __name__ == "__main__":
    main()
