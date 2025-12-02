import os
import sys
from typing import Optional
import json
from  google import genai


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


def main():
    prompt = (
        """
        9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。
        ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
        イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
        ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
        """
    )

    model_id = "gemini-2.5-flash-lite"
    result = call_gemini(prompt, model_id=model_id, temperature=0.2)
    if result is None:
        print("モデルから応答が得られませんでした。", file=sys.stderr)
        return

    print("=== Gemini の出力 ===")
    print(result)


if __name__ == "__main__":
    main()
