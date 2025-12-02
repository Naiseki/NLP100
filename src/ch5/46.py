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
        すだちに関する川柳を10句作ってください。各句は日本語で、できるだけ5-7-5のリズムを意識してください。
        ユーモアや風刺を含めても構いませんが、句以外の説明や補足は含めず、番号付きのリストで次の形式で出力してください。

        1. 句1
        2. 句2
        ...
        10. 句10

        句が5-7-5にならない場合でも、可能な範囲で5-7-5を意識して表現してください。
        """
    )

    model_id = "gemini-2.5-flash-lite"
    result = call_gemini(prompt, model_id=model_id, temperature=0.5)
    if result is None:
        print("モデルから応答が得られませんでした。", file=sys.stderr)
        return

    print("=== Gemini の出力 ===")
    print(result)


if __name__ == "__main__":
    main()
