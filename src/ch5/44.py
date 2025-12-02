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
    prompt = "つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。"

    model_id = "gemini-2.5-flash-lite"
    result = call_gemini(prompt, model_id=model_id, temperature=0.2)
    if result is None:
        print("モデルから応答が得られませんでした。", file=sys.stderr)
        return

    print("=== Gemini の出力 ===")
    print(result)


if __name__ == "__main__":
    main()
