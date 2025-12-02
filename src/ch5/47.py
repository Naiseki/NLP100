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
