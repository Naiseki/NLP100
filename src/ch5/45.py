import os
import sys
from typing import Optional
import json
from  google import genai


def call_gemini(chat, prompt: str) -> Optional[str]:
    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        print("Gemini API 呼び出しエラー:", e, file=sys.stderr)
        return None


def main():
    prompt = "つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。"

    model_id = "gemini-2.5-flash-lite"
    temperature = 0.2
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("環境変数 GEMINI_API_KEY に API キーを設定してください。", file=sys.stderr)
        return
    client = genai.Client(api_key=api_key)
    chat = client.chats.create(
        model=model_id,
        config=genai.types.GenerateContentConfig(temperature=temperature)
    )

    result = call_gemini(chat, prompt)
    if result is None:
        print("モデルから応答が得られませんでした。", file=sys.stderr)
        return

    print("=== Gemini の出力 (1回目) ===")
    print(result)

    additional_prompt = "さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？"
    result2 = call_gemini(chat, additional_prompt)
    if result2 is None:
        print("モデルから応答が得られませんでした。", file=sys.stderr)
        return

    print("=== Gemini の出力 (2回目) ===")
    print(result2)


if __name__ == "__main__":
    main()
