import os
import sys
from typing import Optional
import json
from  google import genai
import re
import tiktoken


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

# ここから変更: トークン数計測用の main を追加
def _estimate_tokens_fallback(text: str):
    """
    簡易トークン推定（tiktokenが使えない場合のフォールバック）
    日本語の連続した文字列を1トークン、それ以外は英数字の連続や個別記号をトークンとみなす。
    """
    pattern = re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+|[A-Za-z0-9]+|[^\s]', re.UNICODE)
    tokens = pattern.findall(text)
    return tokens

def main():
    # 対象テキスト
    text = "吾輩は猫である。名前はまだ無い。\n\nどこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。"

    # cl100k_base を用いる
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text)
    used_method = "tiktoken (cl100k_base)"
    token_count = len(token_ids)

    # 出力表示（既に一部は表示しているが、ファイル出力の代わりに詳細を標準出力に出す）
    print(f"使用した方法: {used_method}")
    print(f"トークン数: {token_count}\n")
    if used_method.startswith("tiktoken"):
        print("先頭トークンID (最初の200個):")
        print(", ".join(map(str, token_ids[:200])))
    else:
        print("先頭トークン (最初の200個):")
        # フォールバックのトークンは多くなる可能性があるため、1行ずつ表示
        for idx, t in enumerate(token_ids[:200], 1):
            print(f"{idx}. {t}")


if __name__ == "__main__":
    main()
