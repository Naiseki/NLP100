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
        以下の例を参考にして，最後の問題に答えてください．
        === 例1 ===
        日本の近代化に関連するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

        ア　府知事・県令からなる地方官会議が設置された。
        イ　廃藩置県が実施され，中央から府知事・県令が派遣される体制になった。
        ウ　すべての藩主が，天皇に領地と領民を返還した。

        解答: ウ→イ→ア

        === 例2 ===
        江戸幕府の北方での対外的な緊張について述べた次の文ア～ウを年代の古い順に正しく並べよ。

        ア　レザノフが長崎に来航したが，幕府が冷淡な対応をしたため，ロシア船が樺太や択捉島を攻撃した。
        イ　ゴローウニンが国後島に上陸し，幕府の役人に捕らえられ抑留された。
        ウ　ラクスマンが根室に来航し，漂流民を届けるとともに通商を求めた。

        解答: ウ→ア→イ

        === 例3 ===
        中居屋重兵衛の生涯の期間におこったできごとについて述べた次のア～ウを，年代の古い順に正しく並べよ。

        ア　アヘン戦争がおこり，清がイギリスに敗北した。
        イ　異国船打払令が出され，外国船を撃退することが命じられた。
        ウ　桜田門外の変がおこり，大老の井伊直弼が暗殺された。

        解答: イ→ア→ウ

        === 例4 ===
        加藤高明が外務大臣として提言を行ってから、内閣総理大臣となり演説を行うまでの時期のできごとについて述べた次のア～ウを，年代の古い順に正しく並べよ。

        ア　朝鮮半島において，独立を求める大衆運動である三・一独立運動が展開された。
        イ　関東大震災後の混乱のなかで，朝鮮人や中国人に対する殺傷事件がおきた。
        ウ　日本政府が，袁世凱政府に対して二十一カ条の要求を突き付けた。

        解答: ウ→ア→イ

        上記の例を参考にして、この問題の解答を出力してください。
        === 問題 ===
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
