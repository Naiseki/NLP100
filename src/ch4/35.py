import sys
import spacy
import ginza

def main():
    text = """
    メロスは激怒した。
    必ず、かの邪智暴虐の王を除かなければならぬと決意した。
    メロスには政治がわからぬ。
    メロスは、村の牧人である。
    笛を吹き、羊と遊んで暮して来た。
    けれども邪悪に対しては、人一倍に敏感であった。
    """
    nlp = spacy.load("ja_ginza")
    ginza.set_split_mode(nlp, "B")

    doc = nlp(text)

    # 最初の文を抽出して係り受け木を可視化
    if doc.sents:
        first_sent = list(doc.sents)[0]
        svg = spacy.displacy.render(first_sent, style="dep", jupyter=False)
        with open("./output/ch4/33_out_dependency_tree.svg", "w", encoding="utf-8") as f:
            f.write(svg)
        print("係り受け木を svg として保存しました。")
    else:
        print("文が検出されませんでした。")


if __name__ == "__main__":
    main()
