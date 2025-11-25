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

    # 「メロス」が主語である場合の述語（動詞）を抽出
    subject = "メロス"
    predicates = []
    for token in doc:
        # nsubj: 能動態の主語（メロスが〜する）
        # nsubj:pass: 受動態の主語（メロスが〜される）
        if token.text == subject and token.dep_ in ["nsubj", "nsubj:pass"]:
            head = token.head
            if head.pos_ == "VERB":
                predicates.append(head.text)

    print(f"「{subject}」が主語であるときの述語:")
    if predicates:
        # setで重複を除く
        for pred in set(predicates):  
            print(pred)
    else:
        print("無しです。")


if __name__ == "__main__":
    main()
