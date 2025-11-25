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

    for token in doc:
        if token.head != token:
            print(f"{token.text}\t{token.head.text}")

if __name__ == "__main__":
    main()
