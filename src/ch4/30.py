import sys
import gzip
import json
from sudachipy import dictionary, tokenizer
from collections import Counter

def main():
    text = """
    メロスは激怒した。
    必ず、かの邪智暴虐の王を除かなければならぬと決意した。
    メロスには政治がわからぬ。
    メロスは、村の牧人である。
    笛を吹き、羊と遊んで暮して来た。
    けれども邪悪に対しては、人一倍に敏感であった。
    """

    tokenizer_obj = dictionary.Dictionary().create()

    tokens = tokenizer_obj.tokenize(text, tokenizer.Tokenizer.SplitMode.B)
    verbs = []
    for t in tokens:
        pos = t.part_of_speech()
        if pos[0] == "動詞":
            verbs.append(t.surface())
    
    print(verbs)


if __name__ == "__main__":
    main()
