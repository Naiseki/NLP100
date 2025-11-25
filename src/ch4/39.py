import sys
import gzip
import json
import re
import matplotlib.pyplot as plt
from typing import Dict, Optional, TextIO
import MeCab
from collections import Counter

def clean_wiki_markup(value: str) -> str:
    """MediaWiki マークアップを可能な限り除去"""
    v = value.strip()

    # 1. 強調マークアップ（''、'''、'''''）除去
    v = re.sub(r"'{2,5}", "", v)

    # 2. 内部リンク [[リンク]] / [[リンク|表示]] → 表示部分のみ
    def replace_link(match):
        inner = match.group(1)
        if '|' in inner:
            return inner.split('|')[1]
        return inner
    v = re.sub(r'\[\[([^\]]+)\]\]', replace_link, v)

    # 3. 外部リンク [URL 表示] → 表示部分のみ
    v = re.sub(r'\[https?://[^\s]+ ([^\]]+)\]', r'\1', v)

    # 4. テンプレート {{lang|en|...}} や {{仮リンク|…}} → 内部のテキストのみ
    v = re.sub(r'\{\{[^{}|]+\|[^{}]*\|([^{}]+)\}\}', r'\1', v)
    v = re.sub(r'\{\{[^{}]+\}\}', '', v)  # 残りのテンプレートは削除

    # 5. <ref>…</ref> や <br /> 等の HTML タグを削除
    v = re.sub(r'<ref[^>]*>.*?</ref>', '', v, flags=re.DOTALL)
    v = re.sub(r'<[^>]+>', '', v)

    # 6. 複数空白や改行を整理
    v = re.sub(r'\s+', ' ', v).strip()

    return v

def main():
    path = "input/jawiki-country.json.gz"
    corpus = []
    with gzip.open(path, "rb") as file:
        for line in file:
            article_dict = json.loads(line)
            cleaned_text = clean_wiki_markup(article_dict["text"])
            corpus.append(cleaned_text)
    
    # MeCab で形態素解析
    tagger = MeCab.Tagger("-Ochasen")

    words = []
    for cleaned_text in corpus:
        if not cleaned_text.strip():
            continue
        node = tagger.parseToNode(cleaned_text)
        while node:
            if node.surface and node.feature.split(",")[0] == "名詞":
                words.append(node.surface)
            node = node.next
    
    word_counts = Counter(words)
    # 頻度を降順にソート
    sorted_frequencies = sorted(word_counts.values(), reverse=True)
    ranks = range(1, len(sorted_frequencies) + 1)
    
    # 両対数グラフをプロット
    plt.loglog(ranks, sorted_frequencies)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf\"s Law")
    plt.savefig("./output/ch4/39_out_zipf_plot.png")
    plt.close()


if __name__ == "__main__":
    main()
