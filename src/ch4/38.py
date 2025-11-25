import sys
import gzip
import json
import re
import math
from typing import Dict, Optional, TextIO
import MeCab
from collections import Counter

def find_article(stream, target_title):
    for line in stream:
        article_dict = json.loads(line)
        if target_title == article_dict["title"]:
            return article_dict
    return None

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
    
    # 全記事の数Nと各名詞のDF（出現記事数）を計算
    N = 0
    df_counter = Counter()
    with gzip.open(path, "rb") as file:
        # 1行につき1記事
        for line in file:
            article_dict = json.loads(line)
            N += 1
            cleaned_text = clean_wiki_markup(article_dict["text"])
            if not cleaned_text.strip():
                continue
            tagger = MeCab.Tagger("-Ochasen")
            node = tagger.parseToNode(cleaned_text)
            article_words = set()
            while node:
                if node.surface and node.feature.split(',')[0] == "名詞":
                    article_words.add(node.surface)
                node = node.next
            for word in article_words:
                df_counter[word] += 1
    
    # 日本に関する記事を抽出
    with gzip.open(path, "rb") as file:
        japan_article = find_article(file, "日本")
    if not japan_article:
        print("日本に関する記事が見つかりません", file=sys.stderr)
        return
    
    cleaned_text = clean_wiki_markup(japan_article["text"])
    
    # 日本の記事でTF (日本の記事におけるその単語の出現回数)を計算
    tagger = MeCab.Tagger("-Ochasen")
    node = tagger.parseToNode(cleaned_text)
    tf_counter = Counter()
    while node:
        if node.surface and node.feature.split(',')[0] == "名詞":
            tf_counter[node.surface] += 1
        node = node.next
    
    # TF-IDF計算
    tfidf_scores = {}
    for word, tf in tf_counter.items():
        # 含まれていなかった場合は0を返す
        df = df_counter.get(word, 0)
        if df > 0:
            idf = math.log(N / df)
            tfidf = tf * idf
            tfidf_scores[word] = (tf, idf, tfidf)
    
    # 上位20語を表示
    sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1][2], reverse=True)[:20]
    print("TF-IDFスコア上位20語:")
    for word, (tf, idf, tfidf) in sorted_tfidf:
        print(f"{word}: TF={tf}, IDF={idf:.2f}, TF-IDF={tfidf:.2f}")


if __name__ == "__main__":
    main()
