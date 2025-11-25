import sys
import gzip
import json
import re
from typing import Dict, Optional, TextIO
import MeCab
from collections import Counter

def find_article(stream: TextIO, target_title: str) -> Optional[Dict[str, str]]:
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
    corpus = []
    with gzip.open(path, "rb") as file:
        for line in file:
            article_dict = json.loads(line)
            cleaned_text = clean_wiki_markup(article_dict["text"])
            corpus.append(cleaned_text)
    
    full_corpus = " ".join(corpus)
    # MeCab で形態素解析
    try:
        # Taggerの初期化はループの外で一度だけ行う
        tagger = MeCab.Tagger("-Ochasen")
    except Exception as e:
        print("MeCab の初期化に失敗しました:", e, file=sys.stderr)
        return

    words = []
    for cleaned_text in corpus: # full_corpus ではなく、corpusリストをループ
        if not cleaned_text.strip():
            continue # 空のテキストはスキップ
            
        try:
            # 記事ごとの解析
            node = tagger.parseToNode(cleaned_text) 
        except Exception as e:
            # 個々の記事の解析に失敗しても全体は止めない
            print(f"警告: 記事の形態素解析に失敗しました: {e}", file=sys.stderr)
            continue
            
        while node:
            if node.surface and node.feature: # node.featureのチェックも追加
                features = node.feature.split(',')
                # featuresが期待通りに分割されない可能性も考慮して長さチェック
                if len(features) > 0:
                    pos = features[0]
                    # 主要な品詞を対象
                    if pos in ["名詞", "動詞", "形容詞", "副詞"]:
                        # 語幹や原型（features[6]）を使うと精度が上がるけど、ここではnode.surfaceのままで
                        words.append(node.surface)
            node = node.next
    
    
    word_counts = Counter(words)
    print("出現頻度の高い20語:")
    for word, count in word_counts.most_common(20):
        print(f"{word}: {count}")


if __name__ == "__main__":
    main()
