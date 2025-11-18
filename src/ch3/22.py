import sys
import gzip
import json
import re

def find_article(stream, target_title):
    for line in stream:
        article_dict = json.loads(line)
        if target_title in article_dict["title"]:
            return article_dict
    return None

def filter_category_lines(text):
    # [[Category:から始まり，|]以外の文字が続く部分をキャプチャ
    pattern = re.compile(r'\[\[Category:([^|\]]+)', re.MULTILINE)
    return pattern.findall(text)

def main():
    path = "input/jawiki-country.json.gz"
    with gzip.open(path, "rb") as file:
        target_title = "イギリス"
        article = find_article(file, target_title)
        if not article:
            print(f"'{target_title}'の記事は見つかりませんでした。")

        for line in filter_category_lines(article["text"]):
            print(line)


if __name__ == "__main__":
    main()
