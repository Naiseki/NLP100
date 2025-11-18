import sys
import gzip
import json
import re

def find_article(stream, target_title):
    for line in stream:
        article_dict = json.loads(line)
        if target_title == article_dict["title"]:
            return article_dict
    return None

def extract_files(text):
    pattern = re.compile(r"\[\[ファイル:([^|\]]+)", re.MULTILINE)
    return pattern.findall(text)

def main():
    path = "input/jawiki-country.json.gz"
    with gzip.open(path, "rb") as file:
        target_title = "イギリス"
        article = find_article(file, target_title)
        if not article:
            print(f"'{target_title}'の記事は見つかりませんでした。")
            return

        for filename in extract_files(article["text"]):
            print(filename)


if __name__ == "__main__":
    main()
