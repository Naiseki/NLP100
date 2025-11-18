import sys
import gzip
import json

def find_article(stream, target_title):
    for line in stream:
        article_dict = json.loads(line)
        if target_title == article_dict["title"]:
            return article_dict
    return None

def main():
    path = "input/jawiki-country.json.gz"
    with gzip.open(path, "rb") as file:
        target_title = "イギリス"
        article = find_article(file, target_title)
        if article:
            print(article["text"])
        else:
            print(f"'{target_title}'の記事は見つかりませんでした。")


if __name__ == "__main__":
    main()
