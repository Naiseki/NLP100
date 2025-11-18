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

def extract_base_info(text):
    base_info_pattern = re.compile(r"\{\{基礎情報.*?\n(.*?)\n\}\}", re.DOTALL)
    
    match = base_info_pattern.search(text)
    base_info_text = match.group(1) if match else ""

    field_pattern = re.compile(r"\|\s*([^=\n]+?)\s*=\s*(.*?)(?=\n\|\s*[^=\n]+?\s*=|\n?$)", re.DOTALL)
    fields = {}
    for key, value in field_pattern.findall(base_info_text):
        cleaned_value = re.sub(r"'{2,5}", "", value.strip())
        fields[key.strip()] = cleaned_value 
    return fields

def main():
    path = "input/jawiki-country.json.gz"
    with gzip.open(path, "rb") as file:
        target_title = "イギリス"
        article = find_article(file, target_title)
        if not article:
            print(f"'{target_title}'の記事は見つかりませんでした。")
            return

        for key, value in extract_base_info(article["text"]).items():
            print(f"{key}: {value}")



if __name__ == "__main__":
    main()
