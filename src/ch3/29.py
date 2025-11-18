import sys
import gzip
import json
import re
import requests

def find_article(stream, target_title):
    for line in stream:
        article_dict = json.loads(line)
        if target_title == article_dict["title"]:
            return article_dict
    return None

def clean_wiki_markup(value):
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

def extract_base_info(text):
    base_info_pattern = re.compile(r"\{\{基礎情報.*?\n(.*?)\n\}\}", re.DOTALL)
    
    match = base_info_pattern.search(text)
    base_info_text = match.group(1) if match else ""

    field_pattern = re.compile(r"\|\s*([^=\n]+?)\s*=\s*(.*?)(?=\n\|\s*[^=\n]+?\s*=|\n?$)", re.DOTALL)
    fields = {}
    for key, value in field_pattern.findall(base_info_text):
        cleaned_value = clean_wiki_markup(value.strip())
        fields[key.strip()] = cleaned_value 
    return fields

def get_image_url(file_name):
    # MediaWiki API URL (ウィキペディア英語版の場合)
    API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

    # 1. ファイル名を 'File:' 形式に整形
    file_title = file_name
    if not file_title.lower().startswith("file:"):
        file_title = "File:" + file_title

    # 2. API パラメータを設定
    params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }

    headers = {"User-Agent": "NLP100"}
    response = requests.get(API_ENDPOINT, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    # 4. URL 抽出
    pages = data.get("query", {}).get("pages", {})
    image_url = None
    for page in pages.values():
        iinfo = page.get("imageinfo")
        if iinfo:
            image_url = iinfo[0]["url"]
            break

    return image_url


def main():
    path = "input/jawiki-country.json.gz"
    with gzip.open(path, "rb") as file:
        target_title = "イギリス"
        article = find_article(file, target_title)
        if not article:
            print(f"'{target_title}'の記事は見つかりませんでした。")
            return

        fields = extract_base_info(article["text"])
        print(get_image_url(fields["国旗画像"]))



if __name__ == "__main__":
    main()
