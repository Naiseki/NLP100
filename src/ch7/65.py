import ast
import os
import argparse
from typing import List, Dict, Any, Tuple, Optional
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# テキストをBoWに変換
def text_to_bow(text: str) -> Dict[str, int]:
    tokens = text.split()
    bow = {}
    for t in tokens:
        bow[t] = bow.get(t, 0) + 1
    return bow

# 与えられたテキストのポジネガを予測する
def pred(text: str, model: Any, vec: Any) -> int:
	feat = text_to_bow(text)
	X = vec.transform([feat])
	return model.predict(X)[0]


# モデル＋ベクトライザ読み込み
def load_model(path: str = "output/ch7/62_logreg.joblib") -> Tuple[Any, Any]:
	"""モデル＋ベクトライザ読み込み"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
	data = joblib.load(path)
	# {"model": model, "vectorizer": vec}形式で保存されている場合
	if isinstance(data, dict) and "model" in data and "vectorizer" in data:
		return data["model"], data["vectorizer"]
	raise ValueError("サポートされていないモデルファイル形式です。")


def main():
	model, vec = load_model()
	text = "the worst movie I ‘ve ever seen"
	prediction = pred(text, model, vec)
	print(f"テキスト: {text}")
	print(f"予測ラベル: {prediction}")


if __name__ == "__main__":
	main()
