import ast
import os
import argparse
from typing import List, Dict, Any, Tuple, Optional
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_bow_file(path: str) -> List[Dict[str, Any]]:
	"""1行ごとに辞書を持つファイルを読み込む"""
	items = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			try:
				obj = ast.literal_eval(line)
				if isinstance(obj, dict):
					items.append(obj)
			except Exception:
				# 不正行は無視
				continue
	return items


def eval(model: any, vec: any, dataset_path: str):
	items = load_bow_file(dataset_path)

	X_dicts = [it["feature"] for it in items]
	y_true = [int(it["label"]) for it in items]

	X = vec.transform(X_dicts)
	y_pred = model.predict(X)
	classes = getattr(model, "classes_", None)

	acc = accuracy_score(y_true, y_pred)
	print("Classification report:")
	print(classification_report(y_true, y_pred, labels=classes))

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
	print("学習データでの評価:")
	eval(model, vec, "output/ch7/61_train_bow.txt")
	print("検証データでの評価:")
	eval(model, vec, "output/ch7/61_dev_bow.txt")


if __name__ == "__main__":
	main()
