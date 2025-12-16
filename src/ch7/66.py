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


def eval(dev_path: str, pred_out: str):
	dev_items = load_bow_file(dev_path)

	X_dev_dicts = [it["feature"] for it in dev_items]
	y_dev = [int(it["label"]) for it in dev_items]

	model, vec = load_model()

	first = dev_items[0]
	feat = first.get("feature", {})
	X = vec.transform([feat])
	probs = model.predict_proba(X)[0]
	classes = getattr(model, "classes_", None)


	X_dev = vec.transform(X_dev_dicts)
	y_pred = model.predict(X_dev)

	cm = confusion_matrix(y_dev, y_pred, labels=classes)
	print("混同行列 (行: 真値, 列: 予測):")
	print(cm)


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
	eval("output/ch7/61_dev_bow.txt", "output/ch7/66_dev_pred.txt")


if __name__ == "__main__":
	main()
