import ast
import os
from typing import List, Dict, Any
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


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


def train(train_path: str, model_out: str) -> None:
	train_items = load_bow_file(train_path)

	X_train_dicts = [it["feature"] for it in train_items]
	y_train = [int(it["label"]) for it in train_items]

	vec = DictVectorizer(sparse=True)
	X_train = vec.fit_transform(X_train_dicts)

	model = LogisticRegression(solver="liblinear", max_iter=1000)
	model.fit(X_train, y_train)

	# モデル＋ベクトライザ保存
	joblib.dump({"model": model, "vectorizer": vec}, model_out)

def main():
    train_path = "output/ch7/61_train_bow.txt"
    model_out_path = "output/ch7/62_logreg.joblib"
    train(train_path, model_out_path)

if __name__ == "__main__":
	main()
