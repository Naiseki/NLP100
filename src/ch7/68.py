import ast
import os
import argparse
from typing import List, Dict, Any, Tuple, Optional
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


# 1行ごとに辞書を持つファイルを読み込む
def load_bow_file(path: str) -> List[Dict[str, Any]]:
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


# モデルの重みの上位/下位特徴量を表示する
def show_top_features(model: BaseEstimator, vec: DictVectorizer, top_n: int = 20):
	feat_names = vec.get_feature_names_out()

	coefs = getattr(model, "coef_", None)
	if coefs is None:
		print("モデルに coef_ がありません。")
		return

	if coefs.ndim > 1 and coefs.shape[0] != 1:
		raise ValueError("このコードは二値分類のみをサポートします。モデルは多クラスです。")

	w = coefs.ravel()
	idx_top = np.argsort(w)[-top_n:][::-1]
	idx_bot = np.argsort(w)[:top_n]

	print(f"重みの高い特徴量トップ{top_n}:")
	for i in idx_top:
		name = feat_names[i] if i < len(feat_names) else f"f{i}"
		print(f"{name}\t{w[i]:.6f}")

	print(f"\n重みの低い特徴量トップ{top_n}:")
	for i in idx_bot:
		name = feat_names[i] if i < len(feat_names) else f"f{i}"
		print(f"{name}\t{w[i]:.6f}")


# モデル＋ベクトライザ読み込み
def load_model(path: str = "output/ch7/62_logreg.joblib") -> Tuple[BaseEstimator, DictVectorizer]:
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
	# 重みの上位/下位特徴量を表示
	show_top_features(model, vec, top_n=20)


if __name__ == "__main__":
	main()
