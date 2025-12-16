import ast
import os
from typing import List, Dict, Any
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


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


def train(train_path: str, C: float = 1.0):
	train_items = load_bow_file(train_path)
	X_train_dicts = [it["feature"] for it in train_items]
	y_train = [int(it["label"]) for it in train_items]
	vec = DictVectorizer(sparse=True)
	X_train = vec.fit_transform(X_train_dicts)
	model = LogisticRegression(solver="liblinear", max_iter=1000, C=C)
	model.fit(X_train, y_train)
	return model, vec

def eval(model, vec, dev_path: str):
	dev_items = load_bow_file(dev_path)
	X_dev_dicts = [it["feature"] for it in dev_items]
	y_dev = [int(it["label"]) for it in dev_items]
	X_dev = vec.transform(X_dev_dicts)
	y_pred = model.predict(X_dev)
	acc = accuracy_score(y_dev, y_pred)
	return acc

def main():
	train_path = "output/ch7/61_train_bow.txt"
	dev_path = "output/ch7/61_dev_bow.txt"
	C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
	acc_list = []
	for C in C_list:
		model, vec = train(train_path, C)
		acc = eval(model, vec, dev_path)
		acc_list.append(acc)
		print(f"C={C}: accuracy={acc:.4f}")
	# グラフ描画
	plt.figure()
	plt.plot(C_list, acc_list, marker='o')
	plt.xscale('log')
	plt.xlabel('Regularization parameter C')
	plt.ylabel('Accuracy on dev set')
	plt.title('Effect of regularization (C) on accuracy')
	plt.grid(True)
	plt.savefig("output/ch7/69_c_vs_acc.png")


if __name__ == "__main__":
	main()
