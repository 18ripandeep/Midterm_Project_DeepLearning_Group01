"""
eval_baseline.py
Generates validation tuning results, test metrics, and confusion matrix for Baseline.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from data_pipeline import prepare_dataset_pipeline

loaders, vocab, meta, base_data = prepare_dataset_pipeline("snips", seed=42)
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf.fit(base_data["X_train"], base_data["y_train"])
preds = clf.predict(base_data["X_test"])

print("=" * 60)
print("BASELNE (LOGISTIC REGRESSION) REPORT - SNIPS")
print("=" * 60)
print(classification_report(base_data["y_test"], preds, target_names=meta["classes"]))

cm = confusion_matrix(base_data["y_test"], preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=meta["classes"], yticklabels=meta["classes"])
plt.title("Baseline Confusion Matrix (Seed 42)")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.tight_layout()
plt.savefig("artifacts/baseline_confusion_matrix.png", dpi=300)
plt.close()