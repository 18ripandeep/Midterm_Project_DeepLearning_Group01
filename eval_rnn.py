"""
eval_rnn.py
Prints seed results, ablation comparison, and generates confusion matrix for RNN.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from data_pipeline import prepare_dataset_pipeline
from models import TextRNN

with open("artifacts/results.json") as f:
    data = json.load(f)["SNIPS"]

print("=" * 65)
print("RECURRENT NETWORK EVALUATION & ABLATION (REPORT SECTIONS 4 & 5)")
print("=" * 65)
for mod in ["RNN_BiLSTM", "RNN_UniLSTM"]:
    f1s = [data[mod]["seeds"][str(s)]["test_f1"] for s in [42, 123, 999]]
    print(f"{mod:<22} | Seed 42: {f1s[0]:.4f} | Seed 123: {f1s[1]:.4f} | Seed 999: {f1s[2]:.4f} | Mean: {np.mean(f1s):.4f} ± {np.ptp(f1s):.4f}")

loaders, vocab, meta, _ = prepare_dataset_pipeline("snips", seed=42)
_, _, test_loader = loaders
model = TextRNN(len(vocab), num_classes=meta["num_classes"], bidirectional=True)
preds, labels = [], []
with torch.no_grad():
    for b in test_loader:
        preds.extend(torch.argmax(model(b["input_ids"]), dim=1).cpu().numpy())
        labels.extend(b["label"].cpu().numpy())

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=meta["classes"], yticklabels=meta["classes"])
plt.title("Bi-LSTM Confusion Matrix (Seed 42)")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.tight_layout()
plt.savefig("artifacts/rnn_confusion_matrix.png", dpi=300)
plt.close()