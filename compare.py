"""
compare.py
Generates consolidated report tables and publication-grade comparison figures.
Directly produces deliverables for Report Sections 4, 5, 6, and 7.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/results.json") as f:
    res = json.load(f)

snips = res["SNIPS"]
atis = res["ATIS"]
SEEDS = ["42", "123", "999"]


# -------------------------------------------------------------------------
# 1. Print Report Section Tables
# -------------------------------------------------------------------------
print("\n" + "=" * 85)
print("TABLE 1: REPORT SECTION 4 - CORE COMPARISON (SNIPS)")
print("=" * 85)
print(f"{'Architecture':<24} | {'Seed 42':<8} | {'Seed 123':<8} | {'Seed 999':<8} | {'Mean Macro F1':<13} | {'Spread':<8}")
print("-" * 85)
core_models = ["Baseline", "CNN_Standard", "RNN_BiLSTM", "Transformer_Standard"]
for m in core_models:
    vals = [snips[m]["seeds"][s]["test_f1"] for s in SEEDS]
    print(f"{m:<24} | {vals[0]:.4f}   | {vals[1]:.4f}   | {vals[2]:.4f}   | {np.mean(vals):.4f}        | {np.ptp(vals):.4f}")

print("\n" + "=" * 85)
print("TABLE 2: REPORT SECTION 5 - ABLATION STUDIES (SNIPS)")
print("=" * 85)
print(f"{'Family':<15} | {'Component Tested':<22} | {'Standard F1':<12} | {'Ablated F1':<12} | {'Delta Δ':<8}")
print("-" * 85)
ablations = [
    ("Convolutional", "Batch Normalization", "CNN_Standard", "CNN_NoBN"),
    ("Recurrent",     "Bidirectionality",   "RNN_BiLSTM",   "RNN_UniLSTM"),
    ("Attention",     "Positional Encoding", "Transformer_Standard", "Transformer_NoPos")
]
for fam, comp, std_k, abl_k in ablations:
    std_f1 = np.mean([snips[std_k]["seeds"][s]["test_f1"] for s in SEEDS])
    abl_f1 = np.mean([snips[abl_k]["seeds"][s]["test_f1"] for s in SEEDS])
    print(f"{fam:<15} | {comp:<22} | {std_f1:<12.4f} | {abl_f1:<12.4f} | {(abl_f1 - std_f1):+.4f}")

print("\n" + "=" * 85)
print("TABLE 3: REPORT SECTION 6 - SECOND DATASET GENERALIZATION (ATIS)")
print("=" * 85)
print(f"{'Architecture':<24} | {'SNIPS Mean F1':<15} | {'ATIS Mean F1':<15} | {'Ranking Preserved?'}")
print("-" * 85)
snips_means = {m: np.mean([snips[m]["seeds"][s]["test_f1"] for s in SEEDS]) for m in core_models}
atis_means = {m: np.mean([atis[m]["seeds"][s]["test_f1"] for s in SEEDS]) for m in core_models}
snips_rank = sorted(snips_means, key=snips_means.get, reverse=True)
atis_rank = sorted(atis_means, key=atis_means.get, reverse=True)
preserved = (snips_rank == atis_rank)
for m in core_models:
    print(f"{m:<24} | {snips_means[m]:<15.4f} | {atis_means[m]:<15.4f} | {'Rank: ' + str(atis_rank.index(m) + 1)}")
print(f"Overall Inductive Bias Ranking Generalization: {'CONFIRMED' if preserved else 'REVERSED / MODIFIED'}")

print("\n" + "=" * 85)
print("TABLE 4: REPORT SECTION 7 - OPERATIONAL INFERENCE COSTS")
print("=" * 85)
print(f"{'Architecture':<24} | {'Parameters':<12} | {'Latency / Sample (ms)':<22}")
print("-" * 85)
for m, c in snips["costs"].items():
    print(f"{m:<24} | {c['params']:<12,d} | {c['latency_ms']:<22}")


# -------------------------------------------------------------------------
# 2. Publication Figures for Report and Slides
# -------------------------------------------------------------------------
# Figure 1: Training Loss Convergence
plt.figure(figsize=(9, 5))
for m, color in [("CNN_Standard", "#1b9e77"), ("RNN_BiLSTM", "#7570b3"), ("Transformer_Standard", "#e7298a")]:
    loss_curve = snips[m]["loss_curves"]["42"]
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, marker="o", label=m, color=color, linewidth=2)
plt.title("Figure 1: Training Loss Convergence across Architecture Families (Seed 42)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/figure1_loss_convergence.png", dpi=300)
plt.close()

# Figure 2: Generalization Comparison across Datasets
x = np.arange(len(core_models))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, [snips_means[m] for m in core_models], width, label="Primary: SNIPS", color="#2b5c8f")
ax.bar(x + width/2, [atis_means[m] for m in core_models], width, label="Secondary: ATIS", color="#e66101")
ax.set_ylabel("Macro F1-Score")
ax.set_title("Figure 2: Performance Generalization across Datasets")
ax.set_xticks(x)
ax.set_xticklabels(core_models)
ax.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("artifacts/figure2_dataset_generalization.png", dpi=300)
plt.close()

# Figure 3: Cost-Accuracy Pareto Frontier
plt.figure(figsize=(8, 6))
costs = snips["costs"]
for m, color in [("CNN_Standard", "#1b9e77"), ("RNN_BiLSTM", "#7570b3"), ("Transformer_Standard", "#e7298a")]:
    lat = costs[m]["latency_ms"]
    f1 = snips_means[m]
    size = costs[m]["params"] / 200
    plt.scatter(lat, f1, s=size, label=f"{m} ({costs[m]['params']:,} params)", color=color, alpha=0.7)
    plt.annotate(m, (lat + 0.005, f1), fontsize=10)
plt.title("Figure 3: Inference Latency vs. Macro F1 Pareto Frontier")
plt.xlabel("Inference Latency per Sample (ms)")
plt.ylabel("Mean Test Macro F1")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(scatterpoints=1, loc="lower right")
plt.tight_layout()
plt.savefig("artifacts/figure3_pareto_frontier.png", dpi=300)
plt.close()