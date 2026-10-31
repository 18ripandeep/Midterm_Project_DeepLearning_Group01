"""
main.py
Midterm Project.
Executes Baseline, CNN, RNN, and Transformer + Ablations over Seeds 42, 123, 999.
Exports unified experimental artifacts to artifacts/results.json.
"""

import copy
import json
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data_pipeline import prepare_dataset_pipeline
from models import TextCNN, TextRNN, TextTransformer

SEEDS = [42, 123, 999]
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.001

# Force CPU execution on Mac to avoid unsupported MPS nested tensor operators
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model: nn.Module, loader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for b in loader:
            x, y = b["input_ids"].to(device), b["label"].to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            targets.extend(y.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        accuracy_score(targets, preds),
        f1_score(targets, preds, average="macro", zero_division=0)
    )


def profile_inference_cost(model: nn.Module, batch, device: torch.device, reps: int = 100) -> Dict:
    model.eval()
    x = batch["input_ids"].to(device)
    b_size = x.size(0)
    params = count_parameters(model)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(10): _ = model(x)
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            for _ in range(reps): _ = model(x)
        end_event.record()
        torch.cuda.synchronize()
        latency = start_event.elapsed_time(end_event) / (reps * b_size)
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        with torch.no_grad():
            for _ in range(5): _ = model(x)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(reps): _ = model(x)
        latency = ((time.perf_counter() - t0) * 1000) / (reps * b_size)
        peak_vram = 0.0

    return {"params": params, "latency_ms": round(latency, 4), "peak_vram_mb": round(peak_vram, 2)}


def train_deep_model(model: nn.Module, train_loader, val_loader, test_loader) -> Tuple[Dict, List[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_f1, best_weights = -1.0, None
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for b in train_loader:
            x, y = b["input_ids"].to(DEVICE), b["label"].to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        loss_history.append(avg_train_loss)
        _, _, v_f1 = evaluate(model, val_loader, DEVICE)
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    _, t_acc, t_f1 = evaluate(model, test_loader, DEVICE)
    return {"val_f1": best_val_f1, "test_acc": t_acc, "test_f1": t_f1}, loss_history


def tune_baseline(baseline_data: dict, seed: int) -> Dict:
    """
    Tuning the baseline Logistic Regression with stronger regularization 
    to create a fair, defensible floor (preventing artificial over-parameterization).
    """
    best_c, best_val_f1 = 0.01, -1.0
    
    # Test tighter regularization values (smaller C = stronger regularization)
    for c in [0.0001, 0.001, 0.01, 0.1]:
        clf = LogisticRegression(
            C=c, 
            penalty='l2', 
            solver='lbfgs', 
            max_iter=1000, 
            random_state=seed
        )
        clf.fit(baseline_data["X_train"], baseline_data["y_train"])
        val_preds = clf.predict(baseline_data["X_val"])
        val_f1 = f1_score(baseline_data["y_val"], val_preds, average="macro", zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1, best_c = val_f1, c

    # Train final baseline with the best tuned C parameter found on validation set
    final_clf = LogisticRegression(
        C=best_c, 
        penalty='l2', 
        solver='lbfgs', 
        max_iter=1000, 
        random_state=seed
    )
    final_clf.fit(baseline_data["X_train"], baseline_data["y_train"])
    test_preds = final_clf.predict(baseline_data["X_test"])
    
    return {
        "val_f1": best_val_f1,
        "test_acc": accuracy_score(baseline_data["y_test"], test_preds),
        "test_f1": f1_score(baseline_data["y_test"], test_preds, average="macro", zero_division=0)
    }


def run_full_suite():
    os.makedirs("artifacts", exist_ok=True)
    benchmark_results = {
        "SNIPS": {
            "Baseline": {"seeds": {}},
            "CNN_Standard": {"seeds": {}, "loss_curves": {}},
            "CNN_NoBN": {"seeds": {}, "loss_curves": {}},
            "RNN_BiLSTM": {"seeds": {}, "loss_curves": {}},
            "RNN_UniLSTM": {"seeds": {}, "loss_curves": {}},
            "Transformer_Standard": {"seeds": {}, "loss_curves": {}},
            "Transformer_NoPos": {"seeds": {}, "loss_curves": {}},
            "costs": {}
        },
        "ATIS": {
            "Baseline": {"seeds": {}},
            "CNN_Standard": {"seeds": {}},
            "RNN_BiLSTM": {"seeds": {}},
            "Transformer_Standard": {"seeds": {}}
        }
    }

    # 1. Primary Dataset (SNIPS) Experiments
    print("\n========== Running Primary Dataset: SNIPS ==========")
    for seed in SEEDS:
        print(f"\n--- SEED {seed} ---")
        set_seed(seed)
        loaders, vocab, meta, base_data = prepare_dataset_pipeline("snips", seed=seed, run_eda=(seed == 42))
        tr, va, te = loaders
        v_size, n_cls = len(vocab), meta["num_classes"]

        # Baseline
        print("Training Baseline (Logistic Regression)...")
        benchmark_results["SNIPS"]["Baseline"]["seeds"][seed] = tune_baseline(base_data, seed)

        # CNN Variants
        print("Training CNN (Standard)...")
        cnn_std = TextCNN(v_size, num_classes=n_cls, use_batch_norm=True).to(DEVICE)
        res, l_hist = train_deep_model(cnn_std, tr, va, te)
        benchmark_results["SNIPS"]["CNN_Standard"]["seeds"][seed] = res
        benchmark_results["SNIPS"]["CNN_Standard"]["loss_curves"][seed] = l_hist

        print("Training CNN (Ablation: No BatchNorm)...")
        cnn_nobn = TextCNN(v_size, num_classes=n_cls, use_batch_norm=False).to(DEVICE)
        res, l_hist = train_deep_model(cnn_nobn, tr, va, te)
        benchmark_results["SNIPS"]["CNN_NoBN"]["seeds"][seed] = res
        benchmark_results["SNIPS"]["CNN_NoBN"]["loss_curves"][seed] = l_hist

        # RNN Variants
        print("Training RNN (Bi-LSTM)...")
        rnn_bi = TextRNN(v_size, num_classes=n_cls, bidirectional=True).to(DEVICE)
        res, l_hist = train_deep_model(rnn_bi, tr, va, te)
        benchmark_results["SNIPS"]["RNN_BiLSTM"]["seeds"][seed] = res
        benchmark_results["SNIPS"]["RNN_BiLSTM"]["loss_curves"][seed] = l_hist

        print("Training RNN (Ablation: Uni-LSTM)...")
        rnn_uni = TextRNN(v_size, num_classes=n_cls, bidirectional=False).to(DEVICE)
        res, l_hist = train_deep_model(rnn_uni, tr, va, te)
        benchmark_results["SNIPS"]["RNN_UniLSTM"]["seeds"][seed] = res
        benchmark_results["SNIPS"]["RNN_UniLSTM"]["loss_curves"][seed] = l_hist

        # Transformer Variants
        print("Training Transformer (Standard)...")
        tf_std = TextTransformer(v_size, num_classes=n_cls, use_positional=True).to(DEVICE)
        res, l_hist = train_deep_model(tf_std, tr, va, te)
        benchmark_results["SNIPS"]["Transformer_Standard"]["seeds"][seed] = res
        benchmark_results["SNIPS"]["Transformer_Standard"]["loss_curves"][seed] = l_hist

        print("Training Transformer (Ablation: No Positional)...")
        tf_nopos = TextTransformer(v_size, num_classes=n_cls, use_positional=False).to(DEVICE)
        res, l_hist = train_deep_model(tf_nopos, tr, va, te)
        benchmark_results["SNIPS"]["Transformer_NoPos"]["seeds"][seed] = res
        benchmark_results["SNIPS"]["Transformer_NoPos"]["loss_curves"][seed] = l_hist

        # Inference Profiling once
        if seed == 42:
            sample_b = next(iter(te))
            benchmark_results["SNIPS"]["costs"]["CNN_Standard"] = profile_inference_cost(cnn_std, sample_b, DEVICE)
            benchmark_results["SNIPS"]["costs"]["RNN_BiLSTM"] = profile_inference_cost(rnn_bi, sample_b, DEVICE)
            benchmark_results["SNIPS"]["costs"]["Transformer_Standard"] = profile_inference_cost(tf_std, sample_b, DEVICE)

    # 2. Secondary Dataset (ATIS) Check
    print("\n========== Running Secondary Dataset Check: ATIS ==========")
    for seed in SEEDS:
        print(f"--- ATIS SEED {seed} ---")
        set_seed(seed)
        loaders, vocab, meta, base_data = prepare_dataset_pipeline("atis", seed=seed, run_eda=(seed == 42))
        tr, va, te = loaders
        v_size, n_cls = len(vocab), meta["num_classes"]

        benchmark_results["ATIS"]["Baseline"]["seeds"][seed] = tune_baseline(base_data, seed)
        
        cnn = TextCNN(v_size, num_classes=n_cls, use_batch_norm=True).to(DEVICE)
        benchmark_results["ATIS"]["CNN_Standard"]["seeds"][seed], _ = train_deep_model(cnn, tr, va, te)

        rnn = TextRNN(v_size, num_classes=n_cls, bidirectional=True).to(DEVICE)
        benchmark_results["ATIS"]["RNN_BiLSTM"]["seeds"][seed], _ = train_deep_model(rnn, tr, va, te)

        tf = TextTransformer(v_size, num_classes=n_cls, use_positional=True).to(DEVICE)
        benchmark_results["ATIS"]["Transformer_Standard"]["seeds"][seed], _ = train_deep_model(tf, tr, va, te)

    with open("artifacts/results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print("\nBenchmark successfully complete. All outputs saved to artifacts/results.json")


if __name__ == "__main__":
    run_full_suite()
