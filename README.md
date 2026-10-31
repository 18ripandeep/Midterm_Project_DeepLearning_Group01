# AIGC 5500 Midterm Project: Comparing Convolutional, Recurrent, and Attention-Based Architectures

**Course:** AIGC 5500 — Advanced Deep Learning  
**Project:** Midterm Project: Architecture Comparison on Short Text Classification  
**Group:** Group 01  
**Primary Dataset:** SNIPS Voice Assistant Intent Dataset (7 Classes)  
**Secondary Dataset:** ATIS Airline Travel Information System (Flight Booking Intents)  

---

## 1. Research Question

> **How do the inductive biases of Convolutional (1D CNN), Recurrent (Bi-LSTM), and Attention-Based (Transformer Encoder) architectures affect classification performance, convergence, and inference efficiency on short text intent classification when trained strictly from scratch under identical budgets?**

We benchmark all three deep neural network families against a defensible unigram Bag-of-Words Logistic Regression baseline across three distinct seeds (`42`, `123`, `999`), conduct one controlled ablation per family, and test whether architectural rankings generalize across domains to a second dataset (ATIS).

---

## 2. Datasets & Access

1. **Primary Dataset: SNIPS Voice Assistant**
   * **Domain:** Spoken virtual assistant commands (*PlayMusic, BookRestaurant, GetWeather, RateBook, SearchScreeningEvent, AddToPlaylist, SearchCreativeWork*).
   * **Size:** ~14,400 samples split into 70% Train, 15% Validation, and 15% Test.
   * **Sequence Length:** Short command utterances (~9 words on average).
   * **Source:** Extracted dynamically via `load_dataset("benayas/snips")`.
2. **Secondary Dataset: ATIS**
   * **Domain:** Specialized airline travel and reservation requests.
   * **Size:** ~5,800 samples split into identical 70/15/15 stratified partitions.
   * **Source:** Extracted dynamically via `load_dataset("tuetschek/atis")`.

---

## 3. Project Architecture & File Structure

```text
Midterm_Project_DeepLearning_Group01/
├── data_pipeline.py      # Data extraction, EDA plots, tokenization, train-only vocab, DataLoaders
├── models.py             # PyTorch models: 1D CNN, Bi-LSTM, Transformer with ablation toggles
├── main.py               # Master experiment orchestrator: runs all 33 seed/ablation runs
├── eval_baseline.py      # Baseline evaluation, hyperparameter tuning & confusion matrix
├── eval_cnn.py           # CNN evaluation, ablation metrics & confusion matrix
├── eval_rnn.py           # Recurrent (Bi-LSTM) evaluation, ablation metrics & confusion matrix
├── eval_transformer.py   # Transformer evaluation, ablation metrics & confusion matrix
├── compare.py            # Consolidated comparison tables & publication figures (Figures 1, 2, 3)
├── requirements.txt      # Pinned environment dependencies
└── artifacts/            # Generated outputs, JSON logs, confusion matrices, and figures
```

---

## 4. How to Reproduce All Results

### Step 1: Environment Setup
Clone the repository and set up a fresh virtual environment:
```bash
git clone https://github.com/your-username/Midterm_Project_DeepLearning_Group01.git
cd Midterm_Project_DeepLearning_Group01
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run Full Benchmark Pipeline
Execute the full benchmark suite across all seeds (`42`, `123`, `999`):
```bash
python main.py
```
*This trains the baseline, the three deep models, and all ablations on both SNIPS and ATIS, automatically exporting raw metrics to `artifacts/results.json`.*

### Step 3: Generate Report Tables & Figures
```bash
python compare.py
```
*Prints the exact numerical tables for Sections 4, 5, 6, and 7 of the report and saves Figures 1, 2, and 3 into `artifacts/`.*

### Step 4: Generate Individual Confusion Matrices
```bash
python eval_baseline.py
python eval_cnn.py
python eval_rnn.py
python eval_transformer.py
```

---

## 5. Summary of Experimental Results

### Core Comparison (Primary Dataset: SNIPS)
*Decisions guided by validation performance; evaluated once on the test split.*

| Architecture | Seed 42 | Seed 123 | Seed 999 | Mean Macro F1 | Spread |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Reg)** | 0.9234 | 0.9211 | 0.9275 | **0.9240** | 0.0064 |
| **Convolutional (1D CNN)** | 0.9807 | 0.9796 | 0.9834 | **0.9812** | 0.0038 |
| **Recurrent (Bi-LSTM)** | 0.9785 | 0.9791 | 0.9801 | **0.9792** | 0.0016 |
| **Attention (Transformer)** | 0.9783 | 0.9742 | 0.9752 | **0.9759** | 0.0041 |

### Ablation Studies (SNIPS)
*Each ablation isolates and modifies exactly one architectural component.*

| Family | Component Tested | Standard Macro F1 | Ablated Macro F1 | Delta (Δ) | Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Convolutional** | Batch Normalization | 0.9812 | 0.9780 | -0.0032 | BatchNorm accelerates training stability and regularizes feature activations. |
| **Recurrent** | Bidirectionality | 0.9792 | 0.9695 | -0.0097 | Backward context significantly aids in intent disambiguation for early command verbs. |
| **Attention** | Positional Encoding | 0.9759 | 0.9680 | -0.0079 | Without position encodings, self-attention degenerates into an un-ordered set function. |

### Operational Inference Cost
*Measured under fixed test batch conditions.*

| Architecture | Total Parameters | Latency / Sample (ms) | Peak VRAM / Memory |
| :--- | :---: | :---: | :---: |
| **1D CNN** | 273,223 | 0.138 ms | Minimal |
| **Bi-LSTM** | 301,895 | 0.086 ms | Minimal |
| **Transformer Encoder** | 301,831 | 0.039 ms | Minimal |

---

## 6. Video Presentation

* **Link:** [OneDrive Video Presentation Link](https://humber-my.sharepoint.com/personal/your-link-here)
* **Presenter:** Ankit Rai
* **Duration:** 17 minutes 12 seconds
* **Access Permissions:** Humber-wide viewable link + directly shared with `hossein.pourmodheji@humber.ca`.

---

## 7. Group Contribution Table

| Member Name | Humber Student ID | Assigned Piece & Role | Key Responsibilities |
| :--- | :--- | :--- | :--- |
| **Member 1** | N01234567 | Baseline & Shared Evaluation | Data pipeline, stratified splitting, TF-IDF baseline tuning, profiling metrics |
| **Member 2** | N01234568 | Convolutional Model (1D CNN) | TextCNN implementation, BatchNorm ablation, kernel analysis |
| **Member 3** | N01234569 | Recurrent Model (LSTM) | Bi-LSTM architecture, masked pooling fix, unidirectionality ablation |
| **Member 4** | N01234570 | Attention Model (Transformer) | Transformer Encoder, padding masks, positional encoding ablation |
| **Member 5** | N01234571 | Second Dataset Check (ATIS) | ATIS pipeline, cross-dataset ranking analysis, report consolidation |

---

## 8. Academic Integrity & References

1. Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. EMNLP 2014.
2. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
3. Coucke, A., et al. (2018). *Snips Voice Platform: An embedded Spoken Language Understanding system*. arXiv:1805.10190.
4. Hemphill, C. T., et al. (1990). *The ATIS spoken language systems pilot corpus*. DARPA Speech and Natural Language Workshop.