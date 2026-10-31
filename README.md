# AIGC 5500 Midterm Project: Comparing Convolutional, Recurrent, and Attention-Based Architectures

**Course:** AIGC 5500 — Advanced Deep Learning  
**Project:** Midterm Project: Architecture Comparison on Short Text Classification  
**Group:** Group 01  
**Primary Dataset:** SNIPS Voice Assistant Intent Dataset (7 Classes)  
**Secondary Dataset:** ATIS Airline Travel Information System (Flight Booking Intents)  

---

## 1. Research Question

> **How do the inductive biases of Convolutional (1D CNN), Recurrent (Bi-LSTM), and Attention-Based (Transformer Encoder) architectures affect classification performance, convergence, and inference efficiency on short text intent classification when trained strictly from scratch under identical budgets?**

We evaluate all three deep neural network families against a defensible unigram Bag-of-Words Logistic Regression baseline across three random seeds (`42`, `123`, `999`), perform one controlled ablation per family, and test whether architectural rankings hold when transitioning from multi-domain voice assistant queries (SNIPS) to specialized flight-booking requests (ATIS).

---

## 2. Datasets & Access

1. **Primary Dataset: SNIPS Voice Assistant**
   * **Domain:** 7 balanced virtual assistant intent classes (*AddToPlaylist, BookRestaurant, GetWeather, PlayMusic, RateBook, SearchCreativeWork, SearchScreeningEvent*).
   * **Size:** ~14,484 samples partitioned into 70% Train, 15% Validation, and 15% Test.
   * **Sequence Length:** Short command utterances (~9 tokens on average, max length capped at 35).
   * **Source:** Extracted dynamically via `load_dataset("benayas/snips")`.

2. **Secondary Dataset: ATIS (Airline Travel Information System)**
   * **Domain:** Specialized airline travel and flight reservation queries with class imbalance.
   * **Size:** ~5,871 samples partitioned using identical 70/15/15 stratified splits.
   * **Source:** Extracted dynamically via `load_dataset("tuetschek/atis")`.

---

## 3. Project Architecture & File Structure

```text
Midterm_Project_DeepLearning_Group01/
├── data_pipeline.py      # Data extraction, EDA generation, tokenization, train-only vocab, DataLoaders
├── models.py             # PyTorch models: 1D CNN, Bi-LSTM, Transformer Encoder with ablation toggles
├── main.py               # Master experiment orchestrator: runs all 33 seed/ablation runs across seeds
├── eval_baseline.py      # Baseline evaluation, validation C-tuning & confusion matrix
├── eval_cnn.py           # CNN evaluation, BatchNorm ablation metrics & confusion matrix
├── eval_rnn.py           # Recurrent (Bi-LSTM) evaluation, bidirectionality ablation & confusion matrix
├── eval_transformer.py   # Transformer evaluation, positional ablation & confusion matrix
├── compare.py            # Consolidated comparison tables & publication figures (Figures 1, 2, 3)
├── requirements.txt      # Pinned environment dependencies
└── artifacts/            # Generated outputs, JSON logs, confusion matrices, and figure plots
```

---

## 4. How to Reproduce All Results

### Step 1: Environment Setup
Clone the repository and install dependencies in an isolated virtual environment:

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
*Trains the baseline, the three deep models, and all ablations on both SNIPS and ATIS, automatically exporting results to `artifacts/results.json`.*

### Step 3: Generate Report Tables & Figures
```bash
python compare.py
```
*Outputs the formatted tables for Sections 4, 5, 6, and 7 of the report and writes Figures 1, 2, and 3 to `artifacts/`.*

### Step 4: Generate Individual Confusion Matrices
```bash
python eval_baseline.py
python eval_cnn.py
python eval_rnn.py
python eval_transformer.py
```

---

## 5. Experimental Results (Exact Outputs)

### Core Comparison (Primary Dataset: SNIPS)
*Hyperparameter selection guided by validation performance; test set evaluated once.*

| Architecture | Seed 42 | Seed 123 | Seed 999 | Mean Macro F1 | Spread |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Reg)** | 0.9848 | 0.9834 | 0.9830 | **0.9837** | 0.0018 |
| **Convolutional (1D CNN)** | 0.9848 | 0.9806 | 0.9830 | **0.9828** | 0.0042 |
| **Recurrent (Bi-LSTM)** | 0.9829 | 0.9811 | 0.9798 | **0.9813** | 0.0032 |
| **Attention (Transformer)** | 0.9838 | 0.9825 | 0.9760 | **0.9808** | 0.0078 |

### Ablation Studies (SNIPS)
*Each ablation isolates and modifies exactly one architectural component, holding all other hyperparameters fixed.*

| Family | Component Tested | Standard Macro F1 | Ablated Macro F1 | Delta ($\Delta$) | Empirical Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Convolutional** | Batch Normalization | 0.9828 | 0.9822 | **-0.0006** | Removing BatchNorm slightly decreases generalization stability across small mini-batches. |
| **Recurrent** | Bidirectionality | 0.9813 | 0.9799 | **-0.0014** | Unidirectional processing suffers a drop because future contextual tokens help disambiguate early command verbs. |
| **Attention** | Positional Encoding | 0.9808 | 0.9803 | **-0.0005** | Removing positional encodings marginally reduces performance, as short keyword presence dominates over syntactic ordering in SNIPS. |

### Second Dataset Generalization Check (ATIS)
*Evaluating whether the ranking and inductive biases observed on SNIPS generalize to ATIS.*

| Architecture | SNIPS Mean F1 | ATIS Mean F1 | ATIS Rank | Architectural Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Logistic Reg)** | 0.9837 | 0.7426 | Rank 4 | Bag-of-words degrades significantly on ATIS due to severe class imbalance and overlapping flight vocabulary. |
| **Convolutional (1D CNN)** | 0.9828 | 0.7953 | Rank 3 | Captures local phrasal triggers (*"flight from"*, *"round trip"*), improving over baseline by +5.27%. |
| **Recurrent (Bi-LSTM)** | 0.9813 | 0.8161 | Rank 2 | Masked pooling over temporal steps effectively resolves complex sequential syntax in flight requests. |
| **Attention (Transformer)** | 0.9808 | 0.8192 | Rank 1 | Pairwise self-attention provides the strongest generalization on ATIS by directly linking source and destination slots. |

> **Conclusion on Second Dataset:** **Ranking Reversal Confirmed.** While keyword-dense SNIPS allows simpler local/BoW features to perform competitively, the syntactic complexity and slot dependencies of ATIS cause the architectural ranking to reverse, with deep sequence models—led by the Transformer (0.8192) and Bi-LSTM (0.8161)—clearly outperforming the baseline (0.7426).

### Operational Inference Cost Profiling
*Measured on test batches with identical sequence length ($L=35$ tokens).*

| Architecture | Total Trainable Parameters | Inference Latency / Sample (ms) | Peak Memory Allocation |
| :--- | :---: | :---: | :--- |
| **Convolutional (1D CNN)** | 587,071 | 0.3646 ms | Minimal (CPU/MPS BLAS) |
| **Recurrent (Bi-LSTM)** | 734,855 | 0.1320 ms | Minimal (CPU/MPS BLAS) |
| **Attention (Transformer)** | 734,727 | 0.0909 ms | Minimal (CPU/MPS BLAS) |

*Parameter Counts:* All three deep architectures are within a factor of ~1.25 of each other, satisfying the project's parameter budget requirement.

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
| **Member 1** | N01234567 | Baseline & Shared Evaluation | Data pipeline, stratified splitting, Logistic Regression grid tuning, inference cost profiling |
| **Member 2** | N01234568 | Convolutional Model (1D CNN) | TextCNN implementation, BatchNorm ablation study, convolutional receptive field analysis |
| **Member 3** | N01234569 | Recurrent Model (LSTM) | Bi-LSTM implementation, masked pooling mechanism, unidirectional ablation |
| **Member 4** | N01234570 | Attention Model (Transformer) | Transformer Encoder, padding masking, positional encoding ablation |
| **Member 5** | N01234571 | Second Dataset Check (ATIS) | ATIS pipeline porting, cross-dataset ranking analysis, report consolidation |

---

## 8. Academic Integrity & References

1. Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. EMNLP 2014.
2. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
3. Coucke, A., et al. (2018). *Snips Voice Platform: An embedded Spoken Language Understanding system*. arXiv:1805.10190.
4. Hemphill, C. T., et al. (1990). *The ATIS spoken language systems pilot corpus*. DARPA Speech and Natural Language Workshop.