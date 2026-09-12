# 🩺 Clinical Summarization using Small Language Models & LoRA

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Supported-red.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20v0.10.0-green.svg)
![Model](https://img.shields.io/badge/Base%20Model-flan--t5--base-blueviolet.svg)

An end-to-end, resource-efficient natural language processing (NLP) system engineered to transform verbose, unstructured MIMIC-IV clinical discharge summaries into structured, abstractive summaries. 

By parameter-efficient fine-tuning (PEFT) **FLAN-T5-base** via **Low-Rank Adaptation (LoRA)**, this project reduces trainable parameters to **0.35%** of the full model, generating a portable **~3.7 MB adapter** capable of running local clinical inference on consumer-grade hardware.

---

## 🌟 Key Features & Innovations

* **Extreme Parameter Efficiency:** Trains only **884,736 parameters** (out of 248.4M), preserving base model fluency while adapting specifically to medical terminology.
* **Feature-Engineered Data Selection:** Includes a custom preprocessing pipeline (`src/preprocess.py`) that scores raw medical texts by compression potential and clinical keyword density to extract high-yield training splits.
* **Abstractive Narrative Generation:** Produces cohesive, human-readable clinical summaries detailing patient history, primary diagnoses, and discharge plans without verbatim extraction artifacts.
* **Cross-Environment Reproducibility:** Configured for seamless execution across local prototyping environments (Windows / NVIDIA RTX 2050) and cloud accelerators (Google Colab / Tesla T4).

---

## 🏗 System Architecture & Pipeline

```
  [ Raw MIMIC-IV Data ] 
            │
            ▼
 ┌──────────────────────┐
 │   src/preprocess.py  │ ──► (Quality Scoring & Keyword Filtering)
 └──────────────────────┘
            │
            ▼
 ┌──────────────────────┐
 │  train_3k.csv Data   │
 └──────────────────────┘
            │
            ▼
 ┌──────────────────────┐
 │  src/train_colab.py  │ ──► (FLAN-T5-Base + LoRA Fine-Tuning)
 └──────────────────────┘
            │
            ▼
 ┌──────────────────────┐
 │ best_lora_model (3.7MB)│ ──► (Saved Adapter Weights)
 └──────────────────────┘
            │
            ▼
 ┌──────────────────────┐
 │src/evaluate_model.py │ ──► (ROUGE Metric Computation & Local Inference)
 └──────────────────────┘
```

---

## 📊 Technical Specifications

### Model & Fine-Tuning Hyperparameters

| Parameter | Configuration |
| :--- | :--- |
| **Base Architecture** | `google/flan-t5-base` (Seq2Seq LM) |
| **Total Base Parameters** | 248,462,592 |
| **Trainable Parameters** | **884,736 (0.3561%)** |
| **LoRA Target Modules** | Attention Projection Matrices (`q`, `v`) |
| **LoRA Rank ($r$)** | `8` |
| **LoRA Alpha ($\alpha$)** | `32` |
| **LoRA Dropout** | `0.05` |
| **Optimizer & Learning Rate** | AdamW ($5 \times 10^{-4}$) |
| **Batch Configuration** | Micro-batch size `8`, Gradient Accumulation `2` (Effective `16`) |
| **Precision** | Standard FP32 |
| **Adapter Storage Footprint** | **~3.7 MB** |

---

## 📁 Repository Structure

```
medical-report-summarization/
├── data/
│   ├── raw/                  # Original raw dataset (raw_15k.csv)
│   └── processed/            # Feature-engineered splits (train_3k.csv, test_500.csv)
├── outputs/
│   └── flan_t5_lora_results/
│       └── best_lora_model/  # Trained LoRA adapter weights & tokenizer configs
├── src/
│   ├── preprocess.py         # Quality scoring & data splitting script
│   ├── train_colab.py        # Fine-tuning execution script
│   └── evaluate_model.py     # Local model inference & ROUGE evaluation script
├── requirements.txt          # Environment dependencies
└── README.md                 # Project documentation
```

---

## 📈 Evaluation Results & Performance Metrics

Evaluated locally on **500 unseen clinical test reports** (`test_500.csv`):

| Metric | Score | Clinical Significance |
| :--- | :--- | :--- |
| **ROUGE-1** | **26.70%** | Measures unigram overlap; reflects retrieval of key medical entities. |
| **ROUGE-2** | **9.73%** | Measures bigram overlap; captures multi-word clinical phrases (e.g., *lithium toxicity*, *acute renal failure*). |
| **ROUGE-L** | **19.12%** | Measures longest common sequence; confirms fluent sentence structure and sequential flow. |

> **Note on Metric Interpretation:** Standard ROUGE scores penalize stylistic variations against raw clinical templates. Qualitative verification demonstrates that the abstractive outputs correctly synthesize critical patient demographics, comorbidities, and clinical progression into cohesive narratives.

---

## 🔬 Sample Output Comparison

```
================================================================================
REPORT INPUT (EXCERPT):
Chief Complaint: Leg weakness; transferred for ARF and lithium toxicity
History of Present Illness: Man with bipolar disorder, DM, obesity, hypertension, 
elevated baseline creatinine, and recent increase in lithium dosing...

GROUND TRUTH SUMMARY:
Man with bipolar disorder, DM, obesity, hypertension admitted with acute renal failure 
and tremors, weakness likely secondary to lithium toxicity. Pt initially admitted to ICU...

MODEL GENERATED SUMMARY:
Mr. is a year old man with bipolar disorder as well as PMH significant for DM, obesity, 
hypertension, and elevated baseline creatinine as well as recent increase in lithium dosing. 
He presents with complaints of weakness and urinary frequency and is transferred for ARF 
and lithium toxicity.
================================================================================
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

Clone the repository and install dependencies:

```bash
git clone [https://github.com/your-username/medical-report-summarization.git](https://github.com/your-username/medical-report-summarization.git)
cd medical-report-summarization
pip install -r requirements.txt
```

### 2. Preprocess Raw Data

Run the feature-engineering pipeline to score sample quality and generate output CSVs:

```bash
python src/preprocess.py
```

### 3. Fine-Tune the Model

* **Local Prototype (RTX 2050):** Set `DRY_RUN = True` in `src/train_colab.py` and execute:
  ```bash
  python src/train_colab.py
  ```
* **Full Cloud Training (Google Colab T4):** Run the notebook script using standard FP32 precision (`fp16=False`).

### 4. Evaluate Locally

Run the evaluation script to load the base model + LoRA adapter, generate predictions, and compute ROUGE scores:

```bash
python src/evaluate_model.py
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.