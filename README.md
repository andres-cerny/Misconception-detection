# NI-MLP Semestral Work — Mining Misconceptions in Mathematics

> **Course:** NI-MLP — Machine Learning for NLP  
> **FIT CTU Prague** · Academic Year 2025/2026  
> **Author:** Ondřej Černý

---

## Overview

This project develops a system to predict the **misconception** behind an incorrect answer choice in multiple-choice mathematics questions. It is based on the [Eedi - Mining Misconceptions in Mathematics Kaggle competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview).

Given a math question, its correct answer, and a student's chosen distractor (incorrect answer), the goal is to identify which of 2,587 possible misconceptions best explains the student's error. Submissions are evaluated using **MAP@25** — the Mean Average Precision at 25 predicted misconceptions per question.

The work is split across three notebooks, best read in order via the `index.html` entry point.

---

## Repository Structure

```
mlp-cernyo14/
├── index.html                          # Entry point — read this first
├── task_intro_data_preprocess.ipynb    # Data exploration and preprocessing
├── task_intro_data_preprocess.html     # Rendered version
├── similarity_search.ipynb             # Approach 1: embedding-based similarity search
├── similarity_search.html              # Rendered version
├── fine_tuning_bge_large.ipynb         # BGE model fine-tuning
├── fine_tuning_bge_large.html          # Rendered version
├── zero_shot.ipynb                     # Approach 2: LLM-generated misconceptions + retrieval
├── zero_shot.html                      # Rendered version
├── convert_to_html.py                  # Utility to render notebooks to HTML
├── example_description.png             # Example question illustration
├── losses/
│   └── bge_large_loss_final.csv        # Training loss and correlation history
├── data/
│   ├── train.csv                       # Original Kaggle training set
│   ├── test.csv                        # Original Kaggle test set
│   ├── misconception_mapping.csv       # 2,587 misconception ID-to-name mappings
│   ├── sample_submission.csv           # Kaggle submission format reference
│   ├── train_df_train.csv              # Local train split
│   ├── train_df_test.csv               # Local test split
│   └── train_df_long.csv               # Long-format training data
└── cernyo14_mlp.zip                    # Packaged submission archive
```

> **Note:** The `data/` folder contains Kaggle competition datasets. Download them from the [competition page](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics) if not present locally.

---

## Methodology

### 1. Data and Task Setup (`task_intro_data_preprocess`)

The training data contains math questions with four answer options (one correct, three distractors) along with the true misconception for each distractor. Each question is enriched with metadata: `ConstructName`, `SubjectName`, `QuestionText`, and the text of each answer option.

The misconception mapping dataset provides 2,587 unique misconception names. The evaluation metric is MAP@25 — for each question-answer pair, the model must return a ranked list of 25 misconception IDs, and the true misconception should appear as high as possible in the list.

Key preprocessing steps:
- All text columns (`ConstructName`, `SubjectName`, `QuestionText`, `AnswerText`) are concatenated into a single `all_text` field per question-answer pair.
- The training data is split into a local train and test set for offline evaluation.
- The `QuestionId` and `Answer` columns are combined into a `QuestionId_Answer` key to match the Kaggle submission format.

### 2. BGE Embedding Model Fine-Tuning (`fine_tuning_bge_large`)

The `bge-large-en-v1.5` sentence embedding model (1024-dimensional embeddings) is fine-tuned on positive pairs using `MultipleNegativesRankingLoss` from the `SentenceTransformers` library. Each pair maps an `all_text` string (source) to its correct `MisconceptionName` (target), training the model to embed semantically related strings closer together in vector space.

Evaluation during training uses `EmbeddingSimilarityEvaluator`, tracking Spearman and Pearson rank correlations of cosine similarities against ground-truth labels. The training loss and correlation curves are saved to `losses/bge_large_loss_final.csv`.

### 3. Approach 1 — Similarity Search (`similarity_search`)

The fine-tuned BGE model encodes all `all_text` strings from the test set and all 2,587 misconception names from the mapping dataset. For each question-answer pair, the top 25 most similar misconception embeddings are retrieved using either **semantic search** or **cosine similarity** — both yielded identical MAP@25 scores.

**Result:** MAP@25 ≈ **0.21** (local test set) / **0.23** (Kaggle leaderboard).

### 4. Approach 2 — Zero-Shot LLM Generation + Retrieval (`zero_shot`)

To improve over pure similarity search, a two-stage retrieval-augmented generation pipeline is used:

1. **First retrieval:** The fine-tuned BGE model retrieves the top 100 candidate misconceptions for each question via semantic search.
2. **LLM generation:** [QWEN 2.5](https://huggingface.co/Qwen) is prompted with the question, the correct answer, the student's incorrect answer, and the top 100 retrieved misconceptions as context examples. The model is instructed to generate a concise free-text description of the likely misconception.
3. **Post-processing:** LLM outputs that consist only of a number (where the model selected from the provided examples rather than generating text) are mapped back to the corresponding misconception name.
4. **Second retrieval:** The generated misconception text is concatenated with its original prompt and re-encoded by the fine-tuned BGE model. Semantic search then retrieves the top 25 closest matches from the full misconception mapping dataset.

**Result:** MAP@25 ≈ **0.39** (local test set) / **0.37** (Kaggle leaderboard).

---

## Results and Limitations

| Approach | MAP@25 (local) | MAP@25 (Kaggle) |
|---|---|---|
| Similarity Search (BGE fine-tuned) | ~0.21 | 0.23 |
| Zero-Shot LLM + Retrieval (QWEN 2.5) | ~0.39 | 0.37 |
| Best competitor (reference) | — | 0.63 |

The LLM-augmented approach nearly doubled MAP@25 over similarity search alone, placing in the **top 300 out of 1,449 competitors**. The remaining gap to the top of the leaderboard is largely attributable to compute constraints: the top competitors either fully fine-tuned LLMs or applied LoRA adapters, approaches that were not feasible with available resources.

Other identified avenues for improvement:
- **Triplet fine-tuning:** The BGE model was fine-tuned on positive pairs only. Using anchor-positive-negative triplets with `TripletLoss` would push the model to better distinguish between similar misconceptions.
- **LoRA fine-tuning of the LLM:** Fine-tuning QWEN 2.5 with low-rank adaptation on the training labels would likely produce more precise misconception descriptions.
- **Prompt engineering:** The LLM occasionally returns a reference number instead of a free-text misconception. A more constrained output format could reduce this failure mode.

---

## How to Read the Work

The recommended reading order follows the notebook pipeline:

1. `index.html` — start here for a guided overview
2. `task_intro_data_preprocess.html` — task description, data exploration, preprocessing
3. `similarity_search.html` — Approach 1 and BGE embedding baseline
4. `fine_tuning_bge_large.html` — embedding fine-tuning details
5. `zero_shot.html` — Approach 2: LLM generation + second-stage retrieval

All notebooks are pre-rendered as HTML files with outputs included. Re-running the notebooks requires the Kaggle datasets in `data/` and a GPU for LLM inference with QWEN 2.5.

---

## Getting Started

**Install dependencies:**
```bash
pip install sentence-transformers transformers torch pandas numpy scikit-learn matplotlib seaborn jupyterlab
```

**Run a notebook:**
```bash
jupyter lab similarity_search.ipynb
```

**Re-render HTML outputs:**
```bash
python convert_to_html.py
```

---

## Key Libraries

| Library | Purpose |
|---|---|
| `sentence-transformers` | BGE model loading, fine-tuning, and semantic search |
| `transformers` | QWEN 2.5 LLM inference |
| `torch` | GPU-accelerated model inference and training |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |

---

## Course Context

**NI-MLP** (Machine Learning for NLP) is a master's-level course at FIT CTU Prague covering modern NLP techniques including text embeddings, language model fine-tuning, information retrieval, and generative models.

This project applies core NI-MLP concepts in a practical setting:
- Dense retrieval using fine-tuned sentence embedding models
- Contrastive learning with `MultipleNegativesRankingLoss`
- Retrieval-Augmented Generation (RAG) with a prompted LLM
- Evaluation using ranking metrics (MAP@25)
