# Toxic Comment Classification with DistilBERT
### Assignment 03: Transfer Learning, Fine-Tuning, Hyperparameter Tuning and Interpretability
Skills for Industry Competitiveness and Innovation Program (SICIP), BRAC University

| | |
|---|---|
| **Name** | Khandakar Humyra Oyshi |
| **Student ID** | 1000128211 |

---

## 1. Task
Given a user comment, classify it as **toxic** or **non-toxic**. The classifier is trained and evaluated on the
[Civil Comments](https://huggingface.co/datasets/google/civil_comments) dataset and then tested on
hand-written real-world comments. The pipeline covers transfer learning (frozen baseline), end-to-end
fine-tuning, hyperparameter tuning and interpretability.

## 2. Repository structure
```
.
├── MLDLAssignment_3.ipynb   # complete pipeline, with all outputs
├── README.md                # this technical report
├── requirements.txt
├── images/                  # figures produced by the notebook and used below
└── .gitignore               # excludes data, checkpoints, caches
```

## 3. How to run
1. Open `MLDLAssignment_3.ipynb` in Google Colab.
2. *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all*. The full run takes about 45 minutes on a T4; the Optuna search is about 15 minutes of that.

All randomness is seeded (`SEED = 42`). Hardware used: Google Colab, NVIDIA Tesla T4.

## 4. Data and preprocessing
- Loaded with `load_dataset("google/civil_comments")`. The comment is in `text`, the score in `toxicity`.
- **Label threshold: `toxicity >= 0.5` → toxic (1), otherwise non-toxic (0).** A score of 0.5 means at least
  half of the annotators rated the comment toxic, which is the convention used by the dataset authors.
- Because of compute limits I used a **random subset** (seed 42) of each official split:

| Split | Comments | Non-toxic | Toxic | % toxic |
|---|---|---|---|---|
| Train | 20,000 | 18,387 | 1,613 | 8.06 % |
| Validation | 4,000 | 3,662 | 338 | 8.45 % |
| Test | 4,000 | 3,684 | 316 | 7.90 % |

- Texts are tokenised with the DistilBERT tokenizer and truncated to **128 tokens**.
- **Class imbalance:** only ~8% of comments are toxic. A model that always predicts "non-toxic" would already
  reach ~92% accuracy, so accuracy alone says little. I report accuracy together with **precision, recall and
  F1 of the toxic class**, macro-F1, ROC-AUC and PR-AUC, and use **toxic-class F1** to select models.

## 5. Model choice: `distilbert-base-uncased`
- It is a BERT-based model distilled from BERT-base: about **40% smaller and 60% faster while keeping ~97%
  of BERT's language-understanding performance** (Sanh et al., 2019).
- Speed matters here because hyperparameter tuning needs many complete fine-tuning runs on a free Colab GPU.
  Each trial took 34–126 seconds, which made a 12-trial search possible.
- The uncased version suits online comments, where capitalisation is inconsistent ("STUPID", "Stupid", "stupid").

## 6. Pipeline

### 6.1 Transfer learning: frozen baseline
DistilBERT is loaded with **all weights frozen** (`requires_grad = False`; the notebook prints
`Trainable encoder params: 0`) and used only as a feature extractor. Each comment becomes a 768-dim vector
(mean of the last hidden states over the non-padding tokens). A **logistic regression** with standardised
features and `class_weight="balanced"` is trained on these vectors.

### 6.2 Fine-tuning end-to-end
All weights are unfrozen and a 2-class classification head is trained together with the encoder using the
Hugging Face `Trainer`. To handle the imbalance, the loss is a **class-weighted cross-entropy** with weights
`[1.0, 3.38]` (toxic weight = sqrt(neg/pos)). Default hyperparameters: learning rate 2e-5, batch size 32,
2 epochs, weight decay 0.01, 10% warmup, fp16.

### 6.3 Hyperparameter tuning
- **Method:** Optuna with the **TPE sampler** (Tree-structured Parzen Estimator, a Bayesian optimisation
  method), **12 trials**. TPE uses the results of earlier trials to propose new settings, which is more
  efficient than grid search for a 5-dimensional space.
- **Objective:** toxic-class F1 on the validation set.
- **Budget control:** each trial trains on an 8,000-comment slice of the training subset. The best configuration
  is then retrained on all 20,000 training comments and evaluated once on the test set.

| Hyperparameter | Search range | Best value (trial 0) |
|---|---|---|
| Learning rate | 1e-5 – 6e-5 (log scale) | 1.96e-5 |
| Batch size | {16, 32} | 16 |
| Epochs | 1 – 3 | 2 |
| Weight decay | 0.0 – 0.1 | 0.0156 |
| Warmup ratio | 0.0 – 0.2 | 0.031 |

**Tuning results (screenshot):** all 12 trials sorted by validation F1, best trial highlighted.

![Optuna trials](images.zip/tuning_table.png)

| Trial | Val F1 | LR | Batch | Epochs | Weight decay | Warmup | Val precision | Val recall | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| **0** | **0.6380** | 1.96e-5 | 16 | 2 | 0.016 | 0.031 | 0.692 | 0.592 | 74 |
| 2 | 0.6218 | 4.44e-5 | 16 | 1 | 0.030 | 0.105 | 0.628 | 0.615 | 39 |
| 1 | 0.6197 | 1.11e-5 | 16 | 3 | 0.002 | 0.194 | 0.615 | 0.624 | 126 |
| 9 | 0.6192 | 2.92e-5 | 16 | 1 | 0.005 | 0.065 | 0.609 | 0.630 | 39 |
| 7 | 0.6187 | 1.06e-5 | 16 | 2 | 0.031 | 0.104 | 0.641 | 0.598 | 74 |
| 4 | 0.6181 | 2.26e-5 | 16 | 2 | 0.059 | 0.009 | 0.666 | 0.577 | 74 |
| 5 | 0.6164 | 2.97e-5 | 16 | 3 | 0.097 | 0.162 | 0.691 | 0.556 | 108 |
| 11 | 0.6163 | 4.94e-5 | 16 | 1 | 0.046 | 0.083 | 0.595 | 0.639 | 39 |
| 8 | 0.6132 | 2.66e-5 | 32 | 3 | 0.094 | 0.179 | 0.670 | 0.565 | 91 |
| 10 | 0.6113 | 3.50e-5 | 16 | 3 | 0.032 | 0.005 | 0.650 | 0.577 | 108 |
| 6 | 0.6110 | 1.73e-5 | 32 | 2 | 0.012 | 0.099 | 0.585 | 0.639 | 62 |
| 3 | 0.5825 | 2.17e-5 | 32 | 1 | 0.029 | 0.073 | 0.516 | 0.669 | 34 |

![Optimisation history](images/optuna_history.png)
![Hyperparameter importance](images/optuna_importance.png)
![Slice plot](images/optuna_slice.png)

## 7. Results (test set, 4,000 comments)

| Model | Accuracy | Precision | Recall | F1 (toxic) | Macro-F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Frozen DistilBERT + LogReg (baseline) | 0.8070 | 0.2427 | **0.6804** | 0.3577 | 0.6221 | 0.8479 | 0.4270 |
| Fine-tuned DistilBERT (default HP) | 0.9400 | 0.6105 | 0.6646 | **0.6364** | 0.8018 | 0.9289 | **0.6911** |
| Fine-tuned DistilBERT (Optuna-tuned HP) | **0.9422** | **0.6332** | 0.6392 | 0.6362 | **0.8024** | **0.9326** | 0.6721 |

Confusion matrices on the test set (316 toxic, 3,684 non-toxic):

| Model | True positives | False negatives | False positives | True negatives |
|---|---|---|---|---|
| Frozen baseline | 215 | 101 | 671 | 2,913 |
| Fine-tuned (default) | 210 | 106 | 134 | 3,550 |
| Fine-tuned (tuned) | 202 | 114 | 117 | 3,567 |

![Model comparison](images/model_comparison.png)
![Confusion matrices](images/confusion_matrices.png)

All models meet the required accuracy of at least 70%: 80.7% for the frozen baseline and 94.0–94.2% for
the fine-tuned models.

## 8. Justification of the results

### 8.1 Without tuning: frozen baseline vs fine-tuned
Fine-tuning is the largest improvement in the whole pipeline: accuracy rises from **80.7% to 94.0%**,
toxic-class F1 from **0.358 to 0.636** (+78%), and PR-AUC from **0.427 to 0.691**.

- **Why the frozen model is weaker:** DistilBERT was pre-trained for general language modelling, not for
  toxicity. Its frozen, mean-pooled embeddings mostly encode topic and style, so a comment criticising a
  politician and a comment insulting a person can end up close together. A logistic regression can only draw
  one straight boundary in that fixed space, so it cannot separate the classes well (ROC-AUC 0.848).
- **Why its precision is so low (0.24):** `class_weight="balanced"` makes every toxic example count ~11×.
  With poorly separated features, the model keeps recall high (0.68) by flagging many borderline comments.
  It produces **671 false positives**, compared with 134 after fine-tuning.
- **Why fine-tuning helps:** with all layers trainable, the representation itself is reshaped around toxicity
  cues (insults, slurs, hostile phrasing aimed at a person) and their context. The fine-tuned model reaches
  almost the same recall (0.66 vs 0.68) with 2.5× higher precision.

### 8.2 With tuning: default vs Optuna-tuned
Tuning gave **no meaningful change in F1** (0.6364 → 0.6362). It shifted the precision/recall balance:
precision rose from 0.611 to 0.633, recall fell from 0.665 to 0.639, accuracy rose slightly (94.0% → 94.2%)
and ROC-AUC rose (0.929 → 0.933), while PR-AUC fell (0.691 → 0.672). The reasons:

1. **The defaults were already near the optimum.** The best trial (lr 1.96e-5, 2 epochs) is almost the same
   as the default configuration (lr 2e-5, 2 epochs). The main difference is batch size 16 instead of 32.
2. **Most trials were not guided by the sampler.** Optuna's TPE sampler starts with 10 random trials before it
   builds its model, so with 12 trials only 2 were guided by earlier results. The best trial was trial 0,
   one of the random ones.
3. **Trials used an 8,000-comment slice.** Settings that are best on 8k examples do not transfer perfectly to
   the 20k-comment retraining. For example, with fewer warmup steps (3%) the model became more conservative
   and predicted fewer toxic comments.
4. **The differences are within noise.** The validation set has only 338 toxic comments. Apart from trial 3,
   all trials fall between F1 0.611 and 0.638, and a difference of a few comments moves F1 by ~0.01.

**What the search did show:**
- **Batch size 16 is better than 32.** 8 of the top 8 trials used batch 16. With 8k training examples, a smaller
  batch gives twice as many update steps per epoch.
- **Too little training hurts.** The worst trial (3: batch 32, 1 epoch, F1 0.582) had the fewest update steps
  (250) and the lowest precision. It had not yet learned to separate borderline comments.
- **More epochs did not help.** 3-epoch trials (1, 5, 8, 10) did not beat 2-epoch trials, which suggests the
  model starts to overfit the small training slice after 2 epochs.

**Conclusion:** the big gain comes from fine-tuning itself; hyperparameter tuning confirmed that the standard
BERT fine-tuning settings are close to optimal for this task and data size. More trials, more training data,
and tuning the decision threshold would be the next steps.

## 9. Real-world test
Comments written by hand, not taken from the dataset, classified by the tuned model. The *Expected* column is
my own judgement and serves as the check.

| Input | p(toxic) | Predicted | Expected | Check |
|---|---|---|---|---|
| Thanks for sharing this, the article was really well researched. | 0.002 | non_toxic | non_toxic | ✅ |
| I completely disagree with the mayor's new parking policy, it will hurt small shops. | 0.002 | non_toxic | non_toxic | ✅ |
| You are a complete idiot and nobody wants to hear your stupid opinions. | 0.994 | toxic | toxic | ✅ |
| Shut up, loser. Go back to whatever hole you crawled out of. | 0.993 | toxic | toxic | ✅ |
| As a gay man, I think this law will finally protect families like mine. | 0.952 | toxic | non_toxic | ❌ |
| This movie was so bad it killed me, I laughed the whole time. | 0.006 | non_toxic | non_toxic | ✅ |
| Oh great, another genius who thinks reading one headline makes him an expert. | 0.003 | non_toxic | toxic | ❌ |
| People like you are the reason this country is going down the drain, pathetic. | 0.994 | toxic | toxic | ✅ |

**Check: 6 / 8 correct.**
- **Correct:** direct insults ("idiot", "loser", "pathetic") are flagged with >99% confidence. Strong
  disagreement with a policy and a violent figure of speech ("it killed me") are correctly left as non-toxic,
  so the model does not just react to negative words.
- **Wrong 1 – identity bias:** a positive sentence containing "gay" is flagged as toxic (0.952). In Civil
  Comments, identity terms appear far more often in toxic comments than in neutral ones, so the model learned
  "gay" itself as a toxicity cue. SHAP confirms this in Section 10 (example #6). This is a known bias of models
  trained on this dataset (Borkan et al., 2019).
- **Wrong 2 – sarcasm:** a sarcastic jab with no explicit insult word scores only 0.003. The model relies on
  explicit toxic vocabulary and misses toxicity that depends on tone. (This comment is also borderline, and
  many annotators might rate it non-toxic.)

## 10. Interpretability

### Method: SHAP
I used **SHAP** (Partition explainer with a text masker). SHAP hides groups of tokens and measures how the
predicted toxic probability changes, giving each token a contribution. Positive values push towards toxic,
negative values push towards non-toxic.

**Why SHAP:**
- The token contributions add up to the model's actual output, so the explanation is faithful to the
  prediction.
- It is model-agnostic: it treats the fine-tuned model as a black box and needs no changes to it.
- Unlike attention weights, SHAP measures the effect of a token on the output. Attention only shows where the
  model looks, not why it decides.
- Compared with LIME, it is more stable between runs and is based on Shapley values from game theory.

### Selected predictions
A small, deliberately chosen set: 2 true positives, 1 true negative, 1 false positive and 1 false negative
from the test set, plus 2 real-world comments.

| # | Case | True | Pred | p(toxic) | Comment (start) | Top tokens → toxic | Top tokens → non-toxic |
|---|---|---|---|---|---|---|---|
| 0 | True positive | toxic | toxic | 0.988 | "That's such a ridiculous comment ...... If you…" | ridiculous (+0.49) | ? (−0.04), comment (−0.03) |
| 1 | True positive | toxic | toxic | 0.992 | "Stupid to hold these meetings in the heart of…" | Stupid (+0.42), and (+0.12) | to (−0.19), these (−0.07) |
| 2 | True negative | non_toxic | non_toxic | 0.007 | "Got your till foil hat on again lefty !" | on (+0.01) | till (−0.08), again (−0.03) |
| 3 | False positive | non_toxic | toxic | 0.759 | "Well aint you just a consistant persisitant sh…" | a (+0.11), shrill (+0.09), you (+0.08) | consist (−0.05) |
| 4 | False negative | toxic | non_toxic | 0.011 | "The Palestinians should thank this guy for kee…" | ruined (+0.02), guy (+0.02) | thank (−0.06), peace (−0.04) |
| 5 | Real-world | toxic | toxic | 0.994 | "You are a complete idiot and nobody wants to…" | stupid (+0.41), idiot (+0.37) | opinions (−0.01) |
| 6 | Real-world | non_toxic | toxic | 0.952 | "As a gay man, I think this law will finally…" | gay (+0.36), man (+0.14) | finally (−0.02) |

![SHAP example 0: true positive](images/shap_bar_0.png)
![SHAP example 6: identity-term false positive](images/shap_bar_6.png)

Per-example bar charts: `images/shap_bar_0.png` … `images/shap_bar_6.png`. The interactive token-level view
is in `images/shap_text.html`.

### Analysis
- **#0 and #5 (correct, insults):** the decision is driven almost entirely by one or two insult words
  ("ridiculous" +0.49; "stupid" +0.41 and "idiot" +0.37). The rest of the sentence contributes almost nothing.
  The model has learned the right lexical cues.
- **#1 (correct, but a revealing case):** "Stupid" (+0.42) drives the prediction, and "to" pushes the other way
  (−0.19). In "Stupid to hold these meetings…", "stupid" describes a decision, not a person. The model notices
  some of that context, but the insult word still dominates.
- **#2 (true negative):** "Got your till [tin] foil hat on again lefty !" is mocking, but the annotators rated it below
  0.5 and so did the model. No word pushes strongly towards toxic, so the model sees mild sarcasm as acceptable.
- **#3 (false positive):** the toxicity comes from "shrill" and from words aimed at a person ("you", "a").
  The comment is rude and addressed directly at someone, so the prediction is reasonable. The annotators
  happened to score it below 0.5, which shows that some errors come from **label subjectivity**, not the model.
- **#4 (false negative):** the comment is political and its toxicity is implicit. Words like "thank", "peace"
  and "negotiating" push strongly towards non-toxic, and no explicit insult is present. The model **misses
  toxicity that is carried by meaning rather than by vocabulary**.
- **#6 (real-world false positive):** "gay" alone contributes +0.36 and "man" +0.14, which is enough to flip a
  friendly sentence to 95% toxic. This shows directly the **identity-term bias** found in Section 9.

**Overall:** the model has learned explicit toxic vocabulary very well, which explains the high accuracy.
Its errors come from three sources: (1) identity terms treated as toxic, (2) implicit or sarcastic toxicity
without insult words, and (3) borderline comments where annotators themselves disagree. Bias-mitigation
methods or more context-rich training data would be needed to fix the first two.

## 11. Limitations
- Trained on 20k of ~1.8M comments. More data would most likely help rare and implicit cases.
- 12 Optuna trials on an 8k slice is a small search budget, and 10 of them were random start-up trials.
- A fixed 0.5 decision threshold is used. Choosing the threshold on validation data could trade precision for
  recall as needed.
- Civil Comments labels are subjective, and models trained on it over-flag identity terms (Section 9, #6).

## 12. References
- Borkan, D. et al. (2019). *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification.*
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.*
- Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.*
- Lundberg, S. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.*
