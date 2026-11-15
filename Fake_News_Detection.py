"""
#DAT 3202 - Machine Learning
#Continuous Assessment Test III - Mini Project

Title   : Detecting Fake News Headlines Circulating on Kenyan Social Media
Problem : Binary classification of Kenyan news headlines as "Fake" or "Real"
          using engineered text and metadata features.

Author  : Mukami (Strathmore University, BSc Statistics & Data Science)

Pipeline:
    1. Load and inspect the dataset (fake_news_dataset.csv)
    2. Construct training features:
         - Numeric   : Word_Count, Exclamation_Count
         - Text-derived: headline length, capital-letter ratio, digit count,
                         question-mark flag, clickbait keyword count
         - Categorical: Source_Platform (one-hot encoded)
         - Textual   : TF-IDF (top 60 unigrams/bigrams) from the Headline
    3. Train & compare four ML algorithms:
         - K-Nearest Neighbours (KNN)
         - Decision Tree (DT)
         - Artificial Neural Network (ANN / MLPClassifier)
         - Support Vector Machine (SVM)
       using 2-fold cross-validation (as required by the CAT brief).
    4. Also produce a held-out train/test split (70/30) so we can report a
       confusion matrix and classification report for discussion purposes.
    5. Save all numeric results (JSON/CSV) and charts (PNG) to the
       results/ directory for use in the PowerPoint report.
"""

import json
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack, csr_matrix

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_style("whitegrid")
PALETTE = {
    "KNN": "#028090",
    "Decision Tree": "#00A896",
    "ANN (MLP)": "#02C39A",
    "SVM": "#21295C",
}

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("fake_news_dataset.csv")
print(f"Dataset shape: {df.shape}")
print(df["Label"].value_counts())

# ---------------------------------------------------------------------------
# 2. TRAINING DATA CONSTRUCTION (feature engineering)
# ---------------------------------------------------------------------------
CLICKBAIT_WORDS = [
    "shocking", "confirmed", "secret", "scandal", "breaking", "urgent",
    "must", "click", "warning", "exposed", "leaked", "viral", "cure",
    "miracle", "banned", "shocking", "won't believe", "click here",
]


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    headline = data["Headline"].astype(str)

    data["Headline_Length"] = headline.str.len()
    data["Capital_Ratio"] = headline.apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )
    data["Digit_Count"] = headline.apply(lambda t: sum(c.isdigit() for c in t))
    data["Has_Question"] = headline.str.contains(r"\?").astype(int)
    data["Clickbait_Word_Count"] = headline.str.lower().apply(
        lambda t: sum(t.count(w) for w in CLICKBAIT_WORDS)
    )
    return data


df = engineer_features(df)

NUMERIC_FEATURES = [
    "Word_Count",
    "Exclamation_Count",
    "Headline_Length",
    "Capital_Ratio",
    "Digit_Count",
    "Has_Question",
    "Clickbait_Word_Count",
]
CATEGORICAL_FEATURES = ["Source_Platform"]
TEXT_FEATURE = "Headline"
TARGET = "Label_Numeric"  # 1 = Fake, 0 = Real

X_raw = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]]
y = df[TARGET].values

# Preprocessing pipeline for the structured (numeric + categorical) part
struct_transformer = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

tfidf = TfidfVectorizer(max_features=60, ngram_range=(1, 2), stop_words="english")


def build_feature_matrix(data: pd.DataFrame, fit: bool):
    """Fit (on train) or transform (on test/full) structured + TF-IDF features."""
    if fit:
        struct = struct_transformer.fit_transform(data)
        text = tfidf.fit_transform(data[TEXT_FEATURE])
    else:
        struct = struct_transformer.transform(data)
        text = tfidf.transform(data[TEXT_FEATURE])
    struct = csr_matrix(struct)
    return hstack([struct, text]).tocsr()


# Full feature matrix (fit once) used for 2-fold cross-validation
X_full = build_feature_matrix(X_raw, fit=True)
print(f"Final feature matrix shape: {X_full.shape}")

# ---------------------------------------------------------------------------
# 3. MODELS
# ---------------------------------------------------------------------------
models = {
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "ANN (MLP)": MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=2000,
        random_state=RANDOM_STATE,
        early_stopping=True,
    ),
    "SVM": SVC(kernel="rbf", C=2.0, probability=True, random_state=RANDOM_STATE),
}

# ---------------------------------------------------------------------------
# 4. 2-FOLD CROSS-VALIDATION (as required by the assessment brief)
# ---------------------------------------------------------------------------
kf = KFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

print("\n=== 2-Fold Cross-Validation Results ===")
for name, model in models.items():
    scores = cross_val_score(model, X_full, y, cv=kf, scoring="accuracy")
    cv_results[name] = {
        "fold_1_accuracy": round(float(scores[0]), 4),
        "fold_2_accuracy": round(float(scores[1]), 4),
        "mean_accuracy": round(float(scores.mean()), 4),
        "std_accuracy": round(float(scores.std()), 4),
    }
    print(f"{name:15s} fold1={scores[0]:.4f}  fold2={scores[1]:.4f}  "
          f"mean={scores.mean():.4f}  std={scores.std():.4f}")

with open(f"{RESULTS_DIR}/cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)

# ---------------------------------------------------------------------------
# 5. HOLD-OUT SPLIT for confusion matrices / classification reports
# ---------------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# Refit preprocessors on the training split only (avoid leakage)
X_train = build_feature_matrix(X_train_raw, fit=True)
X_test = build_feature_matrix(X_test_raw, fit=False)

holdout_results = {}
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    holdout_results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\nAcc={acc:.2f}", fontsize=11)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=200)
plt.close()

with open(f"{RESULTS_DIR}/holdout_results.json", "w") as f:
    json.dump(holdout_results, f, indent=2)

print("\n=== Hold-out (70/30 split) Test Results ===")
for name, res in holdout_results.items():
    print(name, res)

# ---------------------------------------------------------------------------
# 6. CHARTS FOR THE REPORT
# ---------------------------------------------------------------------------

# 6a. CV mean accuracy bar chart (with fold error bars)
names = list(cv_results.keys())
means = [cv_results[n]["mean_accuracy"] for n in names]
stds = [cv_results[n]["std_accuracy"] for n in names]
colors = [PALETTE[n] for n in names]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.015, f"{m:.3f}",
             ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Mean Accuracy (2-fold CV)")
ax.set_title("Model Comparison — 2-Fold Cross-Validation Accuracy")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/cv_accuracy_comparison.png", dpi=200)
plt.close()

# 6b. Fold-by-fold accuracy (grouped bar)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(names))
width = 0.35
f1_vals = [cv_results[n]["fold_1_accuracy"] for n in names]
f2_vals = [cv_results[n]["fold_2_accuracy"] for n in names]
ax.bar(x - width / 2, f1_vals, width, label="Fold 1", color="#028090")
ax.bar(x + width / 2, f2_vals, width, label="Fold 2", color="#F9E795", edgecolor="#21295C")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy")
ax.set_title("Fold-by-Fold Accuracy per Model")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fold_by_fold_accuracy.png", dpi=200)
plt.close()

# 6c. Precision / Recall / F1 grouped bar (hold-out set)
metrics = ["precision", "recall", "f1"]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(names))
width = 0.25
for i, metric in enumerate(metrics):
    vals = [holdout_results[n][metric] for n in names]
    ax.bar(x + (i - 1) * width, vals, width, label=metric.capitalize())
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 on Held-out Test Set (30%)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/precision_recall_f1.png", dpi=200)
plt.close()

# 6d. Class balance chart
fig, ax = plt.subplots(figsize=(5, 5))
df["Label"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", colors=["#990011", "#028090"], ax=ax, ylabel=""
)
ax.set_title("Class Distribution: Fake vs Real")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/class_distribution.png", dpi=200)
plt.close()

# 6e. Source platform distribution
fig, ax = plt.subplots(figsize=(8, 5))
platform_counts = df["Source_Platform"].value_counts()
platform_counts.plot(kind="barh", color="#028090", ax=ax)
ax.set_xlabel("Number of Headlines")
ax.set_title("Headlines by Source Platform")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/platform_distribution.png", dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# 7. TEXT REPORT (classification_report per model) for reference/appendix
# ---------------------------------------------------------------------------
report_text = ""
for name, model in models.items():
    y_pred = model.predict(X_test)
    report_text += f"\n===== {name} =====\n"
    report_text += classification_report(y_test, y_pred, target_names=["Real", "Fake"])

with open(f"{RESULTS_DIR}/classification_reports.txt", "w") as f:
    f.write(report_text)

print("\nAll results and charts saved to the results/ directory.")
print(report_text)
