import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------
# Text Cleaning Function
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-za-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("emails.csv")

# Combine subject + body
df["text"] = (
    df["subject"].fillna("") +
    " " +
    df["body"].fillna("")
)

# Apply preprocessing
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# ---------------------------
# Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# Models
# ---------------------------
models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Naive Bayes":
        MultinomialNB(),

    "Linear SVM":
        LinearSVC(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

    "AdaBoost":
        AdaBoostClassifier(
            random_state=42
        )
}

# ---------------------------
# Compare Models
# ---------------------------
results = []

for name, classifier in models.items():

    print(f"Training {name}...")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000
            )
        ),
        (
            "classifier",
            classifier
        )
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

# ---------------------------
# Results Table
# ---------------------------
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n")
print("=" * 90)
print("MODEL COMPARISON RESULTS")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format="%.4f"
    )
)

# Save comparison results
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nResults saved to model_comparison.csv")

print("\n")
print("Best Model:")
print(results_df.iloc[0])