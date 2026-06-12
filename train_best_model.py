import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Load dataset
df = pd.read_csv("emails.csv")

# Combine subject and body
df["text"] = (
    df["subject"].fillna("") +
    " " +
    df["body"].fillna("")
)

# Preprocess text
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Best model
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("classifier", LinearSVC())
])

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall   :", round(recall_score(y_test, predictions), 4))
print("F1 Score :", round(f1_score(y_test, predictions), 4))

# Save model
joblib.dump(model, "phishing_model_new.pkl")

print("\nBest model saved as phishing_model_new.pkl")
