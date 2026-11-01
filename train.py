"""Train the learning-needs classifier and save it for the Flask app."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    data = pd.read_csv(BASE_DIR / "data" / "training.csv")
    texts = data["text"].str.strip()
    labels = data["label"].str.strip()

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Diagnostic report on held-out synthetic examples:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Real user messages need a separate evaluation set.\n")

    # After the diagnostic check, use all examples for the saved model.
    model.fit(texts, labels)
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "course_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
