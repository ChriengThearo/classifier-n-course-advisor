"""Small Flask interface for the learning-needs classifier."""

import csv
import random
import re
from pathlib import Path

import joblib
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "course_classifier.joblib"
RESPONSES_PATH = BASE_DIR / "data" / "responses.csv"
MAX_MESSAGE_LENGTH = 1000
COMMON_WORDS = {
    "a", "about", "ai", "an", "and", "are", "as", "at", "automate",
    "automation", "be", "better", "can", "content", "could", "course",
    "data", "do", "explore", "for", "goal", "help", "how", "i", "in",
    "is", "it", "learn", "learning", "me", "my", "need", "of", "on",
    "or", "start", "study", "suggest", "the", "to", "track",
    "want", "where", "with", "work", "would", "you", "your",
}

TRACKS = {
    "ai_foundations": {
        "title": "AI Foundations",
        "description": "Start with core AI ideas, common tools, and responsible use.",
    },
    "workflow_automation": {
        "title": "Workflow Automation",
        "description": "Learn to connect tools and simplify repetitive tasks.",
    },
    "content_creation": {
        "title": "AI Content Creation",
        "description": "Practice writing, planning, and producing content with AI tools.",
    },
    "data_analysis": {
        "title": "AI Data Analysis",
        "description": "Explore spreadsheets, charts, and data-driven decisions.",
    },
}

if not MODEL_PATH.exists():
    raise RuntimeError("Model not found. Run '.\\.venv\\Scripts\\python.exe train.py' first.")

model = joblib.load(MODEL_PATH)

with RESPONSES_PATH.open(newline="", encoding="utf-8-sig") as source:
    reader = csv.DictReader(source)
    if reader.fieldnames != ["label", "response"]:
        raise ValueError("responses.csv must have label,response columns")

    responses = {label: [] for label in TRACKS}
    for row in reader:
        label = row["label"].strip()
        sentence = row["response"].strip()
        if label in responses and sentence:
            responses[label].append(sentence)

if any(not sentences for sentences in responses.values()):
    raise ValueError("responses.csv needs at least one sentence for every track")

vectorizer = model.named_steps["tfidf"]
response_vectors = {
    label: vectorizer.transform(sentences)
    for label, sentences in responses.items()
}
app = Flask(__name__)


def choose_response(message: str, label: str) -> str:
    """Vary the wording while favoring sentences related to the message."""
    message_vector = vectorizer.transform([message])
    scores = (response_vectors[label] @ message_vector.T).toarray().ravel()
    message_terms = set(re.findall(r"[a-z]+", message.casefold())) - COMMON_WORDS
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    relevant_indices = [
        index
        for index in ranked_indices
        if message_terms & (set(re.findall(r"[a-z]+", responses[label][index].casefold())) - COMMON_WORDS)
    ]
    if relevant_indices:
        return random.choice([responses[label][index] for index in relevant_indices[:3]])

    return f"Based on your goal, I suggest exploring {TRACKS[label]['title']}."


@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    result = None
    error = None

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if not message:
            error = "Please describe what you want to learn."
        elif len(message) > MAX_MESSAGE_LENGTH:
            error = f"Please keep your message under {MAX_MESSAGE_LENGTH:,} characters."
        else:
            category = model.predict([message])[0]
            track = TRACKS.get(category)
            if track is None:
                error = "We could not suggest a track for this message."
            else:
                result = {
                    **track,
                    "response": choose_response(message, category),
                }

    return render_template(
        "index.html",
        message=message,
        result=result,
        error=error,
        max_message_length=MAX_MESSAGE_LENGTH,
    )


if __name__ == "__main__":
    app.run()
