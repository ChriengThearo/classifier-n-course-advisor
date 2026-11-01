"""Small Flask interface for the learning-needs classifier."""

from pathlib import Path

import joblib
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "course_classifier.joblib"
MAX_MESSAGE_LENGTH = 1000

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
app = Flask(__name__)


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
            result = TRACKS.get(category)
            if result is None:
                error = "We could not suggest a track for this message."

    return render_template(
        "index.html",
        message=message,
        result=result,
        error=error,
        max_message_length=MAX_MESSAGE_LENGTH,
    )


if __name__ == "__main__":
    app.run()
