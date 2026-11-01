"""Fill responses.csv with varied, category-specific example replies.

These are prepared response options, not examples used to train a text generator.
The script preserves existing rows and only adds rows until each label has 250.
"""

import csv
import random
from collections import Counter
from itertools import product
from pathlib import Path


CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "responses.csv"
TARGET_PER_LABEL = 250

OBJECTIVES = {
    "ai_foundations": [
        "understand the core ideas behind AI",
        "build confidence with common AI terms",
        "recognize everyday uses of AI",
        "understand what machine learning means",
        "explore how AI models learn from examples",
        "learn the difference between AI and automation",
        "understand what language models do",
        "identify the strengths and limits of AI tools",
        "learn how prompts affect AI answers",
        "understand why training data matters",
        "recognize when an AI answer needs checking",
        "explore basic concepts in generative AI",
        "understand how classification works",
        "learn how AI systems are evaluated",
        "compare different kinds of AI tools",
        "understand what an AI agent does",
        "recognize the role of data in AI",
        "learn the basics of responsible AI use",
        "understand how AI can make mistakes",
        "explore how text and image models differ",
        "learn what features and labels mean",
        "understand the purpose of training and testing",
        "build a foundation for more advanced AI topics",
        "get comfortable with AI concepts step by step",
        "learn how to describe an AI problem clearly",
        "understand the difference between models and applications",
        "explore how AI predictions are made",
        "learn the basics of neural networks",
        "understand how AI tools fit into daily work",
        "start learning AI with practical examples",
    ],
    "workflow_automation": [
        "identify tasks that can be automated",
        "map a repetitive process into clear steps",
        "connect tools in a simple workflow",
        "reduce manual data entry",
        "set up useful reminders and notifications",
        "automate routine email follow ups",
        "move information between applications",
        "design reliable approval steps",
        "use triggers and actions in a workflow",
        "simplify recurring office tasks",
        "build a basic no code automation",
        "understand when to use an API or webhook",
        "automate a form submission process",
        "create a repeatable document workflow",
        "save time on routine reporting",
        "organize incoming requests automatically",
        "connect a spreadsheet with another tool",
        "design a workflow with clear conditions",
        "test an automation before relying on it",
        "handle errors in an automated process",
        "automate scheduling and confirmations",
        "streamline handoffs between teammates",
        "turn a manual checklist into a workflow",
        "use Python for repeated office tasks",
        "build an automation around customer requests",
        "improve a process with fewer manual steps",
        "choose a suitable automation tool",
        "monitor whether an automation is working",
        "document a workflow for other people",
        "start automating one small task at a time",
    ],
    "content_creation": [
        "plan content for a clear audience",
        "write more effective social media captions",
        "develop ideas for useful posts",
        "draft articles with a consistent voice",
        "create a practical content calendar",
        "improve the clarity of marketing messages",
        "write stronger headlines and introductions",
        "adapt one idea for several channels",
        "outline a short video script",
        "create descriptions for products or services",
        "edit drafts for tone and readability",
        "use AI tools to brainstorm content ideas",
        "turn notes into a complete article",
        "write engaging email content",
        "plan a campaign around one message",
        "create content that fits a brand voice",
        "prepare a simple visual story",
        "write clear calls to action",
        "create educational posts for beginners",
        "turn an interview into shareable content",
        "draft a newsletter for your audience",
        "create a script for a product demo",
        "make content more useful and specific",
        "organize ideas before writing",
        "rewrite content for a different format",
        "review AI drafts before publishing",
        "create a repeatable writing process",
        "choose the right style for each platform",
        "practice writing with useful examples",
        "start producing content with a clear goal",
    ],
    "data_analysis": [
        "turn raw data into useful insights",
        "read patterns in a spreadsheet",
        "clean data before analyzing it",
        "choose a chart that explains a result",
        "compare business metrics over time",
        "identify trends in sales data",
        "understand basic descriptive statistics",
        "build a simple performance dashboard",
        "ask better questions of a dataset",
        "check whether a data claim is supported",
        "summarize a large table clearly",
        "measure changes in key indicators",
        "find missing or unusual values",
        "compare results across customer groups",
        "use pivot tables to explore data",
        "visualize changes across months",
        "interpret survey results carefully",
        "calculate useful business rates",
        "combine data from related tables",
        "explore data with Python and pandas",
        "use SQL to retrieve relevant records",
        "explain a chart in plain language",
        "evaluate the quality of a dataset",
        "distinguish correlation from causation",
        "track a metric with a repeatable method",
        "analyze what changed in a report",
        "create a clear summary of findings",
        "make data driven decisions with context",
        "practice analysis with real examples",
        "start with a simple spreadsheet question",
    ],
}

FRAMES = [
    "Based on your goal, {track} could help you {objective}.",
    "I suggest {track} if you want to {objective}.",
    "The {track} track can help you {objective}.",
    "You may find {track} useful for learning to {objective}.",
    "A practical next step is {track}, where you can {objective}.",
    "Your question points toward {track} as a way to {objective}.",
    "Consider {track} to {objective}.",
    "I would explore {track} first so you can {objective}.",
    "For this learning goal, {track} can guide you as you {objective}.",
    "You could start with {track} and learn to {objective}.",
]

TRACK_NAMES = {
    "ai_foundations": "AI Foundations",
    "workflow_automation": "Workflow Automation",
    "content_creation": "AI Content Creation",
    "data_analysis": "AI Data Analysis",
}


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["label", "response"]:
            raise ValueError("Expected CSV columns: label,response")
        rows = [
            {"label": row["label"].strip(), "response": row["response"].strip()}
            for row in reader
            if row["response"].strip()
        ]

    counts = Counter(row["label"] for row in rows)
    unknown = set(counts) - set(OBJECTIVES)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")

    seen = {row["response"].casefold() for row in rows}
    if len(seen) != len(rows):
        raise ValueError("The existing CSV contains duplicate responses")

    for index, (label, objectives) in enumerate(OBJECTIVES.items()):
        candidates = [
            frame.format(track=TRACK_NAMES[label], objective=objective)
            for frame, objective in product(FRAMES, objectives)
        ]
        random.Random(20261101 + index).shuffle(candidates)
        for response in candidates:
            if counts[label] >= TARGET_PER_LABEL:
                break
            key = response.casefold()
            if key in seen:
                continue
            rows.append({"label": label, "response": response})
            seen.add(key)
            counts[label] += 1
        if counts[label] < TARGET_PER_LABEL:
            raise ValueError(f"Not enough distinct responses for {label}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["label", "response"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} responses to {CSV_PATH}")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
