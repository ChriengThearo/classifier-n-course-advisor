"""Expand the starter CSV with reproducible synthetic learning-goal messages.

The generated rows are practice data. Evaluate the classifier separately on
real, independently written messages before trusting its performance.
"""

import csv
import random
from collections import Counter
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "training.csv"
TARGET_PER_LABEL = 500

TOPICS = {
    "ai_foundations": [
        "artificial intelligence basics",
        "the meaning of machine learning",
        "how AI systems learn from examples",
        "the difference between AI and machine learning",
        "the difference between machine learning and deep learning",
        "neural network fundamentals",
        "what a language model is",
        "how chatbots generate answers",
        "the basics of generative AI",
        "common AI terms and definitions",
        "how training data affects an AI model",
        "the difference between training and prediction",
        "what supervised learning means",
        "what unsupervised learning means",
        "how classification models work",
        "how an AI model is evaluated",
        "what overfitting means in machine learning",
        "why AI sometimes makes mistakes",
        "the limits of current AI tools",
        "responsible use of AI",
        "AI bias and fairness basics",
        "privacy concerns when using AI",
        "how to write a clear AI prompt",
        "the difference between a prompt and model training",
        "how to choose an AI tool as a beginner",
        "first steps with everyday AI tools",
        "how computer vision works at a basic level",
        "how speech recognition works",
        "natural language processing basics",
        "the role of data in artificial intelligence",
        "how algorithms recognize patterns",
        "what model parameters are",
        "what an AI agent does",
        "the basics of retrieval augmented generation",
        "what fine tuning changes in a model",
        "how recommendation systems work",
        "the difference between rules and learned models",
        "how to read AI model results",
        "an introduction to Python for AI",
        "basic mathematics used in machine learning",
        "what features and labels mean",
        "how a training and test split works",
        "why data quality matters for AI",
        "how to spot unreliable AI answers",
        "the history of artificial intelligence",
        "AI concepts for someone with no technical background",
        "the difference between narrow AI and general AI",
        "how transformer models process text",
        "what embeddings represent",
        "how to start learning AI from zero",
    ],
    "workflow_automation": [
        "automating weekly office reports",
        "sending automatic email reminders",
        "routing incoming support tickets",
        "moving form submissions into a spreadsheet",
        "connecting a form to a customer database",
        "automating invoice approvals",
        "setting up a recurring task workflow",
        "reducing repetitive copy and paste work",
        "automating appointment confirmations",
        "sending follow ups after a meeting",
        "processing incoming email attachments",
        "organizing files automatically",
        "syncing contacts between business apps",
        "building a workflow with n8n",
        "creating a Zapier automation",
        "using Python scripts for repetitive tasks",
        "triggering actions when a form is submitted",
        "automating data entry between systems",
        "scheduling routine notifications",
        "automating customer onboarding steps",
        "generating routine documents from a template",
        "extracting information from invoices",
        "sending order status updates automatically",
        "creating an approval workflow for requests",
        "automating calendar invitations",
        "linking a CRM to an email platform",
        "setting up webhooks between applications",
        "monitoring a mailbox and starting a workflow",
        "automating inventory alerts",
        "assigning leads to sales staff automatically",
        "processing expense claims with a workflow",
        "automating daily file backups",
        "sending alerts when a task is overdue",
        "creating a no code business process",
        "turning spreadsheet updates into notifications",
        "automating document review steps",
        "creating a chatbot that routes requests",
        "setting up conditional workflow branches",
        "automating purchase request approvals",
        "collecting survey responses without manual entry",
        "automating meeting note distribution",
        "moving new website leads into a CRM",
        "setting up a task queue for repeated jobs",
        "automating employee leave requests",
        "connecting messaging apps with office tools",
        "creating a workflow for customer complaints",
        "automating receipt filing",
        "using APIs to connect two services",
        "building reliable scheduled automations",
        "replacing manual handoffs between teams",
    ],
    "content_creation": [
        "writing social media captions",
        "planning a month of social posts",
        "drafting blog articles",
        "creating product descriptions",
        "writing marketing emails",
        "developing advertising copy",
        "creating short video scripts",
        "writing a podcast episode outline",
        "designing presentation slides",
        "generating ideas for a campaign",
        "writing website landing page copy",
        "making educational infographics",
        "creating an Instagram carousel",
        "drafting a newsletter",
        "rewriting content for a different audience",
        "improving the tone of brand messages",
        "creating a content calendar",
        "writing search friendly articles",
        "making captions for short videos",
        "turning notes into a polished article",
        "creating images for online posts",
        "writing a webinar description",
        "drafting a press announcement",
        "writing a customer success story",
        "creating a brochure for a service",
        "developing a brand voice guide",
        "writing a script for a product demo",
        "creating engaging headlines",
        "drafting event promotion posts",
        "writing a FAQ page for customers",
        "making thumbnail text for videos",
        "creating promotional video storyboards",
        "writing LinkedIn thought leadership posts",
        "drafting a course description",
        "editing an article for clarity",
        "adapting one story for several channels",
        "writing Facebook ad variations",
        "creating customer testimonial posts",
        "turning an interview into an article",
        "writing clear calls to action",
        "creating a visual campaign concept",
        "drafting messages for a product launch",
        "writing an explainer video narration",
        "creating a series of tutorial posts",
        "making a flyer for an event",
        "writing an email subject line",
        "creating a script for a live stream",
        "planning content for a small business",
        "writing descriptions for an online store",
        "turning research notes into a newsletter",
    ],
    "data_analysis": [
        "analyzing monthly sales figures",
        "finding trends in a spreadsheet",
        "building a business dashboard",
        "visualizing customer data",
        "comparing revenue across quarters",
        "cleaning a messy dataset",
        "calculating customer retention rates",
        "exploring survey results",
        "measuring campaign performance",
        "using pivot tables for reports",
        "writing SQL queries for analysis",
        "creating charts from spreadsheet data",
        "finding unusual values in business data",
        "forecasting next month's demand",
        "understanding basic statistics",
        "comparing sales by region",
        "tracking key performance indicators",
        "analyzing website traffic",
        "segmenting customers by behavior",
        "interpreting an A/B test",
        "measuring average order value",
        "understanding correlations in data",
        "finding patterns in transaction records",
        "making a report from survey responses",
        "analyzing product return rates",
        "comparing expenses against budget",
        "using Python pandas for analysis",
        "creating a time series chart",
        "checking data for missing values",
        "summarizing a large CSV file",
        "measuring employee attendance trends",
        "analyzing course completion rates",
        "exploring customer feedback scores",
        "comparing performance before and after a change",
        "building a report of monthly metrics",
        "analyzing inventory turnover",
        "identifying the best selling products",
        "understanding a distribution of values",
        "designing a useful data visualization",
        "turning raw data into insights",
        "estimating future revenue from past data",
        "examining seasonal patterns",
        "calculating conversion rates",
        "combining tables for analysis",
        "checking whether data supports a claim",
        "interpreting a statistical summary",
        "analyzing support response times",
        "comparing customer groups",
        "finding why a metric changed",
        "reading and explaining dashboard results",
    ],
}

FRAMES = [
    "I want to learn about {topic}.",
    "Can you recommend a course about {topic}?",
    "I am looking for a lesson on {topic}.",
    "Where can I study {topic}?",
    "What is a good starting point for {topic}?",
    "I need practical training on {topic}.",
    "Please suggest a learning path covering {topic}.",
    "I would take a workshop focused on {topic}.",
    "Which class teaches {topic}?",
    "I need a course that explains {topic}.",
    "I am trying to learn about {topic} for work.",
    "Do you have lessons about {topic}?",
    "I am interested in {topic}.",
    "My team needs training on {topic}.",
    "I have questions about {topic}.",
    "My next project involves {topic}.",
    "I am completely new to {topic}.",
    "I need help with {topic}.",
    "Could you guide me through {topic}?",
    "I want to understand {topic} better.",
]


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["text", "label"]:
            raise ValueError("Expected CSV columns: text,label")
        existing = [
            {"text": row["text"].strip(), "label": row["label"].strip()}
            for row in reader
            if row["text"].strip()
        ]

    counts = Counter(row["label"] for row in existing)
    unknown_labels = set(counts) - set(TOPICS)
    if unknown_labels:
        raise ValueError(f"Unknown labels: {sorted(unknown_labels)}")

    seen = {}
    for row in existing:
        key = row["text"].casefold()
        if key in seen and seen[key] != row["label"]:
            raise ValueError(f"Conflicting labels for: {row['text']}")
        seen[key] = row["label"]

    output = existing[:]
    for index, (label, topics) in enumerate(TOPICS.items()):
        candidates = [frame.format(topic=topic) for topic, frame in product(topics, FRAMES)]
        random.Random(20261101 + index).shuffle(candidates)
        for text in candidates:
            if counts[label] >= TARGET_PER_LABEL:
                break
            key = text.casefold()
            if key in seen:
                continue
            output.append({"text": text, "label": label})
            seen[key] = label
            counts[label] += 1
        if counts[label] < TARGET_PER_LABEL:
            raise ValueError(f"Not enough unique examples for {label}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(output)

    print(f"Wrote {len(output)} examples to {CSV_PATH}")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
