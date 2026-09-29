import requests
import json
import re


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen3:4b"


def generate_suggestions(previous_questions=None):

    if previous_questions is None:
        previous_questions = []

    previous_text = "\n".join(
        f"- {q}" for q in previous_questions[-15:]
    )

    prompt = f"""
You are the suggestion generator for the AI-Powered Analytics Agent.

Generate exactly 3 NEW questions that the user can ask this agent.

The agent can help with:

- AWS business data
- Amazon Athena
- Revenue analysis
- Sales analysis
- Booking analysis
- Airline analysis
- Route analysis
- Supplier analysis
- Brand analysis
- Segment analysis
- Cancellation analysis
- Business performance
- Trends and comparisons
- Business insights
- Business recommendations
- Graphs and visualizations

Rules:

1. Generate exactly 3 questions.
2. Questions must be useful and realistic.
3. Do NOT repeat previous questions.
4. Keep questions short.
5. Questions must be related to the capabilities listed above.
6. Do NOT explain anything.
7. Do NOT provide answers.
8. Return ONLY valid JSON.
9. Do NOT use markdown or code fences.

Previous questions:
{previous_text}

Return exactly this format:

{{
    "questions": [
        "question 1",
        "question 2",
        "question 3"
    ]
}}
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,

                # IMPORTANT:
                # Qwen3 uses "think", not "thinking"
                "think": False,

                "keep_alive": "10m",

                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 150
                },

                # Ask Ollama for JSON
                "format": "json"
            },
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        print("SUGGESTION OLLAMA RESULT:", result)

        # Qwen3 should now return the JSON here
        raw = result.get("response", "").strip()

        if not raw:

            # Safety fallback in case the model still puts
            # something in the thinking field
            thinking = result.get("thinking", "").strip()

            if thinking:
                raw = thinking

        if not raw:
            raise ValueError(
                "Suggestion generator returned empty response."
            )

        # Remove accidental markdown fences
        raw = re.sub(
            r"```json|```",
            "",
            raw,
            flags=re.IGNORECASE
        ).strip()

        # Parse JSON
        parsed = json.loads(raw)

        questions = parsed.get("questions", [])

        if not isinstance(questions, list):
            raise ValueError(
                "Invalid questions format."
            )

        # Clean questions
        cleaned_questions = []

        for question in questions:

            if not isinstance(question, str):
                continue

            question = question.strip()

            if question:
                cleaned_questions.append(question)

        # Remove duplicates
        unique_questions = []

        for question in cleaned_questions:

            if question.lower() not in [
                q.lower()
                for q in unique_questions
            ]:
                unique_questions.append(question)

        # Make sure we have 3
        if len(unique_questions) < 3:
            raise ValueError(
                "Model returned fewer than 3 suggestions."
            )

        return unique_questions[:3]

    except Exception as error:

        print(
            "Suggestion generator error:",
            error
        )

        # Safe fallback
        return [
            "Which airline generates the highest revenue?",
            "Which routes have high booking volume but low revenue?",
            "Show me the monthly booking trend."
        ]