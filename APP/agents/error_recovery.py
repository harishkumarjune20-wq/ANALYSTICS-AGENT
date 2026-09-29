"""Nemotron-based Athena SQL error recovery."""

import re

from utils.nemotron_client import generate_with_nemotron


def _clean_repaired_sql(response):
    """Extract one executable SQL statement from Nemotron output."""
    if not response:
        return ""

    text = re.sub(
        r"<think>.*?</think>",
        "",
        str(response),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    fenced = re.findall(
        r"```(?:sql)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        text = fenced[-1].strip()

    match = re.search(r"(?is)\b(?:SELECT|WITH)\b", text)
    if match:
        text = text[match.start():].strip()

    if ";" in text:
        text = text.split(";", 1)[0].strip()

    return text


def fix_sql(failed_sql, error_message, schema):
    """Repair failed Athena SQL using NVIDIA Nemotron."""
    prompt = f"""
You are repairing an Athena SQL query.

Schema:
{schema}

The following Athena SQL failed:
{failed_sql}

Error returned by validation or Athena:
{error_message}

Generate one corrected Athena SQL statement that preserves the original
analytical intent and uses only the supplied schema.

Rules:
- Return ONLY the corrected SQL.
- Do not explain the correction.
- Do not invent tables or columns.
- Keep the query executable in Amazon Athena.
""".strip()

    response = generate_with_nemotron(
        prompt,
        temperature=0.1,
        max_tokens=700,
        timeout=120,
    )
    sql = _clean_repaired_sql(response)

    if not sql:
        raise RuntimeError("Nemotron did not return a valid SQL repair.")

    return sql
