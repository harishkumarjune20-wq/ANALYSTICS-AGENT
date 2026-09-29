import os
import re
from openai import OpenAI


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "nvidia/nemotron-3.5-lightning-30b-a3b"

# SQL generation settings.
# Temperature is kept deterministic for executable SQL.
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 500
REQUEST_TIMEOUT = 180


# ==================================================
# HELPERS
# ==================================================
def _get_nvidia_api_key():
    """Read the NVIDIA API key from environment or Streamlit secrets."""
    api_key = os.getenv("NVIDIA_API_KEY")

    if api_key:
        return api_key.strip()

    # Optional Streamlit Secrets support for deployment environments.
    try:
        import streamlit as st

        secret_key = st.secrets.get("NVIDIA_API_KEY")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass

    return ""


def _is_probable_sql(text):
    """Return True when text looks like a complete SQL statement."""
    if not isinstance(text, str):
        return False

    candidate = text.strip()
    if not re.match(r"^(SELECT|WITH)\b", candidate, re.IGNORECASE):
        return False

    # A WITH statement must contain a CTE definition or eventually a SELECT.
    if re.match(r"^WITH\s+GROUP\s+BY\b", candidate, re.IGNORECASE):
        return False

    if re.match(r"^WITH\s+(ORDER\s+BY|HAVING|FROM|WHERE|JOIN)\b", candidate, re.IGNORECASE):
        return False

    # Ignore parentheses inside quoted strings for this lightweight check.
    in_single_quote = False
    depth = 0
    i = 0
    while i < len(candidate):
        char = candidate[i]

        if char == "'":
            if in_single_quote and i + 1 < len(candidate) and candidate[i + 1] == "'":
                i += 2
                continue
            in_single_quote = not in_single_quote
        elif not in_single_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return False
        i += 1

    if in_single_quote or depth != 0:
        return False

    # A CTE-based query should contain a SELECT after the WITH clause.
    if re.match(r"^WITH\b", candidate, re.IGNORECASE) and not re.search(
        r"\bSELECT\b", candidate, re.IGNORECASE
    ):
        return False

    return True


def _extract_sql_from_fences(sql_response):
    """Extract SQL from fenced blocks, preferring the last valid SQL block."""
    fenced_blocks = re.findall(
        r"```(?:sql|SQL)?\s*(.*?)```",
        sql_response,
        flags=re.DOTALL,
    )

    valid_blocks = []
    for block in fenced_blocks:
        block = block.strip()
        if _is_probable_sql(block):
            valid_blocks.append(block)

    if valid_blocks:
        return valid_blocks[-1]

    return ""


def clean_sql(sql_response):
    """Extract exactly one executable SQL statement from a Nemotron response.

    Nemotron can occasionally include reasoning or multiple SQL drafts even
    when instructed to return SQL only.  We therefore prefer the LAST valid
    fenced SQL block and only fall back to SQL that starts at the beginning
    of a line.  This prevents text such as 'with GROUP BY...' in reasoning
    from being mistaken for the real SQL query.
    """
    if not sql_response:
        return ""

    sql_response = str(sql_response).strip()

    # Remove explicit reasoning blocks first.
    sql_response = re.sub(
        r"<think>.*?</think>",
        "",
        sql_response,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # 1. Prefer the last valid fenced SQL block. Nemotron may show several
    # drafts; the last valid block is normally its final/revised query.
    fenced_sql = _extract_sql_from_fences(sql_response)
    if fenced_sql:
        sql_response = fenced_sql
    else:
        # 2. Defensive fallback: find SQL only when SELECT/WITH begins a line.
        # Do not match 'with GROUP BY' embedded in natural-language reasoning.
        starts = list(
            re.finditer(
                r"(?im)^\s*(SELECT|WITH)\b",
                sql_response,
            )
        )

        candidates = []
        with_candidates = []
        select_candidates = []

        for match in starts:
            candidate = sql_response[match.start():].strip()
            if ";" in candidate:
                candidate = candidate.split(";", 1)[0].strip()

            if not _is_probable_sql(candidate):
                continue

            candidates.append(candidate)

            # A final SELECT inside a CTE query is also a valid SQL-looking
            # candidate. Never choose that inner SELECT when the complete
            # WITH query is available, otherwise CTE definitions are lost.
            if re.match(r"^WITH\b", candidate, re.IGNORECASE):
                with_candidates.append(candidate)
            else:
                select_candidates.append(candidate)

        if with_candidates:
            # Prefer the last complete WITH query. This preserves all CTEs.
            sql_response = with_candidates[-1]
        elif select_candidates:
            sql_response = select_candidates[-1]
        elif candidates:
            sql_response = candidates[-1]
        else:
            return ""

    # Keep the first SQL statement only.
    if ";" in sql_response:
        sql_response = sql_response.split(";", 1)[0].strip()

    sql_response = sql_response.strip()
    if not _is_probable_sql(sql_response):
        return ""

    return sql_response + ";"


def _convert_postgres_postfix_casts(sql):
    """Convert PostgreSQL-style postfix casts to Athena CAST() syntax.

    Handles both simple identifiers and parenthesized expressions, e.g.
    ``amount::DECIMAL`` and ``NULLIF(SUM(x), 0)::DECIMAL``.
    """
    if not isinstance(sql, str) or not sql:
        return sql

    cast_pattern = re.compile(
        r"::\s*(DECIMAL|NUMERIC)(?:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\))?",
        flags=re.IGNORECASE,
    )

    while True:
        match = cast_pattern.search(sql)
        if not match:
            break

        type_name = match.group(1).upper()
        precision = match.group(2)
        scale = match.group(3)
        if precision and scale:
            target_type = f"DECIMAL({precision}, {scale})"
        else:
            target_type = "DECIMAL(38, 10)"

        end = match.start()
        i = end - 1
        while i >= 0 and sql[i].isspace():
            i -= 1

        if i < 0:
            break

        if sql[i] == ')':
            depth = 1
            j = i - 1
            while j >= 0:
                ch = sql[j]
                if ch == ')':
                    depth += 1
                elif ch == '(':
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            if j < 0:
                break
            expr_start = j
            # Include a function/identifier immediately before the opening
            # parenthesis, e.g. SUM(...) or NULLIF(...).
            k = expr_start - 1
            while k >= 0 and sql[k].isspace():
                k -= 1
            while k >= 0 and (sql[k].isalnum() or sql[k] in '._"'):
                k -= 1
            if k + 1 < expr_start:
                expr_start = k + 1
            expr = sql[expr_start:i + 1]
        else:
            j = i
            while j >= 0 and (sql[j].isalnum() or sql[j] in '._"'):
                j -= 1
            expr_start = j + 1
            expr = sql[expr_start:i + 1]

        replacement = f"CAST({expr.strip()} AS {target_type})"
        sql = sql[:expr_start] + replacement + sql[match.end():]

    return sql


def fix_sql(sql):
    """Apply only generic Athena-compatible syntax cleanup."""
    if not isinstance(sql, str):
        return ""

    # Athena/Trino does not support ILIKE.
    sql = re.sub(
        r"(\w+(?:\.\w+)?)\s+ILIKE\s+'(.*?)'",
        r"LOWER(\1) LIKE LOWER('\2')",
        sql,
        flags=re.IGNORECASE,
    )

    # Defensive conversion for PostgreSQL postfix casts. Nemotron is still
    # instructed to generate CAST() directly; this only protects execution
    # if the model nevertheless emits ::DECIMAL/::NUMERIC.
    sql = _convert_postgres_postfix_casts(sql)

    return sql.strip()


def _validate_basic_sql_shape(sql):
    """Catch obvious malformed SQL before sending it to Athena."""
    if not sql or not sql.strip():
        raise ValueError("LLM returned an empty SQL response.")

    # Ignore parentheses inside quoted string literals for this lightweight
    # structural check.
    in_single_quote = False
    depth = 0
    i = 0

    while i < len(sql):
        char = sql[i]

        if char == "'":
            # SQL escapes a single quote by doubling it.
            if in_single_quote and i + 1 < len(sql) and sql[i + 1] == "'":
                i += 2
                continue
            in_single_quote = not in_single_quote

        elif not in_single_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError(
                        "Generated SQL has an unexpected closing parenthesis."
                    )

        i += 1

    if in_single_quote:
        raise ValueError("Generated SQL contains an unterminated string literal.")

    if depth != 0:
        raise ValueError(
            "Generated SQL has unbalanced parentheses."
        )

    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError(
            "Generated SQL must start with SELECT or WITH."
        )

    if not _is_probable_sql(sql):
        raise ValueError(
            "Generated SQL does not have a valid SQL statement structure."
        )


# ==================================================
# SQL GENERATION
# ==================================================
def generate_sql(question, schema):
    """Generate Athena SQL with NVIDIA Nemotron 3.5 Lightning 30B-A3B."""
    api_key = _get_nvidia_api_key()

    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not configured. "
            "Set NVIDIA_API_KEY in the environment or Streamlit secrets."
        )

    prompt = f"""
You are the SQL generation engine for the AI-Powered Analytics Agent.

Generate ONE complete, executable AWS Athena SQL query for the user's
analytics question.

Return ONLY the final SQL query.
Do NOT return reasoning.
Do NOT return markdown.
Do NOT return ```sql fences.
Do NOT return explanations.
Do NOT return multiple SQL statements.

SCHEMA:
{schema}

USER QUESTION:
{question}

ATHENA SQL RULES:
1. Use ONLY tables and columns that exist in the supplied schema.
2. Use AWS Athena / Trino SQL syntax only.
3. The table is normally:
   "<database>"."<table>"
   unless the supplied schema explicitly indicates otherwise.
4. year and month are INTEGER columns. Never use year(month_year) or
   month(month_year).
5. Month names must map to:
   January=1, February=2, March=3, April=4, May=5, June=6,
   July=7, August=8, September=9, October=10, November=11, December=12.
6. For previous/current calendar periods use Athena date_add() correctly.
7. Never use PostgreSQL :: casting syntax. Always use CAST(expression AS DECIMAL(...)) or another valid Athena CAST.
8. Never use INTERVAL.
9. Never use DATE_TRUNC for previous month/year logic.
10. Never add or subtract dates using + or -.
11. Never apply date_add() to integer year/month columns.
12. For period comparisons, aggregate each period correctly before comparing.
13. For monthly results, include both year and month when both are available.
14. Complete every SELECT, FROM, WHERE, GROUP BY, HAVING, JOIN, CTE,
    ORDER BY and LIMIT clause correctly.
15. Every column referenced outside a CTE must be selected by that CTE.
16. Every non-aggregated selected column must be compatible with GROUP BY.
17. Every alias must be defined before it is referenced.
18. Never invent a column, table, alias, status value, or business field.
19. 19. BOOKING-BASED RATES:
    When a metric is booking-based, always use SUM(number_of_bookings),
    never COUNT(*).

    For rates/percentages:
    numerator = SUM(number_of_bookings) for the requested condition
    denominator = SUM(number_of_bookings) for the full population.

    For a rate by a dimension (supplier, month, year, etc.), GROUP BY only
    the requested dimension. Do NOT GROUP BY the condition column such as
    status.

    Example confirmation rate:
    100.0 * SUM(CASE WHEN LOWER(status) = 'confirmed'
                     THEN number_of_bookings ELSE 0 END)
    / NULLIF(SUM(number_of_bookings), 0)

    Apply this generically to all booking-based rates.
    
    - Preserve SELECT * when explicitly requested.
- Do not introduce aggregations (SUM, COUNT, AVG, etc.) unless the user asks for totals, summaries, trends, KPIs, or grouped results.
- Do not add GROUP BY clauses unless aggregation is required.
- Do not replace SELECT * with dimension columns.
- Generate row-level queries when the request is a simple filter.

20. Protect rate/division calculations with NULLIF(..., 0) where appropriate. NULLIF always has exactly two arguments.
21. Use functions such as LOWER(column) only around the actual string expression; never create forms such as alias.LOWER(column).
22. ROUND must have valid Athena arguments, such as ROUND(expression, 2), with all parentheses closed.
23. For "last N months", reason about the latest/required periods correctly
    and handle year boundaries.
24. Balance all parentheses before returning the SQL.
25. Return exactly ONE executable SQL statement ending in a semicolon.
ATHENA DATE/TIME SYNTAX — STRICT RULES:

1. Generate SQL strictly using AWS Athena / Trino SQL syntax.

2. For truncating DATE or TIMESTAMP values to a calendar boundary,
   use:
       date_trunc('month', date_expression)
       date_trunc('year', date_expression)

3. NEVER use TRUNCATE() for DATE or TIMESTAMP values.
   TRUNCATE() must not be used as a replacement for date_trunc().

4. NEVER generate:
       TRUNCATE(CURRENT_DATE, 'MONTH')
       TRUNCATE(CURRENT_DATE, 'YEAR')
       TRUNCATE(date_column, 'MONTH')
       TRUNCATE(date_column, 'YEAR')

5. NEVER use PostgreSQL-style or non-Athena date syntax.

6. For relative calendar periods, use Athena date_add() with a date
   expression, for example:
       date_add('month', -3, date_trunc('month', current_date))

7. For previous complete months, use a half-open date range:
       date_column >= date_add('month', -N, date_trunc('month', current_date))
       AND date_column < date_trunc('month', current_date)

8. For previous complete years, use:
       date_column >= date_add('year', -N, date_trunc('year', current_date))
       AND date_column < date_trunc('year', current_date)

9. Do not use INTERVAL syntax.

10. Do not add or subtract DATE/TIMESTAMP values using + or -.

11. Do not apply date_add() to integer year or month columns.
    The year and month columns in this schema are INTEGER values.

12. If the table provides separate INTEGER year/month columns and the
    user's question can be answered using those columns, prefer the
    year/month columns rather than inventing date expressions.

13. Before returning SQL, verify every date function against Athena/
    Trino syntax.
14. print only sql no other text.    
Before returning the SQL, silently verify:
- CTE scope
- aliases
- GROUP BY
- JOIN conditions
- parentheses
- Athena syntax
- schema columns
- requested time periods
- requested metric definition
""".strip()

    # NVIDIA provides an OpenAI-compatible API. Keep the API call isolated
    # here so the rest of the application architecture remains unchanged.
    try:
        client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=api_key,
            timeout=REQUEST_TIMEOUT,
        )

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise AWS Athena SQL generator. "
                        "Return only executable SQL."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
            stream=True,
        )

        # The NVIDIA example uses streaming. Accumulate all content before
        # cleaning/validating because Athena must receive one complete SQL
        # statement.
        response_parts = []
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content is not None:
                response_parts.append(content)

        raw_response = "".join(response_parts).strip()

    except Exception as exc:
        raise RuntimeError(
            f"NVIDIA Nemotron API request failed: {exc}"
        ) from exc

    if not raw_response:
        raise ValueError(
            "NVIDIA Nemotron returned an empty SQL response."
        )

    print("\n================ RAW NEMOTRON RESPONSE ================\n")
    print(raw_response)

    cleaned_sql = clean_sql(raw_response)
    cleaned_sql = fix_sql(cleaned_sql)

    # IMPORTANT: Athena is the authority for SQL validity.
    # Do not reject generated SQL with a local syntax/shape validator before
    # sending it to Athena. If Athena rejects it, the app will show the real
    # Athena error and the recovery path can handle it.

    print("\n================ FINAL SQL ================\n")
    print(cleaned_sql)

    return cleaned_sql


# Keep the existing public fallback wrapper/interface.
# No unsafe SELECT * fallback is generated here.
def generate_sql_with_fallback(question, schema):
    try:
        return generate_sql(question, schema)
    except Exception as exc:
        raise RuntimeError(
            f"Nemotron SQL generation failed: {exc}"
        ) from exc
