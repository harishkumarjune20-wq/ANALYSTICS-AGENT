import pandas as pd
import re

from utils.ollama_client import generate_with_qwen


# =====================================================
# CLEAN MODEL RESPONSE
# =====================================================

def clean_model_response(text):

    if not text:
        return ""

    text = str(text).strip()

    # Remove accidental thinking tags.

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = re.sub(
        r"</?think>",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove markdown code fences if the model adds them.
    text = text.replace(
        "```text",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    return text.strip()


# =====================================================
# LLM ANALYTICS INSIGHTS / SUMMARY
# =====================================================

def _dataframe_for_llm(df, max_rows=80):
    """Convert an analytics result dataframe into compact LLM-readable text."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "No rows were returned."

    safe_df = df.copy().head(max_rows)

    for col in safe_df.columns:
        safe_df[col] = safe_df[col].map(
            lambda value: "" if pd.isna(value) else str(value)
        )

    return safe_df.to_string(index=False)


def _extract_response_text(result, fallback=""):
    """Extract plain response text from the selected AI provider result."""
    if isinstance(result, str):
        return result.strip() or fallback

    if isinstance(result, dict):
        response = result.get("response")

        if isinstance(response, str):
            return response.strip() or fallback

        if isinstance(response, dict):
            for key in ("response", "answer", "content", "message"):
                value = response.get(key)

                if isinstance(value, str) and value.strip():
                    return value.strip()

        for key in ("answer", "content", "message"):
            value = result.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

    return fallback


def generate_llm_insights_summary(question, df):
    """Generate business insights and summary from the actual analytics result."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return (
            "• No data was returned for analysis.",
            "No data was available for this question."
        )

    data_text = _dataframe_for_llm(df, max_rows=80)

    prompt = f"""
You are the Analytics Agent Business Analyst.

Analyze ONLY the analytics result table supplied below.

User question:
{question}

Returned analytics table:
{data_text}

Generate:

INSIGHTS:
3 to 6 concise bullet points containing meaningful,
data-grounded business observations.

SUMMARY:
2 to 4 concise sentences explaining what the returned
result means.

Rules:
- Use ONLY the returned table and the user's question.
- Never invent numbers, dates, months, suppliers, routes,
  causes, or trends.
- Calculate totals, averages, highest/lowest values, changes,
  shares, or comparisons only from values actually present
  in the returned table.
- If monthly data is present, explicitly identify the highest
  and lowest month when this can be determined from the table.
- If supplier/category data is present, identify the strongest
  and weakest supplier/category when this can be determined.
- If there is only one returned metric/value, explain exactly
  what it shows.
- If the data is insufficient for a deeper conclusion, say so.
- Use readable numbers and do not use scientific notation.
- Do not repeat the complete table.
- Do not mention SQL, AWS, Athena, Python, code, Ollama,
  models, prompts, or internal implementation.
- Do not duplicate the same fact in multiple bullets.

Return EXACTLY:

INSIGHTS:
• ...

SUMMARY:
...
"""

    try:
        answer = generate_with_qwen(
            prompt,
            temperature=0.1,
            max_tokens=160,
            think=False,
            timeout=60,
        )

        text = _extract_response_text(
            answer,
            "INSIGHTS:\n• Unable to generate LLM insights.\n\n"
            "SUMMARY:\nThe analytics result is available, "
            "but an automatic summary could not be generated."
        )

        match = re.search(
            r"INSIGHTS\s*:\s*(.*?)\s*SUMMARY\s*:\s*(.*)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:
            insights = match.group(1).strip()
            summary = match.group(2).strip()

            if insights and summary:
                return insights, summary

    except Exception as error:
        print("LLM insights/summary generator error:", error)

    return (
        "• Unable to generate LLM insights.",
        "The analytics result is available, but an automatic "
        "summary could not be generated."
    )


def generate_response(question, df):
    """Return the same LLM-driven insights/summary structure used by the apps."""
    insights, summary = generate_llm_insights_summary(question, df)

    return {
        "insights": insights,
        "summary": summary,
    }


# =====================================================
# BUSINESS RECOMMENDATION
# =====================================================

def generate_recommendation(
    question,
    df,
    previous_question=""
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame
        )
        or df.empty
    ):

        return (
            "No previous analytics result is available "
            "for a recommendation."
        )

    data_text = _dataframe_for_llm(
        df
    )

    prompt = f"""
You are the business analytics recommendation component
of the AI-Powered Analytics Agent.

Your job is to analyze ONLY the provided previous analytics
result and give practical recommendations that can help
improve company performance.

Do NOT answer unrelated questions.

Do NOT invent facts that are not present in the data.

Do NOT claim that a recommendation is guaranteed to work.

If the data is insufficient for a specific recommendation,
clearly say that more data is needed.

Previous analytics question:
{previous_question}

User's recommendation question:
{question}

Previous analytics result:
{data_text}

Instructions:

1. Analyze the provided result carefully.
2. Identify positive and negative performance patterns.
3. Focus on the metric relevant to the user's question.
4. Give practical actions the company can take.
5. If the result contains monthly sales or month-over-month
   information, discuss the months showing weakness or decline.
6. If growth is negative, suggest actions to investigate
   the cause and improve performance.
7. If growth is positive, suggest ways to maintain or expand it.
8. Base recommendations only on the provided result.
9. Keep the response concise and business-focused.
10. Use bullet points.
11. Do not mention SQL, Athena, Python, Ollama, models,
    prompts, or internal implementation.

Return only the recommendation.
"""

    try:
        answer = generate_with_qwen(
            prompt,
            temperature=0.1,
            max_tokens=200,
            timeout=60,
            think=False,
        )

        if answer:
            return answer

    except Exception as error:
        print(
            "Recommendation generator error:",
            error
        )

    # =================================================
    # SAFE FALLBACK
    # =================================================

    return generate_recommendation_fallback(
        df,
        question
    )


# =====================================================
# FALLBACK RECOMMENDATION
# =====================================================

def generate_recommendation_fallback(
    df,
    question=""
):

    if (
        df is None
        or df.empty
    ):

        return (
            "No previous analytics data is available "
            "to generate recommendations."
        )

    recommendations = []

    columns = {
        str(col).lower(): col
        for col in df.columns
    }

    # =================================================
    # MOM GROWTH
    # =================================================

    mom_column = None

    for column_name, original_name in columns.items():

        if (
            "mom_growth" in column_name
            or "growth_percentage" in column_name
            or "growth" in column_name
        ):

            mom_column = original_name
            break

    if mom_column is not None:

        values = pd.to_numeric(
            df[mom_column],
            errors="coerce"
        ).dropna()

        if not values.empty:

            growth = values.iloc[-1]

            if growth < 0:

                recommendations.append(
                    "• Investigate the drivers of the sales decline "
                    "and focus corrective actions on the affected "
                    "period or segment."
                )

                recommendations.append(
                    "• Review booking volume, supplier performance, "
                    "routes, and cancellations to identify the "
                    "main contributors to the decline."
                )

            elif growth > 0:

                recommendations.append(
                    "• Maintain the factors contributing to positive "
                    "sales growth and identify opportunities to "
                    "replicate them across weaker periods."
                )

                recommendations.append(
                    "• Compare high-performing periods with weaker "
                    "periods to identify repeatable sales drivers."
                )

            else:

                recommendations.append(
                    "• Sales growth is currently flat. Review "
                    "booking volume, supplier mix, routes, and "
                    "customer demand to identify opportunities."
                )

    # =================================================
    # SALES
    # =================================================

    sales_column = None

    for column_name, original_name in columns.items():

        if (
            "total_sales" in column_name
            or column_name == "sales"
            or "revenue" in column_name
        ):

            sales_column = original_name
            break

    if sales_column is not None:

        recommendations.append(
            "• Monitor sales performance regularly and compare "
            "weaker periods against stronger periods to identify "
            "specific improvement opportunities."
        )

    # =================================================
    # GENERAL FALLBACK
    # =================================================

    if not recommendations:

        recommendations.append(
            "• Review the strongest and weakest values in the "
            "previous analytics result and prioritize actions "
            "on the weakest performance areas."
        )

        recommendations.append(
            "• Compare the result with booking, supplier, route, "
            "and cancellation metrics to identify the operational "
            "drivers behind the performance."
        )

        recommendations.append(
            "• Continue monitoring the same metrics after taking "
            "action so that improvement can be measured."
        )

    return "\n".join(
        recommendations
    )