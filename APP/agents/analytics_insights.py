"""Shared deterministic your data environment analytics insights/summary logic.
Extracted from the working Streamlit implementation so LangGraph and UI use the same behavior."""

import re
import pandas as pd
import json
from utils.nemotron_client import generate_with_nemotron

def _format_value(value):
    """Format numeric values consistently for business-facing text."""
    try:
        value = float(value)
        if value.is_integer():
            return f"{value:,.0f}"
        return f"{value:,.2f}"
    except (TypeError, ValueError):
        return str(value)

def _friendly_period(value, month_col=None, include_year=True):
    """Convert month values such as 1 or 'month 1' into readable labels."""
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    text = str(value).strip()
    match = re.fullmatch(r"month\s*(\d{1,2})", text, flags=re.IGNORECASE)
    if match:
        number = int(match.group(1))
        return month_names.get(number, text)

    match = re.fullmatch(r"(\d{4})-(\d{1,2})", text)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        if month in month_names:
            return f"{
    month_names[month]} {year}" if include_year else month_names[month]

    try:
        number = int(float(value))
        if number in month_names and month_col and "month" in str(
            month_col).lower():
            return month_names[number]
    except (TypeError, ValueError):
        pass

    return text

def _deep_dataframe_analysis(question, df):
    """Create deterministic, data-grounded insights and summary.

    Monthly results receive special treatment: highest/lowest month, total,
    average, range and direction are explained instead of showing only
    column totals. This avoids shallow responses such as '12 rows x 2 columns'.
    """
    if df is None or df.empty:
        return (
            "No insights generated because no data was returned.",
            "No data was available for this question."
        )

    work = df.copy()
    month_col = _find_month_column(work)
    year_col = _find_year_column(work)
    metric_col = _find_metric_column(work)

    # -------------------------------------------------
    # MONTHLY BUSINESS ANALYSIS
    # -------------------------------------------------
    if month_col is not None and metric_col is not None:
        monthly, _, monthly_metric = prepare_monthly_series(work)

        if monthly is not None and len(monthly) >= 2:
            values = pd.to_numeric(monthly[monthly_metric], errors="coerce")
            valid = monthly.loc[values.notna()].copy()
            valid[monthly_metric] = values[values.notna()].astype(
                float).to_numpy()

            if not valid.empty:
                highest_row = valid.loc[valid[monthly_metric].idxmax()]
                lowest_row = valid.loc[valid[monthly_metric].idxmin()]
                highest_period = _friendly_period(
                    highest_row["period_label"], month_col, include_year=year_col is not None
                )
                lowest_period = _friendly_period(
                    lowest_row["period_label"], month_col, include_year=year_col is not None
                )

                total = float(valid[monthly_metric].sum())
                average = float(valid[monthly_metric].mean())
                highest_value = float(highest_row[monthly_metric])
                lowest_value = float(lowest_row[monthly_metric])
                value_range = highest_value - lowest_value

                insights = [
                    f"• Monthly {
    monthly_metric.replace(
        '_', ' ')} were tracked across {
            len(valid)} months.",
                    f"• Highest: {highest_period} with {
    _format_value(highest_value)}.",
                    f"• Lowest: {lowest_period} with {
    _format_value(lowest_value)}.",
                    f"• Total across all months: {_format_value(total)}.",
                    f"• Average per month: {_format_value(average)}.",
                    f"• Difference between the highest and lowest month: {
    _format_value(value_range)}."
                ]

                if len(valid) >= 2:
                    first_value = float(valid[monthly_metric].iloc[0])
                    last_value = float(valid[monthly_metric].iloc[-1])
                    change = last_value - first_value
                    if first_value != 0:
                        change_pct = (change / abs(first_value)) * 100
                        direction = "increased" if change > 0 else "decreased" if change < 0 else "remained stable"
                        insights.append(
                            f"• From {valid['period_label'].iloc[0]} to {valid['period_label'].iloc[-1]}, "
                            f"sales {direction} by {
    _format_value(
        abs(change))} ({
            abs(change_pct):.2f}%)."
                        )

                summary = (
                    f"These are the monthly {
    monthly_metric.replace(
        '_', ' ')} results. "
                    f"The highest month was {highest_period} with {
    _format_value(highest_value)}, "
                    f"while the lowest month was {lowest_period} with {
    _format_value(lowest_value)}. "
                    f"Across the {
    len(valid)} months, the total was {
        _format_value(total)} "
                    f"and the average monthly value was {
    _format_value(average)}."
                )
                return "\n".join(insights), summary

    # -------------------------------------------------
    # CATEGORY + METRIC ANALYSIS
    # -------------------------------------------------
    numeric_candidates = _numeric_columns(work)
    if numeric_candidates:
        metric = metric_col if metric_col in numeric_candidates else numeric_candidates[0]
        numeric_values = pd.to_numeric(work[metric], errors="coerce")
        valid = work.loc[numeric_values.notna()].copy()
        valid[metric] = numeric_values[numeric_values.notna()].astype(
            float).to_numpy()

        if not valid.empty:
            total = float(valid[metric].sum())
            average = float(valid[metric].mean())
            highest = valid.loc[valid[metric].idxmax()]
            lowest = valid.loc[valid[metric].idxmin()]
            category_cols = [
    c for c in _category_columns(work) if c in valid.columns]
            insights = [
                f"• {len(valid)} records analyzed.",
                f"• Highest value: {_format_value(highest[metric])}.",
                f"• Lowest value: {_format_value(lowest[metric])}.",
                f"• Total: {_format_value(total)}.",
                f"• Average: {_format_value(average)}."
                ]
            if category_cols:
                category = category_cols[0]
                insights[1] = (
                    f"• Top {category.replace('_',' ').title()}: "
                    f"{highest[category]} → "
                    f"{_format_value(highest[metric])}"
                )
                insights[2] = (
                    f"• Lowest {category.replace('_',' ').title()}: "
                    f"{lowest[category]} → "
                    f"{_format_value(lowest[metric])}"
                )
                summary = (
                    f"{highest[category]} is the top-performing "
                    f"{category.replace('_',' ')} with "
                    f"{_format_value(highest[metric])}. "
                    f"{lowest[category]} recorded the lowest value at "
                    f"{_format_value(lowest[metric])}. "
                    f"The overall total is {_format_value(total)} "
                    f"with an average of {_format_value(average)}."
                )
            else:
                summary = (
                    f"The dataset contains {len(valid)} records. "
                    f"The highest value is {_format_value(highest[metric])} "
                    f"and the lowest value is {_format_value(lowest[metric])}. "
                    f"The total is {_format_value(total)} "
                    f"with an average of {_format_value(average)}."
                )

            return "\n".join(insights), summary

    return (
        f"• The query returned {
    len(df)} row(s) successfully, but there was not enough numeric data "
        "to calculate deeper performance insights.",
        f"The query returned {
    len(df)} row(s) across {
        len(
            df.columns)} column(s)."
    )

def _json_safe(value):
    """Convert pandas/numpy values into JSON-safe Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _build_factual_context(question, df, forecast_df=None):
    """Build a compact, deterministic factual context for Qwen.

    Nemotron receives calculated facts rather than the raw DataFrame. This keeps
    all numeric grounding in Python and prevents the model from inventing
    values that were not present in the Athena result.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "question": question or "",
            "row_count": 0,
            "columns": [],
            "numeric_metrics": {},
            "dimensions": {},
            "group_comparisons": [],
            "period_analysis": {},
            "forecast": {},
        }

    work = df.copy()
    context = {
        "question": question or "",
        "row_count": int(len(work)),
        "columns": [str(c) for c in work.columns],
        "numeric_metrics": {},
        "dimensions": {},
        "group_comparisons": [],
        "period_analysis": {},
        "forecast": {},
    }

    # -------------------------------------------------
    # Numeric facts: total, average, min, max, range
    # -------------------------------------------------
    numeric_columns = _numeric_columns(work)
    for col in numeric_columns:
        values = pd.to_numeric(work[col], errors="coerce").dropna()
        if values.empty:
            continue

        facts = {
            "count": int(values.count()),
            "total": float(values.sum()),
            "average": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "range": float(values.max() - values.min()),
        }

        if len(values) >= 2:
            first = float(values.iloc[0])
            last = float(values.iloc[-1])
            change = last - first
            facts["first_value"] = first
            facts["last_value"] = last
            facts["change"] = float(change)
            if first != 0:
                facts["percentage_change"] = float(
                    (change / abs(first)) * 100
                )

        context["numeric_metrics"][str(col)] = facts

    # -------------------------------------------------
    # Dimension/category facts
    # -------------------------------------------------
    for col in _category_columns(work):
        series = work[col].dropna().astype(str)
        if series.empty:
            continue

        counts = series.value_counts()
        dimension_facts = {
            "unique_count": int(series.nunique()),
        }

        if len(counts) <= 20:
            dimension_facts["available_values"] = [
                str(v) for v in counts.index.tolist()
            ]

        dimension_facts["top_occurrences"] = [
            {
                "value": str(index),
                "count": int(count),
            }
            for index, count in counts.head(10).items()
        ]
        context["dimensions"][str(col)] = dimension_facts

    # -------------------------------------------------
    # Generic category-vs-metric comparisons
    # -------------------------------------------------
    category_columns = _category_columns(work)
    for category in category_columns[:3]:
        if work[category].nunique(dropna=True) > 20:
            continue

        for metric in numeric_columns[:5]:
            metric_values = pd.to_numeric(work[metric], errors="coerce")
            grouped_work = pd.DataFrame({
                "_category": work[category].astype(str),
                "_metric": metric_values,
            }).dropna(subset=["_metric"])

            if grouped_work.empty:
                continue

            grouped = (
                grouped_work.groupby("_category", dropna=False)["_metric"]
                .agg(["sum", "mean", "count", "min", "max"])
                .sort_values("sum", ascending=False)
            )

            if grouped.empty:
                continue

            context["group_comparisons"].append({
                "dimension": str(category),
                "metric": str(metric),
                "highest": {
                    "category": str(grouped.index[0]),
                    "total": float(grouped.iloc[0]["sum"]),
                    "average": float(grouped.iloc[0]["mean"]),
                    "count": int(grouped.iloc[0]["count"]),
                },
                "lowest": {
                    "category": str(grouped.index[-1]),
                    "total": float(grouped.iloc[-1]["sum"]),
                    "average": float(grouped.iloc[-1]["mean"]),
                    "count": int(grouped.iloc[-1]["count"]),
                },
            })

    # -------------------------------------------------
    # Period/month analysis
    # -------------------------------------------------
    monthly, month_col, metric_col = prepare_monthly_series(work)
    if monthly is not None and metric_col is not None and not monthly.empty:
        values = pd.to_numeric(
            monthly[metric_col], errors="coerce"
        )
        valid = monthly.loc[values.notna()].copy()
        valid[metric_col] = values[values.notna()].astype(float).to_numpy()

        if not valid.empty:
            highest = valid.loc[valid[metric_col].idxmax()]
            lowest = valid.loc[valid[metric_col].idxmin()]
            period_facts = {
                "period_column": str(month_col),
                "metric": str(metric_col),
                "period_count": int(len(valid)),
                "highest_period": str(highest["period_label"]),
                "highest_value": float(highest[metric_col]),
                "lowest_period": str(lowest["period_label"]),
                "lowest_value": float(lowest[metric_col]),
                "total": float(valid[metric_col].sum()),
                "average": float(valid[metric_col].mean()),
                "range": float(
                    highest[metric_col] - lowest[metric_col]
                ),
                "values": [
                    {
                        "period": str(row["period_label"]),
                        "value": float(row[metric_col]),
                    }
                    for _, row in valid.iterrows()
                ],
            }

            if len(valid) >= 2:
                first_value = float(valid[metric_col].iloc[0])
                last_value = float(valid[metric_col].iloc[-1])
                change = last_value - first_value
                period_facts["first_period"] = str(
                    valid["period_label"].iloc[0]
                )
                period_facts["last_period"] = str(
                    valid["period_label"].iloc[-1]
                )
                period_facts["change"] = float(change)
                if first_value != 0:
                    period_facts["percentage_change"] = float(
                        (change / abs(first_value)) * 100
                    )

            context["period_analysis"] = period_facts

    # -------------------------------------------------
    # Forecast facts, when visualization/forecast flow supplies them
    # -------------------------------------------------
    if (
        isinstance(forecast_df, pd.DataFrame)
        and not forecast_df.empty
    ):
        forecast_metric = _find_metric_column(forecast_df)
        if forecast_metric is not None:
            forecast_values = pd.to_numeric(
                forecast_df[forecast_metric], errors="coerce"
            )
            valid_forecast = forecast_df.loc[
                forecast_values.notna()
            ].copy()
            valid_forecast[forecast_metric] = forecast_values[
                forecast_values.notna()
            ].astype(float).to_numpy()

            if not valid_forecast.empty:
                high = valid_forecast.loc[
                    valid_forecast[forecast_metric].idxmax()
                ]
                low = valid_forecast.loc[
                    valid_forecast[forecast_metric].idxmin()
                ]
                context["forecast"] = {
                    "metric": str(forecast_metric),
                    "projection_count": int(len(valid_forecast)),
                    "highest_projected_period": str(
                        high.get("period_label", "")
                    ),
                    "highest_projected_value": float(
                        high[forecast_metric]
                    ),
                    "lowest_projected_period": str(
                        low.get("period_label", "")
                    ),
                    "lowest_projected_value": float(
                        low[forecast_metric]
                    ),
                    "values": [
                        {
                            "period": str(row.get("period_label", "")),
                            "value": float(row[forecast_metric]),
                        }
                        for _, row in valid_forecast.iterrows()
                    ],
                    "note": (
                        "These are projected values supplied by the "
                        "application, not historical actuals."
                    ),
                }

    return context


def _parse_model_insights_summary(answer):
    """Parse Qwen's two-section response without trusting model numbers."""
    if not answer:
        return "", ""

    cleaned = str(answer).strip()
    cleaned = re.sub(
        r"```(?:text|markdown)?", "", cleaned, flags=re.IGNORECASE
    ).replace("```", "").strip()

    insights_match = re.search(
        r"(?:^|\n)\s*(?:INSIGHTS|BUSINESS\s+INSIGHTS)\s*:?\s*"
        r"(.*?)(?=\n\s*(?:SUMMARY)\s*:|\Z)",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    summary_match = re.search(
        r"(?:^|\n)\s*SUMMARY\s*:?\s*(.*)$",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    insights = insights_match.group(1).strip() if insights_match else ""
    summary = summary_match.group(1).strip() if summary_match else ""

    if not insights and not summary:
        return cleaned, ""

    return insights, summary


def _generate_ai_insights_summary(question, factual_context):
    """Ask Nemotron to interpret Python-calculated facts only."""
    facts_text = json.dumps(
        factual_context,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    prompt = f"""
You are the Analytics Agent business analyst.

Generate the final BUSINESS INSIGHTS and SUMMARY for the user's question.

USER QUESTION:
{question}

AUTHORITATIVE FACTUAL ANALYSIS:
{facts_text}

Strict grounding rules:
1. The supplied factual analysis is authoritative.
2. Use ONLY facts and numeric values contained in it.
3. Do NOT invent, estimate, round into a different value, or infer
   unsupported numbers.
4. Do not use outside knowledge to add facts or causes.
5. If the supplied facts do not support a conclusion, say that the
   available data is insufficient.
6. Identify meaningful patterns, comparisons, highest/lowest results,
   period performance, category performance, and trend direction when
   those facts are available.
7. Keep the language professional, concise, and business-focused.
8. Directly answer the user's question.
9. Forecast values, when present, must be described as projected/estimated,
   never as historical actuals.
10. Do not mention SQL, Athena, Python, Ollama, LangGraph, or model names,
    prompts, models, or internal implementation.

Return exactly this structure:

INSIGHTS:
- <business insight>
- <business insight>
- <business insight>

SUMMARY:
<one concise paragraph>

Do not add any other sections.
"""

    answer = generate_with_nemotron(
        prompt,
        temperature=0.15,
        max_tokens=500,
        timeout=120,
    )
    return _parse_model_insights_summary(answer)


def get_response(question, df, forecast_df=None):
    """Return AI-generated insights/summary grounded by deterministic facts.

    The deterministic analytics remains the factual grounding layer.
    Nemotron supplies only the final natural-language interpretation.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return (
            "No data was returned for this question.",
            "The query returned no data.",
        )

    factual_context = _build_factual_context(
        question,
        df,
        forecast_df=forecast_df,
    )

    ai_insights, ai_summary = _generate_ai_insights_summary(
        question,
        factual_context,
    )

    # Safe fallback: preserve the working deterministic analytics if Qwen
    # is unavailable, times out, or returns an unusable response.
    fallback_insights, fallback_summary = _deep_dataframe_analysis(
        question,
        df,
    )

    return (
        ai_insights or fallback_insights,
        ai_summary or fallback_summary,
    )


def _find_month_column(df):
    if df is None or df.empty:
        return None
    names = {str(c).lower().strip(): c for c in df.columns}
    for candidate in [
    "month",
    "month_name",
    "month_year",
    "date",
     "booking_month"]:
        if candidate in names:
            return names[candidate]
    for c in df.columns:
        name = str(c).lower()
        if "month" in name or "date" in name:
            return c
    return None

def _find_year_column(df):
    if df is None or df.empty:
        return None
    names = {str(c).lower().strip(): c for c in df.columns}
    return names.get("year")

def _find_metric_column(df):
    if df is None or df.empty:
        return None
    preferred = [
        "total_sales_amount", "sales", "total_sales", "sales_amount",
        "sales_volume", "revenue", "total_revenue", "average_sales_volume",
        "current_sales", "previous_sales", "number_of_bookings",
        "bookings", "booking_count", "number_of_pax", "pax"
    ]
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for candidate in preferred:
        if candidate in lower_map:
            return lower_map[candidate]
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        return numeric_cols[-1]
    return None

def prepare_monthly_series(df):
    """Return a clean monthly dataframe with period_label and metric."""
    if df is None or df.empty:
        return None, None, None

    month_col = _find_month_column(df)
    metric_col = _find_metric_column(df)
    year_col = _find_year_column(df)

    if month_col is None or metric_col is None:
        return None, month_col, metric_col

    work = df.copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[metric_col]).copy()
    if work.empty:
        return None, month_col, metric_col

    month_values = work[month_col]

    # Date/datetime month values
    parsed_dates = pd.to_datetime(month_values, errors="coerce")
    if parsed_dates.notna().sum() >= max(1, int(len(work) * 0.7)
                          ) and not pd.api.types.is_numeric_dtype(month_values):
        work["_period"] = parsed_dates.dt.to_period("M")
        grouped = work.groupby("_period", as_index=False)[metric_col].sum()
        grouped = grouped.sort_values("_period")
        grouped["period_label"] = grouped["_period"].astype(str)
        return grouped[["_period", "period_label",
            metric_col]], month_col, metric_col

    # Numeric month + optional year
    numeric_month = pd.to_numeric(month_values, errors="coerce")
    if numeric_month.notna().all() and numeric_month.between(1, 12).all():
        work["_month_num"] = numeric_month.astype(int)
        if year_col is not None:
            work["_year_num"] = pd.to_numeric(work[year_col], errors="coerce")
        else:
            work["_year_num"] = 2000

        if work["_year_num"].notna().all():
            work["_period"] = pd.to_datetime(
                dict(
                    year=work["_year_num"].astype(int),
                    month=work["_month_num"],
                    day=1,
                ),
                errors="coerce",
            ).dt.to_period("M")
            grouped = work.groupby("_period", as_index=False)[metric_col].sum()
            grouped = grouped.sort_values("_period")
            grouped["period_label"] = grouped["_period"].astype(str)
            return grouped[["_period", "period_label",
                metric_col]], month_col, metric_col

    # Month names such as January/Jan
    month_name_map = {
        "jan": 1, "january": 1, "feb": 2, "february": 2,
        "mar": 3, "march": 3, "apr": 4, "april": 4,
        "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    month_num = work[month_col].astype(
        str).str.lower().str.strip().map(month_name_map)
    if month_num.notna().all():
        work["_month_num"] = month_num.astype(int)
        if year_col is not None:
            work["_year_num"] = pd.to_numeric(work[year_col], errors="coerce")
        else:
            work["_year_num"] = 2000
        work["_period"] = pd.to_datetime(
            dict(
                year=work["_year_num"].astype(int),
                month=work["_month_num"],
                day=1,
            ),
            errors="coerce",
        ).dt.to_period("M")
        grouped = work.groupby("_period", as_index=False)[
                               metric_col].sum().sort_values("_period")
        grouped["period_label"] = grouped["_period"].astype(str)
        return grouped[["_period", "period_label",
            metric_col]], month_col, metric_col

    return None, month_col, metric_col

def _numeric_columns(df):
    """Return numeric-looking columns, including Athena values returned as strings."""
    if df is None or df.empty:
        return []

    result = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            result.append(col)
    return result

def _category_columns(df):
    """Return columns that are useful as chart categories."""
    if df is None or df.empty:
        return []

    result = []
    for col in df.columns:
        name = str(col).lower().strip()

        # Never use technical/helper columns as a chart category.
        if name in {"_period", "forecast"}:
            continue

        converted = pd.to_numeric(df[col], errors="coerce")

        # Keep genuine text/category columns.
        if not pd.api.types.is_numeric_dtype(
    df[col]) and converted.notna().sum() < len(df):
            result.append(col)

    return result

