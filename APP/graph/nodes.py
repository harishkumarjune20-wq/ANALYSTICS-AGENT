import re
import time
from typing import Any
import pandas as pd

try:
    from agents.intent_classifier import classify_intent
    from agents.conversation_agent import generate_conversation_response
    from agents.response_generator import generate_recommendation
    from agents.analytics_insights import get_response, _build_factual_context
    from agents.sql_generator_athena import generate_sql, clean_sql, fix_sql, _validate_basic_sql_shape
    from athena.athena_executor import execute_query
    from utils.schema_loader_athena import get_schema
except ImportError:
    from agents.intent_classifier import classify_intent
    from agents.conversation_agent import generate_conversation_response
    from agents.response_generator import generate_recommendation
    from agents.analytics_insights import get_response, _build_factual_context
    from agents.sql_generator_athena import generate_sql, clean_sql, fix_sql, _validate_basic_sql_shape
    from athena.athena_executor import execute_query
    from utils.schema_loader_athena import get_schema

from .state import AnalyticsState

def _terminal_stage_start(label: str) -> float:
    """Print a visible terminal stage marker and return a monotonic start time."""
    start = time.perf_counter()
    print(f"{label}", flush=True)
    return start


def _terminal_stage_done(label: str, start: float) -> float:
    """Print the elapsed time for a terminal stage."""
    elapsed = time.perf_counter() - start
    print(f"   ⏱ {label}: {elapsed:.2f} sec", flush=True)
    return elapsed


def _normalize_dataframe(df: Any) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if df is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(df)
    except Exception:
        return pd.DataFrame()

def _generic_result_processing(df: Any) -> pd.DataFrame:
    result = _normalize_dataframe(df).copy()
    if result.empty:
        return result
    for col in result.columns:
        if result[col].dtype == object:
            try:
                result[col] = pd.to_numeric(result[col], errors="ignore")
            except Exception:
                pass
    return result

def _prepare_sql_question(state: AnalyticsState) -> str:
    question = (state.get("user_query") or "").strip()
    previous_question = (state.get("previous_question") or "").strip()
    previous_sql = (state.get("previous_sql") or "").strip()
    if not previous_question and not previous_sql:
        return question

    markers = ("above", "previous", "earlier", "same", "those", "these",
               "that", "it", "them", "also", "now", "instead", "change",
               "compare", "comparison", "breakdown", "drill")
    if not any(re.search(r"\b" + re.escape(m) + r"\b", question.lower()) for m in markers):
        return question

    return f"""Previous user question:
{previous_question or "(none)"}

Previous generated SQL:
{previous_sql or "(none)"}

Current user request:
{question}

Use the previous turn only as context. Generate a new generic,
schema-driven Athena SQL statement that answers the current request."""

def understand_query(state: AnalyticsState) -> AnalyticsState:
    # Fast path: app.py may already have classified the request.
    # Reusing the supplied intent avoids a second LLM/classifier call.
    preset_intent = state.get("intent")
    if preset_intent:
        try:
            confidence = float(state.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        return {
            **state,
            "intent": preset_intent,
            "confidence": confidence,
            "status": "intent_classified",
        }

    result = classify_intent(
        state.get("user_query", ""),
        previous_question=state.get("previous_question", ""),
        previous_df=state.get("previous_df"),
    )
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {**state, "intent": result.get("intent", "conversation"),
            "confidence": confidence, "status": "intent_classified"}

def conversation(state: AnalyticsState) -> AnalyticsState:
    answer = generate_conversation_response(state.get("user_query", ""))
    return {**state, "response": str(answer or ""), "status": "completed"}

def business(state: AnalyticsState) -> AnalyticsState:
    answer = generate_recommendation(
        state.get("user_query", ""), state.get("previous_df"),
        previous_question=state.get("previous_question", "")
    )
    return {**state, "recommendations": str(answer or ""),
            "response": str(answer or ""), "status": "completed"}

def visualization(state: AnalyticsState) -> AnalyticsState:
    previous_df = state.get("previous_df")
    if previous_df is None or _normalize_dataframe(previous_df).empty:
        message = ("I can create charts and forecasts from your analytics result, "
                   "but there is no previous analytics data available. "
                   "Please ask an analytics question first.")
    else:
        message = "Previous analytics data is available for visualization."
    return {**state, "result_df": previous_df, "response": message, "status": "completed"}

def generate_sql_node(state: AnalyticsState) -> AnalyticsState:
    start = _terminal_stage_start("🧠 Generating SQL...")
    try:
        schema = state.get("schema") or get_schema()
        sql = generate_sql(_prepare_sql_question(state), schema)
        sql = fix_sql(clean_sql(sql))
        return {**state, "schema": schema, "generated_sql": sql, "validated_sql": "",
                "sql_error": "", "result_error": "", "validation_status": "pending",
                "status": "sql_generated"}
    finally:
        _terminal_stage_done("Generating SQL", start)

def validate_sql_node(state: AnalyticsState) -> AnalyticsState:
    start = _terminal_stage_start("🔍 Validating SQL...")
    sql = state.get("generated_sql", "")
    try:
        _validate_basic_sql_shape(sql)
        validated = fix_sql(clean_sql(sql))
        return {**state, "validated_sql": validated, "generated_sql": validated,
                "validation_status": "valid", "sql_error": "", "status": "sql_validated"}
    except Exception as exc:
        return {**state, "validation_status": "invalid", "sql_error": str(exc),
                "status": "sql_validation_failed"}
    finally:
        _terminal_stage_done("Validating SQL", start)

def _build_repair_question(state: AnalyticsState) -> str:
    failure = state.get("result_error") or state.get("sql_error") or "Unknown SQL failure."
    return f"""Repair the following Athena SQL so it answers the user's original question.

Original user question:
{state.get("user_query", "")}

Previous generated SQL:
{state.get("generated_sql", "")}

Failure returned by validation or Athena:
{failure}

Schema:
{state.get("schema", "")}

Return ONLY one complete executable Athena SQL statement.
Do not explain the correction.
Do not invent tables or columns.
Preserve the original analytical intent.
Use the supplied schema.
Keep the SQL generic and schema-driven."""

def repair_sql_node(state: AnalyticsState) -> AnalyticsState:
    # AWS credential/session failures cannot be fixed by changing the SQL.
    # Keep the existing repair node in the graph, but skip the expensive
    # Nemotron repair call and report the credential problem clearly.
    if state.get("error_type") == "SESSION_EXPIRED":
        message = state.get(
            "result_error",
            "🔐 AWS session token has expired. Please refresh your AWS credentials and try again."
        )
        print("\n❌ AWS SESSION TOKEN EXPIRED")
        print(message)
        print("   ↳ SQL repair skipped because the problem is AWS credentials, not SQL.")
        return {**state, "status": "session_expired"}

    attempts = int(state.get("repair_attempts", 0)) + 1
    start = _terminal_stage_start(f"🛠 Repairing SQL (attempt {attempts})...")
    try:
        schema = state.get("schema") or get_schema()
        repaired = generate_sql(_build_repair_question(state), schema)
        repaired = fix_sql(clean_sql(repaired))
        return {**state, "schema": schema, "generated_sql": repaired,
                "validated_sql": "", "repair_attempts": attempts,
                "sql_error": "", "result_error": "", "validation_status": "pending",
                "status": "sql_repaired"}
    finally:
        _terminal_stage_done(f"Repairing SQL (attempt {attempts})", start)

def execute_sql_node(state: AnalyticsState) -> AnalyticsState:
    start = _terminal_stage_start("☁️ Executing SQL in Athena...")
    sql = state.get("validated_sql") or state.get("generated_sql", "")
    try:
        result = execute_query(sql)
        if isinstance(result, pd.DataFrame):
            return {**state, "result_df": result, "result_error": "",
                    "error_type": "", "status": "athena_completed"}
        if isinstance(result, dict) and result.get("success") is False:
            error_type = str(result.get("error_type", "ATHENA_ERROR"))
            message = str(result.get("message", "Athena query failed."))
            return {
                **state,
                "result_error": message,
                "error_type": error_type,
                "status": "athena_failed",
            }
        return {**state,
                "result_error": "Athena returned an unexpected result format.",
                "error_type": "ATHENA_ERROR",
                "status": "athena_failed"}
    except Exception as exc:
        return {**state, "result_error": str(exc),
                "error_type": "ATHENA_ERROR", "status": "athena_failed"}
    finally:
        _terminal_stage_done("Executing SQL in Athena", start)

def process_results_node(state: AnalyticsState) -> AnalyticsState:
    start = _terminal_stage_start("💡 Creating insights and summary...")
    df = _generic_result_processing(state.get("result_df"))
    try:
        if df.empty:
            return {**state, "result_df": df,
                    "insights": "No data was returned for this question.",
                    "summary": "The query returned no data.",
                    "response": "Query executed successfully, but no data was returned.",
                    "status": "results_processed"}
        # Python creates the authoritative factual grounding context first.
        # Nemotron then turns those facts into the final business language.
        factual_context = _build_factual_context(
            state.get("user_query", ""),
            df,
        )
        insights, summary = get_response(
            state.get("user_query", ""),
            df,
        )

        return {**state, "result_df": df,
                "factual_analysis": factual_context,
                "insights": str(insights or ""),
                "summary": str(summary or ""),
                "response": "Query executed successfully.",
                "status": "results_processed"}
    finally:
        _terminal_stage_done("Creating insights and summary", start)

def failed(state: AnalyticsState) -> AnalyticsState:
    error_type = state.get("error_type", "")

    if error_type == "SESSION_EXPIRED":
        message = (
            "❌ AWS SESSION TOKEN EXPIRED\n"
            "   AWS credentials/session token have expired.\n"
            "   Please refresh your AWS credentials."
        )
        print("\n❌ AWS SESSION TOKEN EXPIRED", flush=True)
        print("   AWS credentials/session token have expired.", flush=True)
        print("   Please refresh your AWS credentials.", flush=True)
        print("\n❌ Analytics workflow stopped.", flush=True)
        return {**state, "response": message, "status": "failed"}

    message = state.get("result_error") or state.get("sql_error") or (
        "The analytics request could not be completed safely."
    )
    return {**state, "response": str(message), "status": "failed"}
