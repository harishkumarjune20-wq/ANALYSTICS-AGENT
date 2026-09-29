from typing import Any, TypedDict

class AnalyticsState(TypedDict, total=False):
    """State carried through the Analytics Agent LangGraph."""
    user_query: str
    previous_question: str
    previous_sql: str
    previous_df: Any
    intent: str
    confidence: float
    schema: str
    generated_sql: str
    validated_sql: str
    sql_error: str
    repair_attempts: int
    result_df: Any
    result_error: str
    error_type: str
    insights: str
    summary: str
    recommendations: str
    response: str
    validation_status: str
    status: str
