from .state import AnalyticsState

def route_after_intent(state: AnalyticsState) -> str:
    intent = state.get("intent", "conversation")
    if intent == "aws_data":
        return "generate_sql"
    if intent == "visualization":
        return "visualization"
    if intent == "business":
        return "business"
    return "conversation"

def route_after_validation(state: AnalyticsState) -> str:
    if state.get("error_type") == "SESSION_EXPIRED":
        return "failed"
    if state.get("validation_status") == "valid":
        return "execute_sql"
    if state.get("repair_attempts", 0) < 2:
        return "repair_sql"
    return "failed"

def route_after_execution(state: AnalyticsState) -> str:
    # Credential/session failures are not SQL failures. Never send them to
    # the SQL-repair LLM because changing SQL cannot refresh AWS credentials.
    if state.get("error_type") == "SESSION_EXPIRED":
        return "failed"
    if state.get("result_error"):
        if state.get("repair_attempts", 0) < 2:
            return "repair_sql"
        return "failed"
    return "process_results"
