from langgraph.graph import END, START, StateGraph

from .state import AnalyticsState
from .nodes import (
    understand_query,
    conversation,
    business,
    visualization,
    generate_sql_node,
    validate_sql_node,
    repair_sql_node,
    execute_sql_node,
    process_results_node,
    failed,
)
from .routers import (
    route_after_intent,
    route_after_validation,
    route_after_execution,
)


def build_analytics_graph():
    """Build the AI-Powered Analytics Agent LangGraph."""

    graph = StateGraph(AnalyticsState)

    graph.add_node("understand_query", understand_query)
    graph.add_node("conversation", conversation)
    graph.add_node("business", business)
    graph.add_node("visualization", visualization)

    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("repair_sql", repair_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("process_results", process_results_node)
    graph.add_node("failed", failed)

    graph.add_edge(START, "understand_query")

    graph.add_conditional_edges(
        "understand_query",
        route_after_intent,
        {
            "generate_sql": "generate_sql",
            "conversation": "conversation",
            "business": "business",
            "visualization": "visualization",
        },
    )

    graph.add_edge("generate_sql", "validate_sql")

    graph.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {
            "execute_sql": "execute_sql",
            "repair_sql": "repair_sql",
            "failed": "failed",
        },
    )

    # A repaired query always comes back through generic validation.
    graph.add_edge("repair_sql", "validate_sql")

    graph.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {
            "process_results": "process_results",
            "repair_sql": "repair_sql",
            "failed": "failed",
        },
    )

    graph.add_edge("process_results", END)
    graph.add_edge("conversation", END)
    graph.add_edge("business", END)
    graph.add_edge("visualization", END)
    graph.add_edge("failed", END)

    return graph.compile()


analytics_graph = build_analytics_graph()
