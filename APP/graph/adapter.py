"""
Streamlit-facing adapter for the Analytics Agent LangGraph.

Keep Streamlit/UI rendering in app.py. This module only prepares the
initial LangGraph state and invokes the already-compiled analytics graph.
"""

from graph.analytics_graph import analytics_graph


def run_analytics_agent(
    question,
    previous_question="",
    previous_sql="",
    previous_df=None,
    schema=None,
):
    """
    Run the Analytics Agent LangGraph.

    The previous question, previous SQL, and previous DataFrame are passed
    into the graph so follow-up questions can reuse the existing context.
    """

    initial_state = {
        "user_query": question or "",
        "previous_question": previous_question or "",
        "previous_sql": previous_sql or "",
        "previous_df": previous_df,
        "repair_attempts": 0,
    }

    # Pass the schema when app.py has already loaded it.
    # If omitted, the graph can use its normal schema-loading behavior.
    if schema:
        initial_state["schema"] = schema

    return analytics_graph.invoke(initial_state)
