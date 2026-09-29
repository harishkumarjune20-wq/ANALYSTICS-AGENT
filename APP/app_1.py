from utils.ollama_client import generate_with_qwen
from utils.nemotron_client import generate_with_nemotron
from agents.conversation_agent import generate_conversation_response
from agents.intent_classifier import classify_intent
from utils.schema_loader_athena import get_schema
try:
    from graph.analytics_graph import analytics_graph
except ImportError:
    from graph.analytics_graph import analytics_graph
from agents.error_recovery import fix_sql
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import re
import base64
import textwrap

def get_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

sidebar_bg = get_base64("assets/sidebar_bg.jpg")

@st.cache_data(show_spinner=False)
def get_schema_cached():
    return get_schema()


# =====================================================
# SESSION STATE
# =====================================================
if "processing" not in st.session_state:
    st.session_state.processing = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_analytics_df" not in st.session_state:
    st.session_state.last_analytics_df = None

if "last_analytics_sql" not in st.session_state:
    st.session_state.last_analytics_sql = ""

if "last_analytics_question" not in st.session_state:
    st.session_state.last_analytics_question = ""

if "last_visualization_df" not in st.session_state:
    st.session_state.last_visualization_df = None

if "last_forecast_df" not in st.session_state:
    st.session_state.last_forecast_df = None

if "business_context_df" not in st.session_state:
    st.session_state.business_context_df = None

if "business_context_question" not in st.session_state:
    st.session_state.business_context_question = ""

if "last_intent" not in st.session_state:
    st.session_state.last_intent = ""


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI-POWERED ANALYTICS AGENT",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(textwrap.dedent("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
""").strip(), unsafe_allow_html=True)

# =====================================================
# CUSTOM CSS - ANALYTICS AGENT CHAT UI
# =====================================================

sidebar_css = """
<style>
/* ---------- APP ---------- */
.stApp {
    background:
        radial-gradient(circle at 75% 5%, rgba(221,214,254,.45), transparent 30%),
        radial-gradient(circle at 20% 20%, rgba(191,219,254,.35), transparent 35%),
        linear-gradient(135deg, #f8fbff 0%, #f5f8ff 50%, #faf8ff 100%);
}

.block-container {
    max-width: 1400px !important;
    padding-top: 1rem !important;
    padding-bottom: 8rem !important;
}

/* ---------- HIDE STREAMLIT CHROME ---------- */
#MainMenu,
footer{
    visibility:hidden !important;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    width: 280px !important;
    position: relative !important;
    background: transparent !important;
    border-right: 1px solid #e5edf8 !important;
    box-shadow: 4px 0 20px rgba(30,64,175,.04);
    overflow: hidden !important;
}

/* Keep every Streamlit sidebar wrapper transparent so the watermark
   remains visible behind the ENTIRE sidebar, not just one component. */
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    background: transparent !important;
}

section[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: url("data:image/jpeg;base64,{sidebar_bg}");
    background-repeat: no-repeat;
    background-position: center center;
    background-size: cover;
    opacity: .20;
    filter: grayscale(20%);
    z-index: 0;
    pointer-events: none;
}

section[data-testid="stSidebar"] > div {
    position: relative;
    z-index: 1;
}

section[data-testid="stSidebar"] * {
    color: #334155;
}

section[data-testid="stSidebar"] hr {
    border-color: #e2e8f0 !important;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    min-height: 42px !important;
    background: #fff !important;
    color: #334155 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 13px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 12px rgba(15,23,42,.05) !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #f8fbff !important;
    border-color: #bfdbfe !important;
    color: #1d4ed8 !important;
}
button[kind="header"] {
    display:block !important;
    visibility:visible !important;
    opacity:1 !important;
}
[data-testid="collapsedControl"]{
    display:block !important;
    visibility:visible !important;
    opacity:1 !important;
}

/* ---------- HEADER ---------- */
.title {
    font-size: 42px;
    font-weight: 700;
    color: #0f172a;
}

.subtitle {
    color: #64748b;
    font-size: 18px;
}

/* ---------- WELCOME ---------- */
.welcome-card {
    background: linear-gradient(135deg, rgba(255,255,255,.96), rgba(250,252,255,.94));
    border: 1px solid rgba(226,232,240,.7);
    border-radius: 28px;
    padding: 46px 50px;
    text-align: center;
    box-shadow: 0 18px 50px rgba(30,64,175,.08);
    margin: 30px auto 25px auto;
    max-width: 960px;
}

.welcome-icon {
    width: 72px;
    height: 72px;
    margin: 0 auto 23px auto;
    border-radius: 23px;
    background: linear-gradient(135deg, #f0f7ff, #f7f3ff);
    border: 1px solid #e6edf8;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 32px;
}

.welcome-title {
    font-size: 31px;
    font-weight: 750;
    color: #0f172a;
    margin: 0 0 12px 0;
}

.welcome-description {
    max-width: 720px;
    margin: auto;
    font-size: 14px;
    line-height: 1.7;
    color: #64748b;
}

/* ---------- CARDS / TABLES ---------- */
.analytics-card {
    background: rgba(255,255,255,.92);
    border: 1px solid #e6edf6;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 6px 22px rgba(15,23,42,.055);
    margin-top: 15px;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,.92) !important;
    border: 1px solid #e5edf7 !important;
    border-radius: 16px !important;
    padding: 18px !important;
    box-shadow: 0 5px 18px rgba(15,23,42,.05) !important;
}

[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
}

[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    border: 1px solid #e5edf7 !important;
    box-shadow: 0 4px 15px rgba(15,23,42,.04);
}

//* ---------- CHAT INPUT ---------- */

[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 20px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;

    width: min(900px, calc(100vw - 340px)) !important;

    max-width: 950px !important;
    z-index: 9999 !important;

    background: transparent !important;
    padding: 0 !important;
}

[data-testid="stChatInput"] > div {
    background: #ffffff !important;
    border: 1px solid #dbe5f2 !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 25px rgba(0,0,0,.08) !important;
    padding: 8px !important;
}

[data-testid="stChatInput"] textarea {
    min-height: 44px !important;
    max-height: 140px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #334155 !important;
    padding: 10px 14px !important;
}

[data-testid="stChatInput"] button {
    width: 40px !important;
    height: 40px !important;
    border-radius: 12px !important;
    border: none !important;

    background: linear-gradient(
        135deg,
        #2563eb,
        #4f46e5
    ) !important;

    color: white !important;
}
.stDownloadButton > button {
    border-radius: 11px !important;
    height: 40px !important;
    font-weight: 600 !important;
    border: 1px solid #dbe5f2 !important;
    background: white !important;
    color: #334155 !important;
}

@media (max-width: 900px) {
    [data-testid="stChatInput"] {
        width: calc(100vw - 32px) !important;
        bottom: 12px !important;
    }
}

::-webkit-scrollbar { width: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
</style>
"""

# Inject the actual image data into CSS safely without using an f-string.
sidebar_css = sidebar_css.replace("{sidebar_bg}", sidebar_bg)
st.markdown(sidebar_css, unsafe_allow_html=True)
#header

col1, col2 = st.columns([1, 8])

with col1:
    st.image("assets/", width=90)

with col2:
    st.markdown(
        textwrap.dedent("""
        <div class="title">
            AI-POWERED ANALYTICS AGENT
        </div>
        <div class="subtitle">
            AI Powered Analytics Dashboard
        </div>
        """).strip(),
        unsafe_allow_html=True
    )
#welcome card
#welcome card
st.markdown(
    textwrap.dedent("""
    <div class="welcome-card">
        <div class="welcome-icon">✈️</div>
        <div class="welcome-title">Hi! Ready to turn your data into insights?</div>
        <div class="welcome-description">
            Ask me about your data environment performance, sales, bookings,
            revenue, suppliers, routes and business trends.
            I can analyze your data and help identify opportunities.
        </div>
    </div>
    """).strip(),
    unsafe_allow_html=True
)
# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.image(
        "assets/",
        width="stretch"
    )

    st.markdown("## your data environment")
    st.caption("Analytics Assistant")

    st.divider()

    if st.button("➕ New Chat", width="stretch"):
        st.session_state.messages = []
        st.session_state.processing = False
        st.session_state.pending_question = None
        st.rerun()

    st.subheader("Workspace")

    st.info(
        "Ask questions about bookings, "
        "sales, suppliers, routes, revenue and trends."
    )

    st.subheader("Actions")

    if st.button("🗑 Clear Chat", width="stretch"):
        st.session_state.messages = []
        st.session_state.processing = False
        st.session_state.pending_question = None
        st.rerun()
   
# =====================================================
# QUESTION NORMALIZATION
# =====================================================


def normalize_question(question):

    if not question:
        return ""

    month_map = {
        "january": "month 1",
        "february": "month 2",
        "march": "month 3",
        "april": "month 4",
        "may": "month 5",
        "june": "month 6",
        "july": "month 7",
        "august": "month 8",
        "september": "month 9",
        "october": "month 10",
        "november": "month 11",
        "december": "month 12"
    }

    normalized = question.lower()

    for month_name, month_number in month_map.items():

        normalized = re.sub(
            rf"\b{month_name}\b",
            month_number,
            normalized
        )

    return normalized


# =====================================================
# SQL POST-PROCESSING
# =====================================================

def fix_numeric_lower(sql):

    if not isinstance(sql, str) or not sql.strip():

        raise ValueError(
            "Generated SQL is empty or is not a string."
        )

    numeric_columns = {
        "year",
        "month",
        "number_of_bookings",
        "number_of_bookings_with_markup",
        "number_of_pax",
        "number_of_segments",
        "number_of_tickets",
        "total_sales_amount",
        "total_markup_converted"
    }

    # -------------------------------------------------
    # CURRENT_DATE()
    # -------------------------------------------------

    sql = re.sub(
        r"\bCURRENT_DATE\s*\(\s*\)",
        "current_date",
        sql,
        flags=re.IGNORECASE
    )

    # -------------------------------------------------
    # INTERVAL -> Athena date_add
    # -------------------------------------------------

    sql = re.sub(
        r"DATE_TRUNC\s*\(\s*'month'\s*,\s*CURRENT_DATE\s*\)"
        r"\s*-\s*INTERVAL\s*'1\s+month'",
        "date_add('month', -1, current_date)",
        sql,
        flags=re.IGNORECASE
    )

    sql = re.sub(
        r"DATE_TRUNC\s*\(\s*'year'\s*,\s*CURRENT_DATE\s*\)"
        r"\s*-\s*INTERVAL\s*'1\s+year'",
        "date_add('year', -1, current_date)",
        sql,
        flags=re.IGNORECASE
    )

    # -------------------------------------------------
    # Remove LOWER() around numeric columns
    # -------------------------------------------------

    for col in numeric_columns:

        sql = re.sub(
            rf"LOWER\s*\(\s*{re.escape(col)}\s*\)",
            col,
            sql,
            flags=re.IGNORECASE
        )

        sql = re.sub(
            rf"UPPER\s*\(\s*{re.escape(col)}\s*\)",
            col,
            sql,
            flags=re.IGNORECASE
        )

    # -------------------------------------------------
    # Remove LOWER() around numeric values
    # -------------------------------------------------

    sql = re.sub(
        r"LOWER\s*\(\s*'(\d+(?:\.\d+)?)'\s*\)",
        r"\1",
        sql,
        flags=re.IGNORECASE
    )

    sql = re.sub(
        r"UPPER\s*\(\s*'(\d+(?:\.\d+)?)'\s*\)",
        r"\1",
        sql,
        flags=re.IGNORECASE
    )

    # -------------------------------------------------
    # Fix year(month_year)
    # -------------------------------------------------

    sql = re.sub(
        r"\byear\s*\(\s*month_year\s*\)",
        "year",
        sql,
        flags=re.IGNORECASE
    )

    # -------------------------------------------------
    # Fix month(month_year)
    # -------------------------------------------------

    sql = re.sub(
        r"\bmonth\s*\(\s*month_year\s*\)",
        "month",
        sql,
        flags=re.IGNORECASE
    )

    # -------------------------------------------------
    # Fix ILIKE
    # -------------------------------------------------

    sql = re.sub(
        r"(\w+)\s+ILIKE\s+'(.*?)'",
        r"LOWER(\1) LIKE LOWER('\2')",
        sql,
        flags=re.IGNORECASE
    )

    return sql
# =====================================================
# STRING / CAPITALIZATION SQL NORMALIZATION
# =====================================================


# =====================================================
# SQL TYPE DEFINITIONS
# =====================================================

NUMERIC_COLUMNS = {
    "year",
    "month",
    "number_of_bookings",
    "number_of_bookings_with_markup",
    "number_of_pax",
    "number_of_segments",
    "number_of_tickets",
    "total_sales_amount",
    "total_markup_converted",
}

DATE_COLUMNS = {
    "report_date",
}


# =====================================================
# DATE COMPARISON NORMALIZATION
# =====================================================

def fix_date_comparisons(sql):
    """
    Protect DATE columns from LOWER()/UPPER() processing.

    report_date is an Athena DATE column.

    Valid:
        report_date = DATE '2026-06-20'
        t1.report_date = DATE '2026-06-20'

    Invalid:
        LOWER(report_date) = LOWER('2026-06-20')
        LOWER(t1.report_date) = LOWER('2026-06-20')
        report_date = '2026-06-20'
    """

    if not isinstance(sql, str) or not sql.strip():
        return sql

    column_ref = (
        r"(?:[A-Za-z_][A-Za-z0-9_]*\.)?"
        r"[A-Za-z_][A-Za-z0-9_]*"
    )

    # -------------------------------------------------
    # Fix corrupted LOWER(report_date) comparisons
    # -------------------------------------------------

    sql = re.sub(
        rf"""
        LOWER\s*\(\s*
        (?P<column>{column_ref}\.?(?:report_date))
        \s*\)
        \s*=\s*
        LOWER\s*\(\s*
        ['"](?P<date>\d{{4}}-\d{{2}}-\d{{2}})['"]
        \s*\)
        """,
        lambda match: (
            f"{match.group('column')} = "
            f"DATE '{match.group('date')}'"
        ),
        sql,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    # -------------------------------------------------
    # Fix corrupted UPPER(report_date) comparisons
    # -------------------------------------------------

    sql = re.sub(
        rf"""
        UPPER\s*\(\s*
        (?P<column>{column_ref}\.?(?:report_date))
        \s*\)
        \s*=\s*
        UPPER\s*\(\s*
        ['"](?P<date>\d{{4}}-\d{{2}}-\d{{2}})['"]
        \s*\)
        """,
        lambda match: (
            f"{match.group('column')} = "
            f"DATE '{match.group('date')}'"
        ),
        sql,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    # -------------------------------------------------
    # Fix plain report_date = 'YYYY-MM-DD'
    # -------------------------------------------------

    sql = re.sub(
        rf"""
        (?P<column>
            (?:[A-Za-z_][A-Za-z0-9_]*\.)?
            report_date
        )
        \s*=\s*
        ['"](?P<date>\d{{4}}-\d{{2}}-\d{{2}})['"]
        """,
        lambda match: (
            f"{match.group('column')} = "
            f"DATE '{match.group('date')}'"
        ),
        sql,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    # -------------------------------------------------
    # Remove LOWER()/UPPER() around DATE columns
    # -------------------------------------------------

    for date_column in DATE_COLUMNS:
        sql = re.sub(
            rf"""
            LOWER\s*\(\s*
            ((?:[A-Za-z_][A-Za-z0-9_]*\.)?{re.escape(date_column)})
            \s*\)
            """,
            r"\1",
            sql,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        sql = re.sub(
            rf"""
            UPPER\s*\(\s*
            ((?:[A-Za-z_][A-Za-z0-9_]*\.)?{re.escape(date_column)})
            \s*\)
            """,
            r"\1",
            sql,
            flags=re.IGNORECASE | re.VERBOSE,
        )

    return sql


# =====================================================
# NUMERIC LOWER/UPPER NORMALIZATION
# =====================================================

def fix_numeric_lower(sql):
    """
    Remove LOWER()/UPPER() from numeric columns and
    numeric literals.
    """

    if not isinstance(sql, str) or not sql.strip():
        return sql

    for column in NUMERIC_COLUMNS:
        qualified_column = (
            rf"(?:[A-Za-z_][A-Za-z0-9_]*\.)?"
            rf"{re.escape(column)}"
        )

        sql = re.sub(
            rf"LOWER\s*\(\s*({qualified_column})\s*\)",
            r"\1",
            sql,
            flags=re.IGNORECASE,
        )

        sql = re.sub(
            rf"UPPER\s*\(\s*({qualified_column})\s*\)",
            r"\1",
            sql,
            flags=re.IGNORECASE,
        )

    # Remove LOWER()/UPPER() around numeric literals
    sql = re.sub(
        r"LOWER\s*\(\s*'(\d+(?:\.\d+)?)'\s*\)",
        r"\1",
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"UPPER\s*\(\s*'(\d+(?:\.\d+)?)'\s*\)",
        r"\1",
        sql,
        flags=re.IGNORECASE,
    )

    return sql


# =====================================================
# STRING / CAPITALIZATION SQL NORMALIZATION
# =====================================================

def fix_string_case_comparisons(sql):
    """
    Normalize string comparisons safely.

    Important:
        t1.status
    becomes:
        LOWER(t1.status)

    It must never become:
        t1.LOWER(status)

    DATE and numeric columns are excluded.
    """

    if not isinstance(sql, str) or not sql.strip():
        return sql

    column_ref = (
        r"(?:[A-Za-z_][A-Za-z0-9_]*\.)?"
        r"[A-Za-z_][A-Za-z0-9_]*"
    )

    protected_columns = NUMERIC_COLUMNS | DATE_COLUMNS

    # -------------------------------------------------
    # column = 'value'
    # -------------------------------------------------

    def column_equals_value(match):
        expression = match.group(1)
        value = match.group(2)

        base_column = expression.rsplit(".", 1)[-1].lower()

        if base_column in protected_columns:
            return match.group(0)

        return f"LOWER({expression}) = LOWER('{value}')"

    sql = re.sub(
        rf"(?<![\w.])({column_ref})\s*=\s*'([^']*)'",
        column_equals_value,
        sql,
        flags=re.IGNORECASE,
    )

    # -------------------------------------------------
    # 'value' = column
    # -------------------------------------------------

    def value_equals_column(match):
        value = match.group(1)
        expression = match.group(2)

        base_column = expression.rsplit(".", 1)[-1].lower()

        if base_column in protected_columns:
            return match.group(0)

        return f"LOWER('{value}') = LOWER({expression})"

    sql = re.sub(
        rf"'([^']*)'\s*=\s*({column_ref})(?![\w.])",
        value_equals_column,
        sql,
        flags=re.IGNORECASE,
    )

    # -------------------------------------------------
    # LOWER(column) = 'value'
    # -------------------------------------------------

    def normalize_lower_comparison(match):
        expression = match.group(1)
        value = match.group(2)

        base_column = expression.rsplit(".", 1)[-1].lower()

        if base_column in DATE_COLUMNS:
            return f"{expression} = DATE '{value}'"

        if base_column in NUMERIC_COLUMNS:
            return f"{expression} = {value}"

        return f"LOWER({expression}) = LOWER('{value}')"

    sql = re.sub(
        rf"LOWER\s*\(\s*({column_ref})\s*\)"
        rf"\s*=\s*'([^']*)'",
        normalize_lower_comparison,
        sql,
        flags=re.IGNORECASE,
    )

    # -------------------------------------------------
    # UPPER(column) = 'value'
    # -------------------------------------------------

    def normalize_upper_comparison(match):
        expression = match.group(1)
        value = match.group(2)

        base_column = expression.rsplit(".", 1)[-1].lower()

        if base_column in DATE_COLUMNS:
            return f"{expression} = DATE '{value}'"

        if base_column in NUMERIC_COLUMNS:
            return f"{expression} = {value}"

        return f"LOWER({expression}) = LOWER('{value}')"

    sql = re.sub(
        rf"UPPER\s*\(\s*({column_ref})\s*\)"
        rf"\s*=\s*'([^']*)'",
        normalize_upper_comparison,
        sql,
        flags=re.IGNORECASE,
    )

    # -------------------------------------------------
    # Remove LOWER()/UPPER() around numeric columns
    # -------------------------------------------------

    for column in NUMERIC_COLUMNS:
        qualified_column = (
            rf"(?:[A-Za-z_][A-Za-z0-9_]*\.)?"
            rf"{re.escape(column)}"
        )

        sql = re.sub(
            rf"LOWER\s*\(\s*({qualified_column})\s*\)",
            r"\1",
            sql,
            flags=re.IGNORECASE,
        )

        sql = re.sub(
            rf"UPPER\s*\(\s*({qualified_column})\s*\)",
            r"\1",
            sql,
            flags=re.IGNORECASE,
        )

    return sql


# =====================================================
# FINAL SQL NORMALIZATION
# =====================================================

def normalize_generated_sql(sql):
    """
    Final SQL cleanup after Nemotron generation.

    This function is also used for follow-up questions.
    """

    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("Generated SQL is empty or invalid.")

    # Date protection must run first.
    sql = fix_date_comparisons(sql)

    # Numeric protection.
    sql = fix_numeric_lower(sql)

    # String normalization must run last.
    sql = fix_string_case_comparisons(sql)

    # Final date protection in case another normalizer
    # modified the expression.
    sql = fix_date_comparisons(sql)

    return sql.strip()





# =====================================================
# FORMAT NUMERIC DATA
# =====================================================

def format_numeric_dataframe(df):
    """Return a display-only copy with numeric values rendered as decimals.

    Athena can return DECIMAL/numeric results through pandas as object/string
    columns.  Looking only at pandas numeric dtypes therefore misses values
    such as ``1.519984734...E9``.  Detect numeric-looking columns as well and
    format them explicitly so Streamlit never displays scientific notation.

    The original dataframe is never modified.
    """
    if df is None:
        return df

    formatted_df = df.copy()

    for col in formatted_df.columns:
        series = formatted_df[col]

        # First try direct numeric conversion.  This handles normal int/float
        # columns and Athena Decimal/object columns.
        converted = pd.to_numeric(series, errors="coerce")
        non_null = series.notna()

        # Only treat a column as numeric when every non-null value is numeric.
        # This prevents names/IDs containing a few digits from being changed.
        if non_null.any() and converted[non_null].notna().all():

            def _format_decimal_value(value):
                if pd.isna(value):
                    return ""

                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return str(value)

                if number.is_integer():
                    return f"{number:,.0f}"

                # Fixed-point formatting is intentional: never use general
                # notation (which can produce 1.52E+09).
                return f"{number:,.2f}"

            formatted_df[col] = converted.map(_format_decimal_value)

    return formatted_df


# =====================================================
# GENERATE INSIGHTS + SUMMARY
# =====================================================

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


def _dataframe_for_llm(df, max_rows=80):
    """Convert an Athena result dataframe into compact LLM-readable text."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "No rows were returned."

    safe_df = df.copy().head(max_rows)
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].map(
            lambda value: "" if pd.isna(value) else str(value)
        )
    return safe_df.to_string(index=False)


def generate_llm_insights_summary(question, df, forecast_df=None):
    """Generate concise business insights and summary with one LLM call."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return ("• No data was returned for analysis.", "No data was available for this question.")

    data_text = _dataframe_for_llm(df, max_rows=200)
    forecast_text = ""
    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        forecast_text = _dataframe_for_llm(forecast_df, max_rows=50)

    forecast_section = ""
    if forecast_text:
        forecast_section = f"""
Projected forecast (NOT actual data):
{forecast_text}

Add one concise forecast observation and clearly label projected/estimated values.
"""

    prompt = f"""
You are the Analytics Agent Business Analyst.

Analyze ONLY the supplied analytics result.

User question:
{question}

Returned analytics table:
{data_text}
{forecast_section}

Return EXACTLY:
INSIGHTS:
• 2 to 4 specific, data-grounded observations with exact values where available.

SUMMARY:
1 to 2 concise sentences explaining the overall result and the key comparison.

Rules:
- Use ONLY the supplied table, forecast, and user question.
- Never invent numbers, dates, months, suppliers, routes, causes, or trends.
- Calculate comparisons, highest/lowest values, totals, averages, changes, or shares only from values actually present.
- When supplier/category data is present, name the highest and lowest when determinable.
- When monthly data is present, name the highest and lowest month when determinable.
- If there is only one returned metric/value, explain exactly what it shows.
- If data is insufficient, say so instead of guessing.
- Do not repeat the complete table.
- Do not mention SQL, AWS, Athena, Python, code, or implementation.
- Keep the response short and business-focused.
"""

    try:
        answer = generate_with_nemotron(
            prompt,
            temperature=0.2,
            max_tokens=500,
            timeout=120,
        )
    except Exception as error:
        print("Nemotron Insights/Summary error:", error)
        return (
            "• Unable to generate insights from the returned data.",
            "The analytics result was returned, but the business summary could not be generated.",
        )
    text = extract_response_text(answer, "INSIGHTS:\n• Unable to generate LLM insights.\n\nSUMMARY:\nUnable to generate the summary.")

    insights_match = re.search(r"INSIGHTS\s*:\s*(.*?)(?:\n\s*SUMMARY\s*:|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
    summary_match = re.search(r"SUMMARY\s*:\s*(.*)\Z", text, flags=re.IGNORECASE | re.DOTALL)

    insights = insights_match.group(1).strip() if insights_match else text.strip()
    summary = summary_match.group(1).strip() if summary_match else "The returned analytics result was analyzed using the available data."
    return insights, summary


def get_response(question, df):
    """Single LLM-based entry point for insights and summary generation."""
    return generate_llm_insights_summary(question, df)


def extract_response_text(result, fallback):

    if isinstance(result, str):

        return result.strip() or fallback

    if isinstance(result, dict):

        response = result.get("response")

        if isinstance(response, str):

            return response.strip() or fallback

        if isinstance(response, dict):

            for key in (
                "response",
                "answer",
                "content",
                "message"
            ):

                value = response.get(key)

                if (
                    isinstance(value, str)
                    and value.strip()
                ):

                    return value.strip()

        for key in (
            "answer",
            "content",
            "message"
        ):

            value = result.get(key)

            if (
                isinstance(value, str)
                and value.strip()
            ):

                return value.strip()

    return fallback


# =====================================================
# PREVIOUS ANALYTICS CHECK
# =====================================================

def has_previous_analytics_result():

    df = st.session_state.get(
        "last_analytics_df"
    )

    return (
        isinstance(df, pd.DataFrame)
        and not df.empty
    )


# =====================================================
# BUSINESS CONTEXT CHECK
# =====================================================

def get_business_context():

    business_df = st.session_state.get(
        "business_context_df"
    )

    if (
        isinstance(business_df, pd.DataFrame)
        and not business_df.empty
    ):

        return business_df

    analytics_df = st.session_state.get(
        "last_analytics_df"
    )

    if (
        isinstance(analytics_df, pd.DataFrame)
        and not analytics_df.empty
    ):

        return analytics_df

    return None


# =====================================================
# AWS FOLLOW-UP DETECTION
# =====================================================

def is_data_followup(question):

    q = (
        question or ""
    ).lower().strip()

    if not has_previous_analytics_result():

        return False

    followup_phrases = [

        "from that",
        "from this",
        "from those",
        "from these",
        "from above",
        "from the above",
        "from previous",
        "from the previous",

        "based on that",
        "based on this",
        "based on the above",

        "using that",
        "using this",

        "only confirmed",
        "confirmed records",
        "confirmed bookings",

        "only cancelled",
        "cancelled records",

        "only api",
        "api bookings",

        "those records",
        "these records",

        "the previous result",
        "previous result",

        "previous records",

        "above result",
        "above records",
        "above data",

        "the above data",
        "the above records",

        "that result",
        "this result",

        "from that result",
        "from this result"

    ]

    return any(
        phrase in q
        for phrase in followup_phrases
    )


# =====================================================
# BUSINESS QUESTION DETECTION
# =====================================================

def is_business_followup(question):

    q = (
        question or ""
    ).lower().strip()

    business_phrases = [

        "how can i improve",
        "how can we improve",
        "how should i improve",
        "how should we improve",

        "how can i increase",
        "how can we increase",

        "how should i increase",
        "how should we increase",

        "how can i reduce",
        "how can we reduce",

        "how should i reduce",
        "how should we reduce",

        "how to improve",
        "how to increase",
        "how to reduce",

        "what should i do",
        "what should we do",

        "what can i do",
        "what can we do",

        "suggest how",
        "suggest ways",

        "recommend",
        "recommendations",

        "strategy",
        "strategies",

        "improve sales",
        "increase sales",
        "grow sales",

        "improve revenue",
        "increase revenue",
        "grow revenue",

        "improve bookings",
        "increase bookings",

        "reduce cancellation",
        "reduce cancellations",

        "overcome the downfall",
        "overcome downfall",
        "overcome the down fall",
        "overcome that downfall",
        "overcome that down fall",

        "business improvement",
        "business recommendation",

        "company improvement",
        "company performance"

    ]

    return any(
        phrase in q
        for phrase in business_phrases
    )


# =====================================================
# COMPANY IMPROVEMENT QUESTION
# =====================================================

def is_company_improvement_question(question):

    q = (
        question or ""
    ).lower().strip()

    action_terms = [

        "improve",
        "increase",
        "reduce",
        "boost",
        "optimize",
        "grow",
        "overcome",
        "recommend",
        "suggest",
        "strategy"

    ]

    business_terms = [

        "sales",
        "revenue",
        "booking",
        "bookings",
        "cancellation",
        "profit",
        "performance",
        "supplier",
        "route",
        "airline",
        "segment",
        "brand",
        "month",
        "monthly",
        "trend",
        "downfall",
        "decline",
        "drop",
        "decrease"

    ]

    return (
        any(
            term in q
            for term in action_terms
        )
        and
        any(
            term in q
            for term in business_terms
        )
    )


# =====================================================
# BUILD AWS FOLLOW-UP QUESTION
# =====================================================

def build_followup_question(question):

    previous_question = st.session_state.get(
        "last_analytics_question",
        ""
    )

    previous_sql = st.session_state.get(
        "last_analytics_sql",
        ""
    )

    return f"""
Previous analytics question:
{previous_question}

Previous Athena SQL:
{previous_sql}

The user is asking a follow-up to the PREVIOUS ATHENA RESULT.

Follow-up rules:

1. Preserve every filter and condition from the
   previous request unless the user explicitly
   asks to remove or change it.

2. Treat these phrases as references to the
   previous analytics result:

   "from that"
   "from this"
   "from previous"
   "from the previous"
   "from above"
   "from the above"
   "those records"
   "these records"
   "above data"
   "previous result"
   "previous records"
   "above records"
   "the above data"
   "that result"
   "this result"

3. If the user asks for another subset,
   ADD the new condition to the previous
   conditions.

4. Example:

   Previous:
   "give confirmed records"

   New:
   "from the above give domestic records"

   The new SQL must represent BOTH:

   status = 'CONFIRMED'

   AND

   domestic condition.

5. Do NOT create an unrelated query.

6. Use only:

   <catalog>.<database>.<table>

7. Use the available schema.

8. If the user asks about API, supplier,
   domestic, international, confirmed,
   cancelled, etc., use the actual column
   and values available in the schema.

9. Return ONLY SQL.

New user request:
{question}
""".strip()
# =====================================================
# GENERATE BUSINESS IMPROVEMENT RESPONSE
# =====================================================


def generate_improvement_response(
    question,
    df
):

    if (
        df is None
        or df.empty
    ):

        return (
            "There is no previous analytics result "
            "available to analyze."
        )

    safe_df = df.copy()

    if len(safe_df) > 20:

        safe_df = safe_df.head(20)

    data_text = safe_df.to_string(
        index=False
    )

    previous_question = st.session_state.get(
        "last_analytics_question",
        ""
    )

    previous_sql = st.session_state.get(
        "last_analytics_sql",
        ""
    )

    prompt = f"""
You are the Analytics Agent
Business Improvement Analyst.

The user is asking for a company-focused
business recommendation.

Use ONLY the analytics result provided below.

Do NOT invent:
- numbers
- trends
- causes
- months
- suppliers
- routes
- performance values

Previous analytics question:
{previous_question}

Previous Athena SQL:
{previous_sql}

Previous analytics result:
{data_text}

Current user question:
{question}

Rules:

1. Give practical company-focused
   recommendations.

2. Every recommendation must be
   grounded in the available result.

3. If the result only contains a
   high-level metric such as:

   current_sales
   previous_sales
   mom_growth_percentage

   explain what the result shows.

4. If the result shows a decline,
   suggest practical areas to investigate
   or improve.

5. Do NOT claim a specific cause unless
   the analytics result proves it.

6. If more data is required to identify
   the exact cause, clearly say what
   analytics should be checked next.

7. Do not discuss:
   SQL
   AWS
   Athena
   Python
   programming
   IT implementation

8. Do not answer unrelated questions.

9. Keep the answer concise and useful
   for a business decision maker.
"""

    answer = generate_with_qwen(
        prompt,
        temperature=0.1,
        max_tokens=200,
        think=False,
        timeout=60,
    )

    return extract_response_text(
        answer,
        "I could not generate a recommendation "
        "from the previous analytics result."
    )
# =====================================================
# VISUALIZATION + FORECAST HELPERS
# =====================================================


def is_visualization_request(question):
    q = (question or "").lower().strip()
    visual_terms = [
        "visual", "visuals", "visualization", "visualizations",
        "visualisation", "visualisations", "visualise", "visualize",
        "chart", "charts", "graph", "graphs", "plot", "plots",
        "dashboard", "show graph", "show graphs", "show chart",
        "show charts", "graphical", "plot it", "plot this",
        "create a chart", "create charts", "create a graph",
        "create graphs", "generate a chart", "generate charts",
        "generate a graph", "generate graphs", "give a chart",
        "give charts", "give a graph", "give graphs",
        "make a chart", "make charts", "make a graph", "make graphs"
    ]
    return any(term in q for term in visual_terms)


def is_forecast_request(question):
    q = (question or "").lower().strip()
    forecast_terms = [
        "forecast", "forecasting", "predict", "prediction", "predicted",
        "upcoming months", "next month", "next months", "future months",
        "future", "projection", "projected", "expected", "estimate",
        "estimated", "upcoming", "what will", "what can we expect"
    ]
    return any(term in q for term in forecast_terms)


def is_business_visualization_request(question):
    q = (question or "").lower().strip()
    business_terms = [
        "improve", "increase", "reduce", "boost", "optimize", "optimise",
        "grow", "growth", "recommend", "recommendation", "strategy",
        "what should we", "what can we", "how can we", "how should we",
        "analyze", "analyse", "insight", "opportunity", "upcoming",
        "forecast", "predict", "future", "project"
    ]
    return is_visualization_request(q) and any(
        term in q for term in business_terms)


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
    """Find the business measure to plot without accidentally using year/month."""
    if df is None or df.empty:
        return None

    preferred = [
        "total_sales_amount", "sales", "total_sales", "sales_amount",
        "sales_volume", "revenue", "total_revenue", "average_sales_volume",
        "current_sales", "previous_sales", "number_of_bookings",
        "bookings", "booking_count", "number_of_pax", "pax",
        "number_of_segments", "number_of_tickets", "count", "total_count"
    ]

    lower_map = {str(c).lower().strip(): c for c in df.columns}

    # Never treat a time/helper column as the business metric.
    excluded = {
        "year", "month", "month_name", "month_year", "date",
        "booking_month", "_period", "_month_num", "_year_num", "forecast"
    }

    for candidate in preferred:
        if candidate in lower_map and candidate not in excluded:
            return lower_map[candidate]

    # Fallback: choose a numeric-looking column, but never year/month.
    numeric_candidates = []
    for col in df.columns:
        name = str(col).lower().strip()
        if name in excluded:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            numeric_candidates.append(col)

    return numeric_candidates[-1] if numeric_candidates else None


def prepare_yearly_series(df):
    """Return a clean yearly dataframe with period_label and a business metric.

    This is intentionally separate from monthly preparation.  A numeric
    ``year`` column must never be mistaken for the y-axis metric.
    """
    if df is None or df.empty:
        return None, None, None

    year_col = _find_year_column(df)
    metric_col = _find_metric_column(df)

    if year_col is None or metric_col is None:
        return None, year_col, metric_col

    work = df[[year_col, metric_col]].copy()
    work["_year_num"] = pd.to_numeric(work[year_col], errors="coerce")
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=["_year_num", metric_col]).copy()

    if work.empty:
        return None, year_col, metric_col

    # Keep plausible calendar years and sort chronologically.
    work["_year_num"] = work["_year_num"].astype(int)
    work = work[
        work["_year_num"].between(1900, 2200)
    ].copy()

    if work.empty:
        return None, year_col, metric_col

    grouped = (
        work.groupby("_year_num", as_index=False)[metric_col]
        .sum()
        .sort_values("_year_num")
    )
    grouped["period_label"] = grouped["_year_num"].astype(str)

    return (
        grouped[["_year_num", "period_label", metric_col]],
        year_col,
        metric_col,
    )


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


def create_monthly_forecast(df, periods=3):
    """Create a simple transparent linear-trend forecast for the next N months."""
    monthly, _, metric_col = prepare_monthly_series(df)
    if monthly is None or len(monthly) < 2:
        return None, monthly, metric_col

    values = pd.to_numeric(
    monthly[metric_col],
     errors="coerce").astype(float).to_numpy()
    x = np.arange(len(values), dtype=float)

    # Linear trend; never allow a negative forecast for sales-like metrics.
    slope, intercept = np.polyfit(x, values, 1)
    future_x = np.arange(len(values), len(values) + periods, dtype=float)
    forecast_values = slope * future_x + intercept
    if any(term in str(metric_col).lower()
           for term in ["sales", "revenue", "booking", "pax", "ticket"]):
        forecast_values = np.maximum(forecast_values, 0)

    last_period = monthly["_period"].iloc[-1]
    future_periods = pd.period_range(
    last_period + 1, periods=periods, freq="M")

    forecast_df = pd.DataFrame({
        "_period": future_periods,
        "period_label": future_periods.astype(str),
        metric_col: forecast_values,
        "forecast": True,
    })
    historical = monthly.copy()
    historical["forecast"] = False
    combined = pd.concat([historical, forecast_df], ignore_index=True)
    return combined, monthly, metric_col


def _requested_chart_type(question):
    """Detect the chart explicitly requested by the user."""
    q = (question or "").lower().strip()

    if any(term in q for term in ["pie", "donut", "doughnut"]):
        return "pie"
    if any(
    term in q for term in [
        "bar",
        "bar chart",
        "bar graph",
         "column chart"]):
        return "bar"
    if any(
    term in q for term in [
        "line",
        "line chart",
        "line graph",
         "trend"]):
        return "line"
    if any(term in q for term in ["area", "area chart", "area graph"]):
        return "area"

    return None


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


def render_visualization(df, question="", forecast=False):
    """Render an interactive Plotly chart from analytics data.

    This function only controls visualization. It does not modify the
    analytics dataframe, SQL generation, Athena execution, or follow-up logic.
    """

    if df is None or df.empty:
        st.info("No analytics data is available for visualization yet.")
        return False, None

    requested_type = _requested_chart_type(question)
    q = (question or "").lower().strip()

    # =====================================================
    # FORECAST
    # =====================================================
    if forecast:
        combined, monthly, metric_col = create_monthly_forecast(
            df,
            periods=3
        )

        if combined is None or metric_col is None:
            st.warning(
                "I need at least two monthly data points to create a forecast."
            )
            return False, None

        plot_df = combined.copy()
        historical = plot_df[plot_df["forecast"] == False].copy()
        projected = plot_df[plot_df["forecast"] == True].copy()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=historical["period_label"],
                y=historical[metric_col],
                mode="lines+markers",
                name="Historical",
                line=dict(width=3),
                marker=dict(size=8)
            )
        )

        forecast_x = pd.concat([
            historical["period_label"].tail(1),
            projected["period_label"]
        ])

        forecast_y = pd.concat([
            historical[metric_col].tail(1),
            projected[metric_col]
        ])

        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode="lines+markers",
                name="Forecast",
                line=dict(dash="dash", width=3),
                marker=dict(size=8)
            )
        )

        fig.update_layout(
            title=f"{metric_col} — 3 Month Forecast",
            template="plotly_white",
            height=500,
            hovermode="x unified",
            margin=dict(l=30, r=30, t=80, b=30)
        )

        st.plotly_chart(fig, width="stretch")
        return True, plot_df

    # =====================================================
    # YEARLY TREND / YEARLY COMPARISON
    # =====================================================
    # Only use the yearly chart when the user explicitly asks for a
    # year-level comparison.  When the result contains both ``year`` and
    # ``month`` columns (for example, monthly sales for one year), the
    # monthly trend below must be preferred so a single year does not
    # collapse all monthly points into one value.
    explicit_yearly = any(
        phrase in q
        for phrase in (
            "yearly",
            "annual",
            "by year",
            "per year",
            "each year",
            "year over year",
            "year-on-year",
            "yoy",
        )
    )

    yearly, _, yearly_metric = prepare_yearly_series(df)

    if explicit_yearly and yearly is not None and yearly_metric is not None:
        plot_df = yearly.copy()
        chart_type = requested_type or "line"

        # Pie charts are not appropriate for an ordered time series.
        if chart_type == "pie":
            chart_type = "line"

        if chart_type == "bar":
            fig = px.bar(
                plot_df,
                x="period_label",
                y=yearly_metric,
                text=yearly_metric,
                title=f"{yearly_metric} by Year"
            )
            fig.update_traces(
                texttemplate="%{text:,.2f}",
                textposition="outside"
            )
        elif chart_type == "area":
            fig = px.area(
                plot_df,
                x="period_label",
                y=yearly_metric,
                title=f"Yearly {yearly_metric} Trend"
            )
        else:
            fig = px.line(
                plot_df,
                x="period_label",
                y=yearly_metric,
                markers=True,
                title=f"Yearly {yearly_metric} Trend"
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=8))

        fig.update_layout(
            template="plotly_white",
            height=500,
            hovermode="x unified",
            margin=dict(l=30, r=30, t=80, b=30),
            xaxis_title="Year",
            yaxis_title=str(yearly_metric)
        )

        st.plotly_chart(fig, width="stretch")
        return True, plot_df

    # =====================================================
    # YEAR-ONLY FALLBACK
    # =====================================================
    # If the analytics result contains only a year column, never plot the
    # year itself as a measure.  Show a record count by year instead.
    year_only_col = _find_year_column(df)
    if year_only_col is not None and yearly_metric is None:
        year_values = pd.to_numeric(df[year_only_col], errors="coerce")
        year_values = year_values.dropna().astype(int)
        year_values = year_values[year_values.between(1900, 2200)]

        if not year_values.empty:
            year_counts = (
                year_values.value_counts()
                .sort_index()
                .rename_axis("year")
                .reset_index(name="record_count")
            )
            year_counts["year"] = year_counts["year"].astype(str)

            chart_type = requested_type or "bar"
            if chart_type == "pie":
                chart_type = "bar"

            if chart_type == "line":
                fig = px.line(
                    year_counts,
                    x="year",
                    y="record_count",
                    markers=True,
                    title="Records by Year"
                )
                fig.update_traces(line=dict(width=3), marker=dict(size=8))
            elif chart_type == "area":
                fig = px.area(
                    year_counts,
                    x="year",
                    y="record_count",
                    title="Records by Year"
                )
            else:
                fig = px.bar(
                    year_counts,
                    x="year",
                    y="record_count",
                    text="record_count",
                    title="Records by Year"
                )
                fig.update_traces(
                    texttemplate="%{text:,.0f}",
                    textposition="outside"
                )

            fig.update_layout(
                template="plotly_white",
                height=500,
                hovermode="x unified",
                margin=dict(l=30, r=30, t=80, b=30),
                xaxis_title="Year",
                yaxis_title="Record Count"
            )

            st.plotly_chart(fig, width="stretch")
            return True, year_counts

    # =====================================================
    # MONTHLY TREND
    # =====================================================
    monthly, _, metric_col = prepare_monthly_series(df)

    if monthly is not None and metric_col is not None:
        plot_df = monthly.copy()
        chart_type = requested_type or "line"

        # A pie chart is not appropriate for a time series.
        if chart_type == "pie":
            chart_type = "line"

        if chart_type == "bar":
            fig = px.bar(
                plot_df,
                x="period_label",
                y=metric_col,
                text=metric_col,
                title=f"{metric_col} by Month"
            )
            fig.update_traces(
    texttemplate="%{text:,.2f}",
     textposition="outside")

        elif chart_type == "area":
            fig = px.area(
                plot_df,
                x="period_label",
                y=metric_col,
                title=f"Monthly {metric_col} Trend"
            )

        else:
            fig = px.line(
                plot_df,
                x="period_label",
                y=metric_col,
                markers=True,
                title=f"Monthly {metric_col} Trend"
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=8))

        fig.update_layout(
            template="plotly_white",
            height=500,
            hovermode="x unified",
            margin=dict(l=30, r=30, t=80, b=30),
            xaxis_title="Month",
            yaxis_title=str(metric_col)
        )

        st.plotly_chart(fig, width="stretch")
        return True, plot_df

    # =====================================================
    # CATEGORY + METRIC
    # =====================================================
    categorical_cols = _category_columns(df)
    numeric_cols = _numeric_columns(df)

    # Prefer the user's explicitly requested chart.
    if categorical_cols and numeric_cols:
        category_col = categorical_cols[0]

        # Prefer the application's known business metric columns.
        preferred_metrics = [
            "total_sales_amount", "sales", "total_sales", "sales_amount",
            "revenue", "total_revenue", "average_sales_volume",
            "number_of_bookings", "bookings", "booking_count",
            "number_of_pax", "pax", "number_of_segments",
            "number_of_tickets"
        ]

        lower_map = {str(c).lower().strip(): c for c in numeric_cols}
        metric_col = next(
            (lower_map[name]
             for name in preferred_metrics if name in lower_map),
            numeric_cols[0]
        )

        chart_data = df[[category_col, metric_col]].copy()
        chart_data[metric_col] = pd.to_numeric(
            chart_data[metric_col],
            errors="coerce"
        )
        chart_data = chart_data.dropna(subset=[metric_col])

        # Aggregate duplicate categories so the chart represents the data
        # correctly instead of plotting repeated raw rows.
        chart_data = (
            chart_data.groupby(category_col, as_index=False)[metric_col]
            .sum()
            .sort_values(metric_col, ascending=False)
        )

        chart_type = requested_type

        if chart_type == "pie":
            # Pie needs non-negative values.
            pie_data = chart_data[chart_data[metric_col] >= 0].copy()

            if not pie_data.empty and pie_data[metric_col].sum() > 0:
                fig = px.pie(
                    pie_data,
                    names=category_col,
                    values=metric_col,
                    hole=0.35,
                    title=f"{metric_col} by {category_col}"
                )
                fig.update_traces(
                    textposition="inside",
                    textinfo="percent+label"
                )
            else:
                fig = px.bar(
                    chart_data,
                    x=category_col,
                    y=metric_col,
                    text=metric_col,
                    title=f"{metric_col} by {category_col}"
                )

        elif chart_type == "line":
            fig = px.line(
                chart_data,
                x=category_col,
                y=metric_col,
                markers=True,
                title=f"{metric_col} by {category_col}"
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=8))

        elif chart_type == "area":
            fig = px.area(
                chart_data,
                x=category_col,
                y=metric_col,
                title=f"{metric_col} by {category_col}"
            )

        else:
            fig = px.bar(
                chart_data,
                x=category_col,
                y=metric_col,
                text=metric_col,
                title=f"{metric_col} by {category_col}"
            )
            fig.update_traces(
    texttemplate="%{text:,.2f}",
     textposition="outside")

        fig.update_layout(
            template="plotly_white",
            height=500,
            margin=dict(l=30, r=30, t=80, b=30),
            xaxis_title=str(category_col),
            yaxis_title=str(metric_col)
        )

        st.plotly_chart(fig, width="stretch")
        return True, chart_data

    # =====================================================
    # MULTIPLE NUMERIC METRICS — COMPARISON / PIE
    # =====================================================
    if len(numeric_cols) >= 2 and len(df) == 1:
        metric_values = []
        for col in numeric_cols:
            value = pd.to_numeric(df[col].iloc[0], errors="coerce")
            if pd.notna(value):
                metric_values.append((col, float(value)))

        if len(metric_values) >= 2:
            pie_df = pd.DataFrame(metric_values, columns=["Metric", "Value"])

            if requested_type == "pie":
                pie_data = pie_df[pie_df["Value"] >= 0].copy()

                if not pie_data.empty and pie_data["Value"].sum() > 0:
                    fig = px.pie(
                        pie_data,
                        names="Metric",
                        values="Value",
                        hole=0.35,
                        title="Metric Distribution"
                    )
                    fig.update_traces(
    textposition="inside",
     textinfo="percent+label")
                else:
                    fig = px.bar(
                        pie_df,
                        x="Metric",
                        y="Value",
                        text="Value",
                        title="Metric Comparison"
                    )

            elif requested_type == "line":
                fig = px.line(
                    pie_df,
                    x="Metric",
                    y="Value",
                    markers=True,
                    title="Metric Comparison"
                )
                fig.update_traces(line=dict(width=3), marker=dict(size=8))

            else:
                fig = px.bar(
                    pie_df,
                    x="Metric",
                    y="Value",
                    text="Value",
                    title="Metric Comparison"
                )
                fig.update_traces(
    texttemplate="%{text:,.2f}",
     textposition="outside")

            fig.update_layout(
                template="plotly_white",
                height=500,
                margin=dict(l=30, r=30, t=80, b=30)
            )

            st.plotly_chart(fig, width="stretch")
            return True, pie_df

    # =====================================================
    # SINGLE NUMERIC METRIC FALLBACK
    # =====================================================
    if numeric_cols:
        metric_col = numeric_cols[0]
        values = pd.to_numeric(df[metric_col], errors="coerce")

        temp_df = pd.DataFrame({
            "Record": range(1, len(df) + 1),
            metric_col: values
        }).dropna()

        chart_type = requested_type or "bar"

        if chart_type == "line":
            fig = px.line(
                temp_df,
                x="Record",
                y=metric_col,
                markers=True,
                title=f"{metric_col} Analysis"
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=8))
        elif chart_type == "area":
            fig = px.area(
                temp_df,
                x="Record",
                y=metric_col,
                title=f"{metric_col} Analysis"
            )
        else:
            fig = px.bar(
                temp_df,
                x="Record",
                y=metric_col,
                text=metric_col,
                title=f"{metric_col} Analysis"
            )
            fig.update_traces(
    texttemplate="%{text:,.2f}",
     textposition="outside")

        fig.update_layout(
            template="plotly_white",
            height=500,
            margin=dict(l=30, r=30, t=80, b=30)
        )

        st.plotly_chart(fig, width="stretch")
        return True, temp_df

    st.warning("I could not identify suitable columns for visualization.")
    return False, None


def generate_business_visual_forecast_response(question, df, forecast_df=None):
    """Explain historical data and forecast without inventing data."""
    if df is None or df.empty:
        return "There is no previous analytics result available to analyze."

    historical = df.copy().head(80)
    history_text = historical.to_string(index=False)
    forecast_text = forecast_df.to_string(
    index=False) if forecast_df is not None else "No forecast was generated."
    previous_question = st.session_state.get("last_analytics_question", "")

    prompt = f"""
You are the Analytics Agent business analyst.

Use ONLY the supplied historical analytics result and forecast.
Do not invent numbers, causes, or business facts.

Previous analytics question:
{previous_question}

Historical analytics result:
{history_text}

Forecast generated by Python using a simple linear trend:
{forecast_text}

User request:
{question}

Rules:
- Explain the historical trend briefly.
- Clearly label forecast values as projected/estimated, not actual.
- Give practical company-focused recommendations tied to the data.
- If the historical data is insufficient to support a strong conclusion, say so.
- Do not discuss SQL, AWS, Athena, Python, or implementation details.
- Keep the answer concise and business-focused.
"""

    answer = generate_with_qwen(
        prompt,
        temperature=0.1,
        max_tokens=200,
        think=False,
        timeout=60,
    )
    return extract_response_text(
        answer,
        "I could not generate a business analysis from the available analytics data."
    )
def get_visual_insights_summary(question, df, forecast_df=None):
    """Generate visualization insights and summary in one LLM call."""
    return generate_llm_insights_summary(
        question,
        df,
        forecast_df=forecast_df,
    )

# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if "sql" in message:

            with st.expander(
                "Generated SQL"
            ):

                st.code(
                    message["sql"],
                    language="sql"
                )
        if message.get("insights"):
            st.markdown(textwrap.dedent(f"""
            <div style="
            background:white;
            padding:25px;
            border-radius:15px;
            border-left:6px solid #f59e0b;
            margin-top:15px;">
            <h3>💡 Insights</h3>
            <div style="line-height:1.8; white-space:pre-line;">{message['insights']}</div>
            </div>
            """).strip(), unsafe_allow_html=True)
        if message.get("summary"):
            st.markdown(textwrap.dedent(f"""
    <div style="
    background:white;
    padding:25px;
    border-radius:15px;
    border-left:6px solid #10b981;
    margin-top:15px;">
    <h3>📝 Summary</h3>
    <p>{message['summary']}</p>
    </div>
    """).strip(), unsafe_allow_html=True)
        if (
            "result" in message
            and message["result"] is not None
            and not message.get("show_chart")
            ):
            st.dataframe(
                message["result"],
                width="stretch"
                )
        if message.get("show_chart") and message.get("result") is not None:
            with st.expander("📈 Visualization", expanded=True):
                render_visualization(
                    message["result"],
                    message.get("question", ""),
                    forecast=message.get("forecast", False)
                )


# =====================================================
# CHAT INPUT
# =====================================================

# The question is first stored as a pending request and Streamlit is
# rerun immediately. This makes the disabled chat state reach the browser
# BEFORE any slow classifier/Ollama/Athena work starts.
question = st.chat_input(
    "Ask an analytical question...",
    disabled=st.session_state.processing
)

if question and not st.session_state.processing:
    current_question = question.strip()

    if not current_question:
        st.stop()

    st.session_state.pending_question = current_question
    st.session_state.processing = True

    # Show the user's message immediately, then rerun with the input locked.
    st.session_state.messages.append({
        "role": "user",
        "content": current_question
    })
    st.rerun()


# =====================================================
# PROCESS QUESTION
# =====================================================

if st.session_state.processing and st.session_state.pending_question:

    current_question = st.session_state.pending_question
    original_question = current_question

    with st.chat_message(
        "assistant",
        avatar="✈️"
    ):

        process = st.empty()

        sql = None

        df = None

        insights = (
            "No insights generated."
        )

        summary = (
            "No summary generated."
        )

        try:

            total_start = time.time()

            intent = "unknown"

            # =============================================
            # INTENT CLASSIFICATION
            # =============================================

            process.info(
                "🧠 Understanding your question..."
            )

            intent_result = classify_intent(
                current_question,
                previous_question=st.session_state.get(
                    "last_analytics_question",
                    ""
                ),
                previous_df=st.session_state.get(
                    "last_analytics_df"
                )
            )

            intent = intent_result.get(
                "intent",
                "unknown"
            )

            # The intent-classification message is only a temporary UI stage.
            # Clear it as soon as classification is finished so it cannot
            # remain underneath the LangGraph stage placeholder.
            process.empty()

            # =============================================
            # ROUTING PRIORITY
            # =============================================
            # Explicit visualization requests MUST be handled as
            # visualization requests.  A phrase such as:
            #   "from the above show visuals"
            #   "give a chart for the above data"
            # contains follow-up wording, but it must NOT be converted
            # back to aws_data/business after being detected as visual.
            #
            # Priority:
            #   1. Explicit visualization / chart / graph
            #   2. Business follow-up / company improvement
            #   3. AWS data follow-up
            #   4. Classifier result
            # =============================================

            if is_visualization_request(current_question):
                intent = "visualization"

            elif (
                is_business_followup(current_question)
                or
                (
                    intent == "business"
                    and is_company_improvement_question(
                        current_question
                    )
                )
            ):
                intent = "business"

            elif is_data_followup(current_question):
                intent = "aws_data"

            print(
                "\n================ INTENT ================\n"
            )
            print(intent)

            # =============================================
            # CONVERSATION
            # =============================================

            if intent == "conversation":

                result = (
                    generate_conversation_response(
                        current_question
                    )
                )

                answer = extract_response_text(
                    result,
                    "Hi! 👋 I'm the AI-Powered Analytics Agent. "
                    "How can I help you?"
                )

                st.markdown(
                    answer
                )

                previous_questions = [
                    message.get(
                        "content",
                        ""
                    )
                    for message
                    in st.session_state.messages
                    if message.get(
                        "role"
                    ) == "user"
                ]

                try:
                    suggestions = [
                        "Show monthly sales",
                        "Show bookings by supplier",
                        "Compare domestic and international bookings"
                        ]

                except Exception as suggestion_error:

                    print(
                        "Suggestion generator error:",
                        suggestion_error
                    )

                    suggestions = []

                suggestions = suggestions[:3]

                if suggestions:

                    st.markdown(
                        "### 💡 You can try:"
                    )

                    for suggestion in suggestions:

                        st.markdown(
                            f"• {suggestion}"
                        )

                conversation_content = answer

                if suggestions:

                    conversation_content += (
                        "\n\n### 💡 You can try:\n\n"
                        +
                        "\n".join(
                            f"• {suggestion}"
                            for suggestion
                            in suggestions
                        )
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": conversation_content
                    }
                )

                st.session_state.last_intent = (
                    "conversation"
                )

                process.empty()

                st.session_state.processing = False
                st.session_state.pending_question = None

                st.rerun()

            # =================================================
            # VISUALIZATION / FORECAST
            # =================================================
            elif intent == "visualization":

                previous_df = st.session_state.get(
                    "last_analytics_df"
                )

                if (
                    not isinstance(previous_df, pd.DataFrame)
                    or previous_df.empty
                ):
                    answer = (
                        "I can create charts and forecasts from your analytics result, "
                        "but there is no previous analytics data available. Please ask "
                        "an Athena analytics question first, for example: "
                        "'Give each month sales'."
                    )
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )
                    process.empty()
                    st.session_state.processing = False
                    st.session_state.pending_question = None
                    st.rerun()

                process.info("📊 Preparing visualization...")

                wants_forecast = is_forecast_request(current_question)
                chart_df = None
                forecast_df = None

                if wants_forecast:
                    combined_forecast, historical_monthly, metric_col = create_monthly_forecast(
                        previous_df,
                        periods=3
                    )
                    if combined_forecast is not None:
                        forecast_df = combined_forecast[
                            combined_forecast["forecast"] == True
                        ].copy()
                else:
                    combined_forecast = None

                visual_insights, visual_summary = get_visual_insights_summary(
                    current_question,
                    previous_df,
                    forecast_df
                )

                if is_business_visualization_request(current_question):
                    answer = generate_business_visual_forecast_response(
                        current_question,
                        previous_df,
                        forecast_df
                    )
                    st.markdown(answer)
                else:
                    answer = "Here is the visualization of the previous analytics result."
                    if wants_forecast:
                        answer += (
                            " The dashed section is a projected 3-month forecast "
                            "based on a simple historical linear trend."
                        )

                st.subheader("📈 Visualization")
                rendered, chart_df = render_visualization(
                    previous_df,
                    current_question,
                    forecast=wants_forecast
                )

                if not rendered:
                    st.warning(
                        "The analytics result was found, but a suitable "
                        "visualization could not be generated from its columns."
                    )

                if not is_business_visualization_request(current_question):
                    st.markdown(answer)

                # Visualization requests also get the same deep, data-grounded
                # Insights and Summary cards as normal analytics requests.
                st.markdown(textwrap.dedent(f"""
                <div style="background:white;padding:25px;border-radius:15px;
                box-shadow:0 4px 15px rgba(0,0,0,0.08);
                border-left:6px solid #f59e0b;margin-top:15px;">
                <h3>💡 Insights</h3>
                <div style="line-height:1.8;white-space:pre-line;">{visual_insights}</div>
                </div>
                """).strip(), unsafe_allow_html=True)

                st.markdown(textwrap.dedent(f"""
                <div style="background:white;padding:25px;border-radius:15px;
                box-shadow:0 4px 15px rgba(0,0,0,0.08);
                border-left:6px solid #10b981;margin-top:15px;">
                <h3>📝 Summary</h3>
                <div style="line-height:1.8;">{visual_summary}</div>
                </div>
                """).strip(), unsafe_allow_html=True)

                # Keep the visualization state explicit in chat history.
                # The chart is rendered again after Streamlit reruns from
                # this stored dataframe + original visualization question.
                message = {
                    "role": "assistant",
                    "content": answer,
                    "result": previous_df.copy(),
                    "show_chart": bool(rendered),
                    "forecast": wants_forecast,
                    "question": current_question,
                    "insights": visual_insights,
                    "summary": visual_summary,
                }
                st.session_state.messages.append(message)

                st.session_state.last_visualization_df = previous_df.copy()
                st.session_state.last_forecast_df = (
                    forecast_df.copy() if isinstance(forecast_df, pd.DataFrame) else None
                )

                process.empty()
                st.session_state.processing = False
                st.session_state.pending_question = None
                st.rerun()
                # =============================================
            # BUSINESS / COMPANY IMPROVEMENT
            # =============================================

            elif intent == "business":

                process.info(
                    "🧠 Analyzing business performance..."
                )

                business_df = (
                    get_business_context()
                )

                # -------------------------------------------------
                # No previous analytics
                # -------------------------------------------------

                if (
                    not isinstance(
                        business_df,
                        pd.DataFrame
                    )
                    or business_df.empty
                ):

                    answer = (
                        "I can help with company improvement "
                        "and business-performance questions, "
                        "but I need analytics data first.\n\n"
                        "Please ask an analytics question such as:\n"
                        "• give sales volume of june\n"
                        "• Show bookings by supplier\n"
                        "• Compare domestic and international bookings"
                    )

                else:

                    answer = (
                        generate_improvement_response(
                            current_question,
                            business_df
                        )
                    )

                    # -------------------------------------------------
                    # Preserve business analytics context
                    # -------------------------------------------------

                    st.session_state.business_context_df = (
                        business_df.copy()
                    )

                    st.session_state.business_context_question = (
                        st.session_state.get(
                            "last_analytics_question",
                            ""
                        )
                    )

                st.markdown(
                    answer
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

                st.session_state.last_intent = (
                    "business"
                )

                process.empty()

                st.session_state.processing = False
                st.session_state.pending_question = None

                st.rerun()

            # =============================================
            # AWS FOLLOW-UP
            # =============================================

            elif (
                intent == "aws_data"
                and is_data_followup(
                    current_question
                )
            ):

                process.info(
                    "🔄 Continuing from the previous analytics result..."
                )

                current_question = (
                    build_followup_question(
                        current_question
                    )
                )

                print(
                    "\n================ FOLLOW-UP CONTEXT ================\n"
                )

                print(
                    current_question
                )

                # IMPORTANT:
                # Do NOT reroute here.
                #
                # Continue directly into the AWS
                # processing block below.

            # =============================================
            # REJECT NON-ANALYTICS QUESTIONS
            # =============================================

            elif intent != "aws_data":

                rejection_message = (
                    "I can only help with your data environment "
                    "company analytics and business-performance "
                    "questions based on the available data.\n\n"
                    "Please ask an analytics question, "
                    "a company-improvement question, "
                    "or say hi/hello/bye."
                )

                st.warning(
                    rejection_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": rejection_message
                    }
                )

                st.session_state.last_intent = (
                    intent
                )

                process.empty()

                st.session_state.processing = False
                st.session_state.pending_question = None

                st.rerun()

            # =============================================
            # AWS DATA -> LANGGRAPH
            # =============================================

            if intent == "aws_data":

                # LangGraph marker is terminal-only. Nothing mentioning
                # LangGraph is rendered in the Streamlit UI.
                graph_start = time.perf_counter()
                print("\n🔗 LangGraph analytics workflow", flush=True)
                print("------------------------------------------------------------", flush=True)

                # One transient UI placeholder is used only for workflow
                # stage messages. It is cleared when the graph finishes.
                stage_area = st.empty()

                def show_stage(message):
                    stage_area.info(message)

                def show_stage_success(message):
                    stage_area.success(message)

                # These are permanent result areas. They are deliberately
                # separate from stage_area so SQL/table remain visible after
                # the transient workflow comments disappear.
                sql_result_area = st.empty()
                table_result_area = st.empty()

                # Schema loading remains outside the graph because the app
                # already caches this expensive operation.
                show_stage("📚 Loading Schema...")
                schema_start = time.time()
                schema = get_schema_cached()
                schema_end = time.time()

                normalized_question = normalize_question(current_question)

                print("\n================ ORIGINAL QUESTION ================\n", flush=True)
                print(current_question, flush=True)
                print("\n================ NORMALIZED QUESTION ================\n", flush=True)
                print(normalized_question, flush=True)

                try:
                    graph_confidence = float(intent_result.get("confidence", 1.0))
                except (TypeError, ValueError):
                    graph_confidence = 1.0

                graph_state = {
                    "user_query": normalized_question,
                    "intent": "aws_data",
                    "confidence": graph_confidence,
                    "previous_question": st.session_state.get("last_analytics_question", ""),
                    "previous_sql": st.session_state.get("last_analytics_sql", ""),
                    "previous_df": st.session_state.get("last_analytics_df"),
                    "schema": schema,
                    "repair_attempts": 0,
                }

                graph_result = {}
                graph_table_displayed = False

                for update in analytics_graph.stream(graph_state):
                    if not isinstance(update, dict):
                        continue

                    for node_name, node_update in update.items():
                        if isinstance(node_update, dict):
                            graph_result.update(node_update)

                        # understand_query is the first graph node. The app
                        # already classified intent, so the next visible stage
                        # is SQL generation.
                        if node_name == "understand_query":
                            show_stage("🧠 Generating SQL...")

                        elif node_name == "generate_sql":
                            sql = (
                                graph_result.get("generated_sql")
                                or graph_result.get("validated_sql")
                                or ""
                            )

                            if sql:
                                with sql_result_area.container():
                                    st.subheader("📄 Generated SQL")
                                    st.code(sql, language="sql")

                            show_stage_success("✅ SQL Generated")
                            # Immediately move to the validation stage.
                            show_stage("🔍 Validating SQL...")

                        elif node_name == "validate_sql":
                            validation_status = graph_result.get("validation_status")

                            if validation_status == "valid":
                                show_stage_success("✅ SQL Validation Passed")
                                show_stage("☁️ Executing SQL in Athena...")
                            elif validation_status in ("repair", "invalid", "failed"):
                                attempt = graph_result.get("repair_attempts", 1)
                                show_stage(f"🛠 Repairing SQL (attempt {attempt})...")
                            else:
                                # Keep the UI moving even if the validator
                                # uses a different status value.
                                show_stage("🔍 Validating SQL...")

                        elif node_name == "repair_sql":
                            repaired_sql = graph_result.get("generated_sql") or ""
                            if repaired_sql:
                                with sql_result_area.container():
                                    st.subheader("📄 Repaired SQL")
                                    st.code(repaired_sql, language="sql")
                            show_stage("🔍 Validating SQL...")

                        elif node_name == "execute_sql":
                            result_df = graph_result.get("result_df")

                            if graph_result.get("error_type") == "SESSION_EXPIRED":
                                show_stage("❌ AWS session token expired")
                            elif isinstance(result_df, pd.DataFrame):
                                df = result_df
                                display_df = format_numeric_dataframe(df)

                                with table_result_area.container():
                                    st.markdown("### 📊 Results")
                                    st.dataframe(display_df, width="stretch")

                                graph_table_displayed = True
                                show_stage_success(f"✅ Retrieved {len(df)} rows")
                                show_stage("💡 Creating insights and summary...")
                            else:
                                show_stage("🛠 Repairing SQL...")

                        elif node_name == "process_results":
                            df = graph_result.get("result_df")
                            insights = graph_result.get("insights") or "No insights generated."
                            summary = graph_result.get("summary") or "No summary generated."
                            show_stage_success("✅ Insights and summary created")

                        elif node_name == "failed":
                            error_type = graph_result.get("error_type")
                            if error_type == "SESSION_EXPIRED":
                                error_message = (
                                    "AWS credentials/session token have expired. "
                                    "Please refresh your AWS credentials."
                                )
                            else:
                                error_message = (
                                    graph_result.get("response")
                                    or graph_result.get("result_error")
                                    or graph_result.get("sql_error")
                                    or "Analytics workflow failed."
                                )
                            raise RuntimeError(str(error_message))

                sql = (
                    graph_result.get("validated_sql")
                    or graph_result.get("generated_sql")
                    or sql
                )
                if isinstance(graph_result.get("result_df"), pd.DataFrame):
                    df = graph_result["result_df"]
                insights = graph_result.get("insights") or insights
                summary = graph_result.get("summary") or summary

                # The transient workflow comments should disappear as soon as
                # LangGraph has completed. SQL/table/results stay on screen.
                stage_area.empty()

                graph_elapsed = time.perf_counter() - graph_start
                print("------------------------------------------------------------", flush=True)
                print(
                    f"🏁 Analytics workflow completed in {graph_elapsed:.2f} sec",
                    flush=True,
                )
                print("------------------------------------------------------------", flush=True)

                if df is None:
                    df = pd.DataFrame()

                formatted_df = format_numeric_dataframe(df)
                display_df = formatted_df

                total_end = time.time()
                execution_time = round(total_end - total_start, 2)

                st.success(f"Query completed in {execution_time} sec")
                st.info(
                    f"""
📚 Schema Load Time:
{round(schema_end - schema_start, 2)} sec

🧠 LangGraph Analytics Workflow Time:
{execution_time} sec

⚡ Athena Query Time:
See terminal stage timing

✅ Total Time:
{execution_time} sec
"""
                )

                # KPI CARDS
                # =============================================
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(
    textwrap.dedent("""
    <div style="
    background:#2563eb;
    color:white;
    padding:20px;
    border-radius:16px;
    text-align:center">
    <h4>Rows</h4>
    <h2>{}</h2>
    </div>
    """).format(len(df)).strip(),
    unsafe_allow_html=True
)
                with col2:
                    st.markdown(
    textwrap.dedent("""
    <div style="
    background:#7c3aed;
    color:white;
    padding:20px;
    border-radius:16px;
    text-align:center">
    <h4>Columns</h4>
    <h2>{}</h2>
    </div>
    """).format(len(df.columns)).strip(),
    unsafe_allow_html=True
)
                with col3:
                    st.markdown(textwrap.dedent(f"""
                    <div style="
                    background:#059669;
                    color:white;
                    padding:20px;
                    border-radius:16px;
                    text-align:center">
                    <h4>Time</h4>
                    <h2>{execution_time}s</h2>
                    </div>
                    """).strip(), unsafe_allow_html=True)
                with col4:
                    st.markdown(
    textwrap.dedent("""
    <div style="
    background:#ea580c;
    color:white;
    padding:20px;
    border-radius:16px;
    text-align:center">
    <h4>Status</h4>
    <h2>✅</h2>
    </div>
    """).strip(),
    unsafe_allow_html=True
)

                # =============================================
                # INSIGHTS
                # =============================================

                if (
                    insights
                    and
                    insights.strip()
                    and
                    insights.strip()
                    != "No insights generated."
                ):
                    st.markdown(textwrap.dedent(f"""
<div style="
background:white;
padding:25px;
border-radius:15px;
box-shadow:0 4px 15px rgba(0,0,0,0.1);
border-left:6px solid #f59e0b;
margin-top:15px;
">
<h3>💡 Insights</h3>
<div style="line-height:1.8; white-space:pre-line;">{insights}</div>
</div>
""").strip(), unsafe_allow_html=True)
                # =============================================
                # SUMMARY
                # =============================================

                if (
                    summary
                    and
                    summary.strip()
                ):
                    st.markdown(textwrap.dedent(f"""
<div style="
background:white;
padding:25px;
border-radius:15px;
box-shadow:0 4px 15px rgba(0,0,0,0.1);
border-left:6px solid #10b981;
margin-top:15px;
">
<h3>📝 Summary</h3>
<div style="line-height:1.8;">{summary}</div>
</div>
""").strip(), unsafe_allow_html=True)


# =============================================
# RESULTS
# =============================================
                    st.markdown(textwrap.dedent("""
                    <div class="card shadow-sm mb-3">
                    <div class="card-header">
                    📊 Results
                    </div>
                    </div>
                    """).strip(), unsafe_allow_html=True)

                    if display_df is not None and not graph_table_displayed:
                        st.dataframe(
                            display_df,
                            width="stretch"
                            )

                # =============================================
                # DOWNLOAD CSV
                # =============================================

                if (
                    formatted_df is not None
                    and display_df is not None
                ):

                    csv = (
                        display_df.to_csv(
                            index=False
                        )
                    )

                    st.download_button(
                        label="⬇ Download Results CSV",
                        data=csv,
                        file_name="query_results.csv",
                        mime="text/csv"
                    )

                # =============================================
                # SAVE ANALYTICS CONTEXT
                # =============================================

                st.session_state.last_analytics_df = (
                    df.copy()
                )

                st.session_state.last_analytics_sql = (
                    sql
                )

                # IMPORTANT:
                # Save the ORIGINAL user question,
                # not the huge follow-up prompt.

                st.session_state.last_analytics_question = (
                    original_question
                )
                st.session_state.last_visualization_df = (
                    df.copy()
                    )
                st.session_state.last_forecast_df = None

                # -------------------------------------------------
                # Also refresh business context.
                #
                # This means the next business question can
                # immediately use the newest AWS result.
                # -------------------------------------------------

                st.session_state.business_context_df = (
                    df.copy()
                )

                st.session_state.business_context_question = (
                    original_question
                )

                # =============================================
                # SAVE CHAT HISTORY
                # =============================================

                # IMPORTANT:
                # Keep Summary/Insights out of the main assistant content.
                # They are rendered separately from message["summary"] and
                # message["insights"]. Putting them in both places caused
                # duplicate Summary/Insights after st.rerun().
                assistant_content = (
                    "✅ Query executed successfully\n\n"
                    f"Rows Returned: "
                    f"{len(df) if df is not None else 0}"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "sql": sql,
                        "result": display_df,
                        "summary": summary,
                        "insights": insights
                    }
                )

                st.session_state.last_intent = (
                    "aws_data"
                )

                # =============================================
                # UNLOCK CHAT
                # =============================================

                st.session_state.processing = False
                st.session_state.pending_question = None

                st.rerun()

        # =================================================
        # ERROR HANDLING
        # =================================================

        except Exception as e:

            error_message = str(e)

            process.empty()

            error_title = (
                "Athena Error"
                if intent == "aws_data"
                else "Agent Error"
            )

            st.error(
                f"{error_title}: {error_message}"
            )


            # =============================================
            # ERROR RECOVERY - AWS ONLY
            # =============================================

            recovery_done = False

            if intent == "aws_data":

                try:

                    if (
                        sql is not None
                        and isinstance(
                            sql,
                            str
                        )
                        and sql.strip()
                    ):

                        schema = get_schema_cached()

                        corrected_sql = fix_sql(
                            sql,
                            error_message,
                            schema
                        )

                        if corrected_sql:

                            corrected_sql = (
                                normalize_generated_sql(
                                    corrected_sql.strip()
                                )
                            )

                            st.subheader(
                                "Suggested Corrected SQL"
                            )

                            st.code(
                                corrected_sql,
                                language="sql"
                            )

                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content":
                                        "❌ Athena Error\n\n"
                                        f"{error_message}\n\n"
                                        "Suggested correction available.",
                                    "sql":
                                        corrected_sql
                                }
                            )

                            recovery_done = True


                except Exception as recovery_error:

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content":
                                "❌ Athena Error\n\n"
                                f"{error_message}\n\n"
                                "SQL recovery failed: "
                                f"{str(recovery_error)}"
                        }
                    )

                    recovery_done = True


            # =============================================
            # GENERAL ERROR
            # =============================================

            if not recovery_done:

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content":
                            f"❌ {error_title}\n\n"
                            f"{error_message}"
                    }
                )


            # =============================================
            # UNLOCK PROCESSING
            # =============================================

            st.session_state.processing = False
            st.session_state.pending_question = None

            st.rerun()
