import re

from utils.ollama_client import generate_with_qwen


# =====================================================
# VALID INTENTS
# =====================================================
# conversation -> greetings / general chat / unsupported topics
# aws_data    -> questions that require company data from Athena
# business    -> company-improvement / business recommendations
VALID_INTENTS = {
    "conversation",
    "aws_data",
    "business",
    "visualization",
}


# =====================================================
# NORMALIZE
# =====================================================

def normalize(text):
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)

    replacements = {
        "waht is": "what is",
        "whats ": "what is ",
        "whta ": "what ",
        "wht ": "what ",
        "hte ": "the ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# =====================================================
# HELPERS
# =====================================================

def has_any(text, terms):
    return any(term in text for term in terms)


def has_previous_analytics_result(previous_df):
    return (
        previous_df is not None
        and hasattr(previous_df, "empty")
        and not previous_df.empty
    )


# =====================================================
# PREVIOUS RESULT REFERENCES
# =====================================================

def has_explicit_previous_reference(q):
    previous_terms = [
        "from that",
        "from this",
        "from the above",
        "from above",
        "from the previous",
        "from previous",
        "based on that",
        "based on this",
        "based on the above",
        "according to this",
        "according to that",
        "using this result",
        "using that result",
        "using the above",
        "using the previous",
        "previous result",
        "previous table",
        "previous data",
        "previous query",
        "the result",
        "this result",
        "that result",
        "the table",
        "this table",
        "that table",
        "the data above",
        "the above data",
        "from those records",
        "from these records",
        "those records",
        "these records",
        "those results",
        "these results",
        "same data",
        "same result",
        "that data",
        "this data",
        "that query",
        "this query",
        "what about that",
        "what about this",
        "compare that",
        "compare this",
    ]

    return has_any(q, previous_terms)


# =====================================================
# CONVERSATION
# =====================================================

def is_conversation(q):
    return q in {
        "hi",
        "hello",
        "hey",
        "hai",
        "hii",
        "hiii",
        "bye",
        "goodbye",
        "thanks",
        "thank you",
        "thankyou",
        "ok",
        "okay",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "how are you",
        "who are you",
        "what can you do",
        "nice",
        "great",
        "cool",
        "good",
        "yes",
        "no",
    }


# =====================================================
# VISUALIZATION / FORECAST
# =====================================================
VISUAL_TERMS = [
    "visual", "visuals", "visualization", "visualisations",
    "visualize", "visualise", "chart", "charts", "graph", "graphs",
    "plot", "plots", "dashboard", "graphical", "show graph",
    "show chart", "plot it", "plot this"
]

FORECAST_TERMS = [
    "forecast", "forecasting", "predict", "prediction", "predicted",
    "upcoming months", "next month", "next months", "future months",
    "future", "projection", "projected", "expected", "estimate",
    "estimated", "upcoming", "what will", "what can we expect"
]


def is_visualization_question(q):
    return has_any(q, VISUAL_TERMS)


def is_forecast_question(q):
    return has_any(q, FORECAST_TERMS)


# =====================================================
# BUSINESS / COMPANY IMPROVEMENT
# =====================================================

BUSINESS_ACTIONS = [
    "improve",
    "improvement",
    "increase",
    "reduce",
    "optimize",
    "optimise",
    "boost",
    "grow",
    "growth",
    "strategy",
    "strategies",
    "recommend",
    "recommendation",
    "recommendations",
    "action",
    "actions",
    "opportunity",
    "opportunities",
    "should we",
    "what should we",
    "what can we",
    "how can we",
    "how should we",
    "how do we",
    "ways to",
]

BUSINESS_METRICS = [
    "sales",
    "revenue",
    "booking",
    "bookings",
    "pax",
    "passenger",
    "passengers",
    "markup",
    "cancellation",
    "cancellations",
    "supplier",
    "suppliers",
    "airline",
    "airlines",
    "route",
    "routes",
    "segment",
    "segments",
    "ticket",
    "tickets",
    "brand",
    "brands",
    "performance",
    "profit",
    "margin",
    "trend",
]

BUSINESS_STARTS = [
    "how can we improve",
    "how can we increase",
    "how can we reduce",
    "how can the company improve",
    "how can the company increase",
    "how can the company reduce",
    "how should we improve",
    "how should we increase",
    "how should we reduce",
    "what should we do",
    "what can we do",
    "what strategy should",
    "what strategies should",
    "ways to improve",
    "ways to increase",
    "ways to reduce",
    "recommend ways",
    "give recommendations",
    "business recommendation",
    "business recommendations",
]


def is_business_question(q):
    # Strong business/recommendation wording.
    if has_any(q, BUSINESS_STARTS):
        return True

    # Generic action + company metric wording.
    return (
        has_any(q, BUSINESS_ACTIONS)
        and has_any(q, BUSINESS_METRICS)
    )


# =====================================================
# NEW ATHENA / ANALYTICS QUESTION
# =====================================================

ANALYTICS_TERMS = [
    "sales",
    "revenue",
    "booking",
    "bookings",
    "passenger",
    "passengers",
    "pax",
    "segment",
    "segments",
    "ticket",
    "tickets",
    "supplier",
    "suppliers",
    "airline",
    "airlines",
    "route",
    "routes",
    "brand",
    "brands",
    "markup",
    "cancellation",
    "cancellations",
    "confirmed",
    "cancelled",
    "canceled",
    "domestic",
    "international",
    "api",
    "trend",
    "growth",
    "month",
    "monthly",
    "year",
    "yearly",
    "mom",
    "yoy",
]

ANALYTICS_ACTIONS = [
    "show",
    "give",
    "find",
    "calculate",
    "count",
    "compare",
    "comparison",
    "list",
    "display",
    "get",
    "analyze",
    "analyse",
    "which",
    "how many",
    "how much",
    "highest",
    "lowest",
    "average",
    "total",
    "trend",
]


def is_new_analytics_question(q):
    strong_phrases = [
        "month over month",
        "month-on-month",
        "month on month",
        "year over year",
        "year-on-year",
        "year on year",
        "monthly sales",
        "monthly revenue",
        "monthly bookings",
        "monthly booking",
        "monthly trend",
        "yearly trend",
        "annual trend",
        "booking volume",
        "number of bookings",
        "number of passengers",
        "number of pax",
        "number of segments",
        "number of tickets",
        "supplier performance",
        "supplier level",
        "airline performance",
        "sales data",
        "revenue data",
        "booking data",
        "confirmed records",
        "cancelled records",
        "canceled records",
        "domestic records",
        "international records",
        "api data",
        "api records",
        "api bookings",
    ]

    if has_any(q, strong_phrases):
        return True

    return (
        has_any(q, ANALYTICS_TERMS)
        and has_any(q, ANALYTICS_ACTIONS)
    )


# =====================================================
# FOLLOW-UP / FILTER OPERATION
# =====================================================

FOLLOWUP_OPERATIONS = [
    "filter",
    "filtered",
    "filter by",
    "sort",
    "sorted",
    "sorting",
    "show only",
    "give only",
    "display only",
    "list only",
    "remove",
    "exclude",
    "top ",
    "bottom ",
    "only domestic",
    "only international",
    "only confirmed",
    "only cancelled",
    "only canceled",
    "only api",
    "give domestic",
    "give international",
    "give confirmed",
    "give cancelled",
    "give canceled",
    "give api",
    "show domestic",
    "show international",
    "show confirmed",
    "show cancelled",
    "show canceled",
    "show api",
    "api data",
    "api records",
    "api bookings",
    "breakdown",
    "break down",
    "more details",
    "more detail",
]


def is_followup_operation(q):
    return has_any(q, FOLLOWUP_OPERATIONS)


# =====================================================
# DIRECT COMPANY DATA TERMS
# =====================================================

COMPANY_DATA_TERMS = [
    "our data",
    "our sales",
    "our revenue",
    "our bookings",
    "our booking",
    "our airlines",
    "our airline",
    "our suppliers",
    "our supplier",
    "our routes",
    "our route",
    "our passengers",
    "our passenger",
    "our pax",
    "our segments",
    "our segment",
    "our tickets",
    "our ticket",
    "our brands",
    "our brand",
    "company data",
    "company sales",
    "company revenue",
    "company bookings",
    "booking data",
    "sales data",
    "revenue data",
    "from athena",
    "query athena",
    "in athena",
]


def has_company_data_terms(q):
    return has_any(q, COMPANY_DATA_TERMS)


# =====================================================
# PREVIOUS QUESTION LOOKS LIKE ANALYTICS
# =====================================================

def previous_question_is_analytics(previous_q):
    return has_any(previous_q, ANALYTICS_TERMS)


# =====================================================
# LLM FALLBACK
# =====================================================

def llm_classify(question):
    prompt = f"""
You are the intent classifier for the AI-Powered Analytics Agent.

Classify the user message into EXACTLY ONE intent:

aws_data
business
visualization
conversation

INTENT RULES

1. aws_data
Use aws_data ONLY when the user is asking to retrieve, filter,
compare, count, calculate, list, or analyze your data environment company data
that should come from the company's Athena dataset.

Examples:
- give confirmed records
- show bookings by supplier
- how many passengers do we have
- show API bookings
- compare sales by month
- from the above give API data
- filter the previous records to confirmed bookings

2. business
Use business when the user is asking for company/business
improvement, recommendations, strategy, opportunities, or actions.
This is NOT the same as retrieving raw data.

Examples:
- how can we improve sales
- how can the company reduce cancellations
- what should we do to increase revenue
- what strategy should we use to improve supplier performance
- what business opportunities can we pursue

If the business question explicitly asks to retrieve data first,
use aws_data instead.

3. visualization
Use visualization when the user asks for charts, graphs, visuals, plots,
visual analysis, or forecasts from the previous/current company analytics result.
Examples:
- give visuals for the above data
- show a chart of the monthly sales
- visualize the previous result
- analyze the data and give visuals for upcoming months

4. conversation
Use conversation for greetings, general chat, SQL explanations,
Python, AWS explanations, programming, IT, general knowledge,
or unrelated questions.

IMPORTANT:
- Never classify a general AWS/IT question as aws_data.
- Never classify a business recommendation as aws_data just because
  it mentions sales, revenue, bookings, suppliers, etc.
- Return ONLY one word.

User message:
{question}
""".strip()

    try:
        result = generate_with_qwen(
            prompt,
            think=False,
        )

        if isinstance(result, dict):
            response = result.get("response", "")
        else:
            response = str(result or "")

        response = normalize(response)

        # Exact/contained response is intentionally handled in this order.
        if re.search(r"\bvisualization\b", response):
            return {"intent": "visualization", "confidence": 0.92}

        if re.search(r"\baws_data\b", response):
            return {"intent": "aws_data", "confidence": 0.90}

        if re.search(r"\bbusiness\b", response):
            return {"intent": "business", "confidence": 0.90}

        return {"intent": "conversation", "confidence": 0.80}

    except Exception as error:
        print("LLM intent classification error:", error)

        # Fail closed: never send an uncertain question to Athena.
        return {
            "intent": "conversation",
            "confidence": 0.50,
        }


# =====================================================
# MAIN CLASSIFIER
# =====================================================

def classify_intent(
    question,
    previous_question="",
    previous_df=None,
):
    q = normalize(question)
    previous_q = normalize(previous_question)
    has_previous_result = has_previous_analytics_result(previous_df)

    # -------------------------------------------------
    # EMPTY
    # -------------------------------------------------
    if not q:
        return {
            "intent": "conversation",
            "confidence": 1.0,
        }

    # -------------------------------------------------
    # BASIC CONVERSATION
    # -------------------------------------------------
    if is_conversation(q):
        return {
            "intent": "conversation",
            "confidence": 1.0,
        }

    # -------------------------------------------------
    # VISUALIZATION FIRST
    #
    # A visual request is a capability request, not a new Athena
    # subset query. The app will use the latest analytics dataframe.
    # This also covers: "analyze the data and give visuals for
    # upcoming months".
    # -------------------------------------------------
    if is_visualization_question(q):
        if has_previous_result:
            return {
                "intent": "visualization",
                "confidence": 0.99,
            }
        # If there is no previous result, an explicit request for a
        # chart of company data can still be handled as visualization;
        # the app will tell the user to run an analytics query first.
        return {
            "intent": "visualization",
            "confidence": 0.92,
        }

    # -------------------------------------------------
    # BUSINESS FIRST
    #
    # This must come BEFORE generic previous-reference handling so:
    # "based on the above, how can we improve sales?" becomes business.
    # -------------------------------------------------
    if is_business_question(q):
        return {
            "intent": "business",
            "confidence": 0.98 if has_previous_result else 0.95,
        }

    # -------------------------------------------------
    # EXPLICIT FOLLOW-UP
    #
    # Any request that refers to previous records/data
    # and asks for another data subset is Athena data.
    #
    # Example:
    # First:  give confirmed records
    # Second: from the above give api data
    #
    # The previous result is used as context by app.py.
    # -------------------------------------------------
    if has_explicit_previous_reference(q):
        if has_previous_result:
            return {
                "intent": "aws_data",
                "confidence": 0.99,
            }

        # No previous result means we cannot safely interpret
        # "above/previous" as company data.
        return {
            "intent": "conversation",
            "confidence": 0.90,
        }

    # -------------------------------------------------
    # NEW ANALYTICS QUESTION
    # -------------------------------------------------
    if is_new_analytics_question(q):
        return {
            "intent": "aws_data",
            "confidence": 0.97,
        }

    # -------------------------------------------------
    # FOLLOW-UP OPERATION WITHOUT EXPLICIT REFERENCE
    #
    # If previous data exists, continue from it.
    # Otherwise allow clear data filters such as
    # "give confirmed records" as a new Athena request.
    # -------------------------------------------------
    if is_followup_operation(q):
        if has_previous_result:
            return {
                "intent": "aws_data",
                "confidence": 0.98,
            }

        if has_any(q, ANALYTICS_TERMS):
            return {
                "intent": "aws_data",
                "confidence": 0.94,
            }

    # -------------------------------------------------
    # COMPANY DATA TERMS
    # -------------------------------------------------
    if has_company_data_terms(q):
        return {
            "intent": "aws_data",
            "confidence": 0.98,
        }

    # -------------------------------------------------
    # ANALYTICS ACTION + DATA TERM
    # -------------------------------------------------
    if (
        has_any(q, ANALYTICS_TERMS)
        and has_any(q, ANALYTICS_ACTIONS)
    ):
        return {
            "intent": "aws_data",
            "confidence": 0.95,
        }

    # -------------------------------------------------
    # PREVIOUS RESULT + ANALYTICS WORDING
    # -------------------------------------------------
    if (
        has_previous_result
        and previous_question_is_analytics(previous_q)
        and has_any(
            q,
            [
                "more",
                "details",
                "breakdown",
                "compare",
                "show",
                "give",
                "find",
                "list",
            ],
        )
    ):
        return {
            "intent": "aws_data",
            "confidence": 0.94,
        }

    # -------------------------------------------------
    # LLM FINAL FALLBACK
    # -------------------------------------------------
    return {
    "intent": "conversation"
}