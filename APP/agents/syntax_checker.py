# agents/syntax_checker.py

"""
SQL validation layer for AI-Powered Analytics Agent.

Uses SQLGlot for structural SQL validation and performs
additional checks for Athena, schema, aliases, aggregation,
dates, MoM logic, and common SQL generation mistakes.
"""

import re

try:
    import sqlglot
    from sqlglot import exp
except ImportError:
    sqlglot = None
    exp = None


# =========================================================
# BASIC CONFIGURATION
# =========================================================

ALLOWED_TABLE = "<table_name>"


# =========================================================
# SIMPLE BOOLEAN VALIDATION
# =========================================================

def validate_sql(sql):
    """
    Backward-compatible validation function.

    Returns:
        True  -> SQL is structurally valid
        False -> SQL is invalid
    """

    result = detect_sql_issues(sql)

    return result.get("valid", False)


# =========================================================
# SQL CLEANING
# =========================================================

def clean_sql(sql):
    """
    Remove markdown fences and surrounding whitespace.
    """

    if sql is None:
        return ""

    sql = str(sql).strip()

    sql = re.sub(
        r"^```(?:sql|SQL)?\s*",
        "",
        sql
    )

    sql = re.sub(
        r"\s*```$",
        "",
        sql
    )

    return sql.strip()


# =========================================================
# PARENTHESES CHECK
# =========================================================

def check_parentheses(sql):

    depth = 0

    in_single_quote = False
    in_double_quote = False

    for char in sql:

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote

        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote

        elif not in_single_quote and not in_double_quote:

            if char == "(":
                depth += 1

            elif char == ")":

                depth -= 1

                if depth < 0:
                    return False

    return (
        depth == 0
        and not in_single_quote
        and not in_double_quote
    )


# =========================================================
# TABLE EXTRACTION
# =========================================================

def extract_tables(sql):

    tables = set()

    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+"
        r"(?:\"[^\"]+\"\.)?"
        r"([A-Za-z_][A-Za-z0-9_]*)",
        re.IGNORECASE
    )

    for match in pattern.finditer(sql):

        table = match.group(1)

        if table:
            tables.add(table.lower())

    return tables


# =========================================================
# COLUMN EXTRACTION FROM SCHEMA
# =========================================================

def extract_schema_columns(schema):

    columns = set()

    if schema is None:
        return columns

    # -----------------------------------------------------
    # Dictionary schema
    # -----------------------------------------------------

    if isinstance(schema, dict):

        for key, value in schema.items():

            key_lower = str(key).lower()

            if key_lower in {
                "columns",
                "column",
                "fields"
            }:

                if isinstance(value, (list, tuple, set)):

                    for item in value:

                        if isinstance(item, dict):

                            name = (
                                item.get("name")
                                or item.get("column_name")
                            )

                            if name:
                                columns.add(
                                    str(name).lower()
                                )

                        else:

                            columns.add(
                                str(item).lower()
                            )

                elif isinstance(value, str):

                    columns.add(
                        value.lower()
                    )

        # Sometimes schema itself is:
        # {"column1": "varchar", "column2": "double"}

        if not columns:

            for key in schema.keys():

                if key not in {
                    "table",
                    "table_name",
                    "database",
                    "schema",
                    "columns",
                    "fields"
                }:

                    columns.add(
                        str(key).lower()
                    )

    # -----------------------------------------------------
    # List schema
    # -----------------------------------------------------

    elif isinstance(schema, (list, tuple, set)):

        for item in schema:

            if isinstance(item, dict):

                name = (
                    item.get("name")
                    or item.get("column_name")
                )

                if name:
                    columns.add(
                        str(name).lower()
                    )

            else:

                columns.add(
                    str(item).lower()
                )

    # -----------------------------------------------------
    # String schema
    # -----------------------------------------------------

    elif isinstance(schema, str):

        for line in schema.splitlines():

            line = line.strip()

            if not line:
                continue

            # Typical:
            # column_name: varchar

            match = re.match(
                r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:|,\s]",
                line
            )

            if match:

                columns.add(
                    match.group(1).lower()
                )

    return columns


# =========================================================
# ATHENA FUNCTION CHECK
# =========================================================

def detect_athena_syntax_issues(sql):

    issues = []

    upper_sql = sql.upper()

    # -----------------------------------------------------
    # MySQL-style CURRENT_DATE()
    # -----------------------------------------------------

    if re.search(
        r"\bCURRENT_DATE\s*\(\s*\)",
        sql,
        re.IGNORECASE
    ):

        issues.append({
            "type": "CURRENT_DATE_DEPENDENCY",
            "severity": "WARNING",
            "message": (
                "CURRENT_DATE() is MySQL-style syntax. "
                "Athena should use CURRENT_DATE."
            )
        })

    # -----------------------------------------------------
    # MySQL DATE_SUB / DATE_ADD
    # -----------------------------------------------------

    if re.search(
        r"\bDATE_SUB\s*\(",
        sql,
        re.IGNORECASE
    ):

        issues.append({
            "type": "ATHENA_DATE_FUNCTION",
            "severity": "ERROR",
            "message": (
                "DATE_SUB() may use non-Athena syntax. "
                "Use Athena/Presto date_add/date_diff syntax."
            )
        })

    if re.search(
        r"\bDATE_ADD\s*\([^)]*,\s*INTERVAL\b",
        sql,
        re.IGNORECASE
    ):

        issues.append({
            "type": "ATHENA_DATE_FUNCTION",
            "severity": "ERROR",
            "message": (
                "MySQL-style DATE_ADD INTERVAL syntax "
                "was detected."
            )
        })

    # -----------------------------------------------------
    # PostgreSQL EXTRACT syntax is usually okay in Athena,
    # but interval syntax can be problematic.
    # -----------------------------------------------------

    if re.search(
        r"\bINTERVAL\s+'[^']+'\s+[A-Z]+",
        sql,
        re.IGNORECASE
    ):

        issues.append({
            "type": "UNSAFE_INTERVAL_SYNTAX",
            "severity": "ERROR",
            "message": (
                "Potential non-Athena interval syntax detected."
            )
        })

    return issues


# =========================================================
# NUMERIC LOWER / UPPER
# =========================================================

def detect_numeric_case_function(sql, schema=None):

    issues = []

    schema_columns = extract_schema_columns(schema)

    pattern = re.compile(
        r"\b(?:LOWER|UPPER)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        re.IGNORECASE
    )

    for match in pattern.finditer(sql):

        column = match.group(1)

        if column.lower() in schema_columns:

            # We cannot know datatype from a plain set,
            # so flag it for schema-aware repair.
            issues.append({
                "type": "LOWER_UPPER_COLUMN_CHECK",
                "severity": "WARNING",
                "message": (
                    f"LOWER()/UPPER() is applied to column "
                    f"'{column}'. Verify that it is a string column."
                )
            })

    return issues


# =========================================================
# AGGREGATION CHECK
# =========================================================

def detect_group_by_issues(sql):

    issues = []

    upper_sql = sql.upper()

    has_aggregate = bool(
        re.search(
            r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\(",
            upper_sql
        )
    )

    has_group_by = bool(
        re.search(
            r"\bGROUP\s+BY\b",
            upper_sql
        )
    )

    if has_aggregate and not has_group_by:

        # COUNT(*) alone is perfectly valid.
        only_count_star = bool(
            re.fullmatch(
                r"[\s\S]*COUNT\s*\(\s*\*\s*\)[\s\S]*",
                sql,
                re.IGNORECASE
            )
        )

        # Don't automatically reject all aggregate queries.
        # Flag only when multiple non-aggregate dimensions
        # appear likely.
        select_part = re.search(
            r"\bSELECT\b(.*?)\bFROM\b",
            sql,
            re.IGNORECASE | re.DOTALL
        )

        if select_part:

            select_text = select_part.group(1)

            cleaned = re.sub(
                r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\([^)]*\)",
                "",
                select_text,
                flags=re.IGNORECASE
            )

            if re.search(
                r"\b[A-Za-z_][A-Za-z0-9_]*\b",
                cleaned
            ):

                issues.append({
                    "type": "MISSING_GROUP_BY",
                    "severity": "ERROR",
                    "message": (
                        "Aggregate functions are used together "
                        "with non-aggregated selected columns "
                        "without GROUP BY."
                    )
                })

    return issues


# =========================================================
# MONTHLY GRAIN CHECK
# =========================================================

def detect_monthly_grain_issues(sql, question=""):

    issues = []

    q = str(question).lower()

    monthly_request = any(
        word in q
        for word in [
            "monthly",
            "each month",
            "per month",
            "month wise",
            "month-wise",
            "month over month",
            "mom"
        ]
    )

    if not monthly_request:
        return issues

    upper_sql = sql.upper()

    has_month = bool(
        re.search(
            r"\bMONTH\s*\(",
            upper_sql
        )
        or re.search(
            r"\bMONTH\b",
            upper_sql
        )
    )

    has_year = bool(
        re.search(
            r"\bYEAR\s*\(",
            upper_sql
        )
        or re.search(
            r"\bYEAR\b",
            upper_sql
        )
    )

    if has_month and not has_year:

        issues.append({
            "type": "MISSING_YEAR_MONTH_GRAIN",
            "severity": "ERROR",
            "message": (
                "Monthly aggregation uses month without "
                "year. This can combine the same month "
                "across different years."
            )
        })

    return issues


# =========================================================
# MOM CHECKS
# =========================================================

def detect_mom_issues(sql, question=""):

    issues = []

    q = str(question).lower()

    is_mom = any(
        phrase in q
        for phrase in [
            "month over month",
            "month-over-month",
            "mom growth",
            "mom"
        ]
    )

    if not is_mom:
        return issues

    upper_sql = sql.upper()

    # -----------------------------------------------------
    # Need previous/current comparison
    # -----------------------------------------------------

    comparison_keywords = [
        "PREVIOUS",
        "CURRENT",
        "LAG",
        "LEAD",
        "JOIN"
    ]

    if not any(
        keyword in upper_sql
        for keyword in comparison_keywords
    ):

        issues.append({
            "type": "MOM_LOGIC",
            "severity": "ERROR",
            "message": (
                "The question requests Month-over-Month "
                "analysis, but the SQL does not appear to "
                "compare two monthly periods."
            )
        })

    # -----------------------------------------------------
    # Division by zero
    # -----------------------------------------------------

    has_division = "/" in sql

    has_zero_protection = bool(
        re.search(
            r"\bNULLIF\s*\(",
            sql,
            re.IGNORECASE
        )
        or re.search(
            r"\bCASE\b.*?\bWHEN\b.*?=\s*0",
            sql,
            re.IGNORECASE | re.DOTALL
        )
    )

    if has_division and not has_zero_protection:

        issues.append({
            "type": "DIVISION_BY_ZERO",
            "severity": "ERROR",
            "message": (
                "Division is used without NULLIF or CASE "
                "protection against a zero denominator."
            )
        })

    return issues


# =========================================================
# NULL HANDLING
# =========================================================

def detect_null_issues(sql, question=""):

    issues = []

    q = str(question).lower()

    growth_request = any(
        word in q
        for word in [
            "growth",
            "percentage change",
            "percentage",
            "mom"
        ]
    )

    if growth_request and "/" in sql:

        upper_sql = sql.upper()

        if (
            "NULLIF(" not in upper_sql
            and "CASE" not in upper_sql
        ):

            issues.append({
                "type": "MISSING_NULL_HANDLING",
                "severity": "WARNING",
                "message": (
                    "Percentage/growth calculation may require "
                    "NULLIF or CASE to safely handle zero/null "
                    "denominators."
                )
            })

    return issues


# =========================================================
# ORDER BY CHECK
# =========================================================

def detect_order_by_issues(sql):

    issues = []

    match = re.search(
        r"\bORDER\s+BY\s+(.+?)(?:\bLIMIT\b|$)",
        sql,
        re.IGNORECASE | re.DOTALL
    )

    if not match:
        return issues

    order_text = match.group(1).strip()

    # Basic detection of obvious unresolved aliases.
    if re.search(
        r"\b[a-zA-Z_][A-Za-z0-9_]*\.[a-zA-Z_][A-Za-z0-9_]*\b",
        order_text
    ):

        issues.append({
            "type": "ORDER_BY_ALIAS",
            "severity": "WARNING",
            "message": (
                "ORDER BY contains a qualified column reference. "
                "Verify that the alias exists in the final query scope."
            )
        })

    return issues


# =========================================================
# SQLGLOT STRUCTURAL VALIDATION
# =========================================================

def sqlglot_validation(sql):

    issues = []

    if sqlglot is None:

        issues.append({
            "type": "SQLGLOT_NOT_INSTALLED",
            "severity": "WARNING",
            "message": (
                "SQLGlot is not installed. "
                "Install it with: python -m pip install sqlglot"
            )
        })

        return issues

    try:

        parsed = sqlglot.parse_one(
            sql,
            read="athena"
        )

        if parsed is None:

            issues.append({
                "type": "SQL_PARSE_ERROR",
                "severity": "ERROR",
                "message": "SQLGlot could not parse the SQL."
            })

    except Exception as error:

        issues.append({
            "type": "SQL_SYNTAX_ERROR",
            "severity": "ERROR",
            "message": str(error)
        })

    return issues


# =========================================================
# SCHEMA / TABLE VALIDATION
# =========================================================

def detect_schema_issues(sql, schema=None):

    issues = []

    tables = extract_tables(sql)

    if tables:

        allowed_table_lower = ALLOWED_TABLE.lower()

        invalid_tables = [
            table
            for table in tables
            if table != allowed_table_lower
        ]

        if invalid_tables:

            issues.append({
                "type": "INVALID_TABLE",
                "severity": "ERROR",
                "message": (
                    "Invalid or hallucinated table(s): "
                    + ", ".join(invalid_tables)
                )
            })

    return issues


# =========================================================
# MAIN ISSUE DETECTOR
# =========================================================

def detect_sql_issues(
    sql,
    schema=None,
    question=""
):

    issues = []

    sql = clean_sql(sql)

    # -----------------------------------------------------
    # Empty SQL
    # -----------------------------------------------------

    if not sql:

        issues.append({
            "type": "EMPTY_SQL",
            "severity": "ERROR",
            "message": "SQL is empty."
        })

        return {
            "valid": False,
            "issues": issues
        }

    # -----------------------------------------------------
    # SELECT / WITH
    # -----------------------------------------------------

    if not re.match(
        r"^\s*(SELECT|WITH)\b",
        sql,
        re.IGNORECASE
    ):

        issues.append({
            "type": "INVALID_SQL_START",
            "severity": "ERROR",
            "message": (
                "SQL must begin with SELECT or WITH."
            )
        })

    # -----------------------------------------------------
    # Parentheses
    # -----------------------------------------------------

    if not check_parentheses(sql):

        issues.append({
            "type": "UNBALANCED_PARENTHESES",
            "severity": "ERROR",
            "message": (
                "SQL contains unbalanced parentheses "
                "or unterminated quoted text."
            )
        })

    # -----------------------------------------------------
    # FROM
    # -----------------------------------------------------

    if not re.search(
        r"\bFROM\b",
        sql,
        re.IGNORECASE
    ):

        issues.append({
            "type": "MISSING_FROM",
            "severity": "ERROR",
            "message": "SQL does not contain FROM."
        })

    # -----------------------------------------------------
    # Schema
    # -----------------------------------------------------

    issues.extend(
        detect_schema_issues(
            sql,
            schema
        )
    )

    # -----------------------------------------------------
    # SQLGlot
    # -----------------------------------------------------

    issues.extend(
        sqlglot_validation(
            sql
        )
    )

    # -----------------------------------------------------
    # Athena
    # -----------------------------------------------------

    issues.extend(
        detect_athena_syntax_issues(
            sql
        )
    )

    # -----------------------------------------------------
    # LOWER / UPPER
    # -----------------------------------------------------

    issues.extend(
        detect_numeric_case_function(
            sql,
            schema
        )
    )

    # -----------------------------------------------------
    # GROUP BY
    # -----------------------------------------------------

    issues.extend(
        detect_group_by_issues(
            sql
        )
    )

    # -----------------------------------------------------
    # Monthly grain
    # -----------------------------------------------------

    issues.extend(
        detect_monthly_grain_issues(
            sql,
            question
        )
    )

    # -----------------------------------------------------
    # MoM
    # -----------------------------------------------------

    issues.extend(
        detect_mom_issues(
            sql,
            question
        )
    )

    # -----------------------------------------------------
    # NULL
    # -----------------------------------------------------

    issues.extend(
        detect_null_issues(
            sql,
            question
        )
    )

    # -----------------------------------------------------
    # ORDER BY
    # -----------------------------------------------------

    issues.extend(
        detect_order_by_issues(
            sql
        )
    )

    # -----------------------------------------------------
    # Deduplicate issues
    # -----------------------------------------------------

    unique = []

    seen = set()

    for issue in issues:

        key = (
            issue.get("type"),
            issue.get("message")
        )

        if key not in seen:

            seen.add(key)
            unique.append(issue)

    issues = unique

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    has_error = any(
        issue.get("severity") == "ERROR"
        for issue in issues
    )

    return {
        "valid": not has_error,
        "issues": issues
    }