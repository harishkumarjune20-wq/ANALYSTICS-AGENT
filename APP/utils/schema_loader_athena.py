def get_schema():
    with open(
        "schema_cache.txt",
        "r",
        encoding="utf-8"
    ) as f:
        schema = f.read()

    return f"""
TABLE:
"<database>"."<table>"

COLUMNS:

{schema}
"""