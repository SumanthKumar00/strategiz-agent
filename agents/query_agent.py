import re
from typing import Optional

import duckdb
from openai import OpenAI

from config import DB_PATH, DEFAULT_ROW_LIMIT, OPENAI_MODEL

SCHEMA = """
orders(order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at,
       order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date)
customers(customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state)
order_items(order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value)
payments(order_id, payment_sequential, payment_type, payment_installments, payment_value)
products(product_id, product_category_name, product_name_lenght, product_description_lenght,
         product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm)
sellers(seller_id, seller_zip_code_prefix, seller_city, seller_state)
"""


def _get_client() -> OpenAI:
    return OpenAI()


def _get_connection():
    return duckdb.connect(DB_PATH, read_only=True)


def _strip_sql_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_sql_candidate(text: str) -> str:
    cleaned = _strip_sql_fences(text)
    lowered = cleaned.lower()
    start_idx = -1
    with_idx = lowered.find("with ")
    select_idx = lowered.find("select ")
    if with_idx != -1 and select_idx != -1:
        start_idx = min(with_idx, select_idx)
    elif with_idx != -1:
        start_idx = with_idx
    elif select_idx != -1:
        start_idx = select_idx
    if start_idx > 0:
        cleaned = cleaned[start_idx:]
    return cleaned.strip()


def _validate_readonly_sql(sql: str) -> None:
    lowered = sql.lower()
    # Remove single-line comments to avoid false positives in comment text.
    lowered = re.sub(r"--.*?$", "", lowered, flags=re.MULTILINE)
    # Collapse whitespace for simpler matching.
    normalized = re.sub(r"\s+", " ", lowered).strip()

    # Allow only SELECT/CTE-style read queries.
    if not (normalized.startswith("select ") or normalized.startswith("with ")):
        raise ValueError("Only read-only SELECT queries are allowed.")

    # Block DDL/DML statements using word boundaries.
    forbidden = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate|merge|replace)\b")
    if forbidden.search(normalized):
        raise ValueError("Only read-only SELECT queries are allowed.")

    if "select " not in normalized:
        raise ValueError("Generated SQL is invalid. Expected a SELECT query.")
    if re.search(r"select\s*\(\s*select", normalized):
        raise ValueError("Generated SQL must return one flat table. Nested SELECT columns are not allowed.")
    if "cross join" in normalized:
        raise ValueError("Generated SQL must return one flat table without CROSS JOIN of summary tables.")
    if " on true" in normalized:
        raise ValueError("Generated SQL must not use JOIN ... ON TRUE across summary tables.")


def _extract_year(question: str) -> Optional[int]:
    match = re.search(r"\b(20\d{2})\b", question or "")
    if not match:
        return None
    return int(match.group(1))


def _is_complex_dashboard_request(question: str) -> bool:
    q = (question or "").lower()
    if "dashboard" not in q:
        return False
    signals = [
        "kpi",
        "revenue",
        "orders",
        "category",
        "customer",
        "state",
        "distribution",
        "trend",
        "top",
    ]
    score = sum(1 for s in signals if s in q)
    return score >= 4


def _safe_dashboard_sql(question: str = "") -> str:
    year = _extract_year(question)
    year_filter = ""
    if year:
        year_filter = (
            f"\n  AND CAST(o.order_purchase_timestamp AS TIMESTAMP) >= CAST('{year}-01-01' AS TIMESTAMP)"
            f"\n  AND CAST(o.order_purchase_timestamp AS TIMESTAMP) < CAST('{year + 1}-01-01' AS TIMESTAMP)"
        )
    return f"""
WITH pay_agg AS (
    SELECT
        order_id,
        SUM(payment_value) AS order_payment_value
    FROM payments
    GROUP BY order_id
),
item_counts AS (
    SELECT
        order_id,
        COUNT(*) AS item_count
    FROM order_items
    GROUP BY order_id
)
SELECT
    DATE_TRUNC('month', CAST(o.order_purchase_timestamp AS TIMESTAMP)) AS order_month,
    o.order_id,
    o.customer_id,
    c.customer_state,
    oi.product_id,
    p.product_category_name,
    oi.seller_id,
    s.seller_state,
    (pa.order_payment_value / NULLIF(ic.item_count, 0)) AS payment_value
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
JOIN sellers s ON oi.seller_id = s.seller_id
JOIN pay_agg pa ON o.order_id = pa.order_id
JOIN item_counts ic ON o.order_id = ic.order_id
WHERE o.order_status = 'delivered'
{year_filter}
LIMIT {DEFAULT_ROW_LIMIT}
""".strip()


def generate_sql(question: str, previous_sql: Optional[str] = None, error_text: Optional[str] = None) -> str:
    if previous_sql is None and _is_complex_dashboard_request(question):
        return _safe_dashboard_sql(question)

    lower_q = (question or "").lower()
    lower_err = (error_text or "").lower()
    dashboard_like = any(
        k in lower_q
        for k in ["dashboard", "business performance", "total revenue", "top customers", "customer distribution", "revenue distribution"]
    )
    if dashboard_like and any(k in lower_err for k in ["binder error", "does not have a column", "not found", "join ... on true"]):
        return _safe_dashboard_sql(question)

    retry_context = ""
    if previous_sql and error_text:
        retry_context = f"""
Previous SQL that failed:
{previous_sql}

DuckDB error:
{error_text}

Fix the SQL so it runs correctly in DuckDB while preserving the business intent.
"""

    prompt = f"""
You are a senior analytics engineer.
Convert the business question into a single DuckDB SELECT query.
Rules:
- Use only these tables/columns:
{SCHEMA}
- Always generate one executable SQL query that returns one flat table (single result set).
- Never generate nested SELECT expressions that return tables, such as:
  SELECT (SELECT * FROM ...), (SELECT * FROM ...);
- Do not use SELECT subqueries in the projection list.
- Never use CROSS JOIN to combine independent summary tables.
- Never use JOIN ... ON TRUE to combine independent summary tables.
- If the question asks for multiple business perspectives, return a single denormalized analytic table
  at a consistent grain (for example: order_id/order_date/customer/category/state/revenue), not a Cartesian product.
- Do not mix GROUP BY aggregations with window functions over raw table columns in the same SELECT.
- If window metrics are needed, first aggregate in a CTE/subquery, then apply window functions in an outer SELECT.
- Prefer explicit JOIN keys:
  orders.customer_id = customers.customer_id
  orders.order_id = order_items.order_id
  orders.order_id = payments.order_id
  order_items.product_id = products.product_id
  order_items.seller_id = sellers.seller_id
- Use explicit joins with aliases exactly like:
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  JOIN order_items oi ON o.order_id = oi.order_id
  JOIN products p ON oi.product_id = p.product_id
  JOIN payments pay ON o.order_id = pay.order_id
  JOIN sellers s ON oi.seller_id = s.seller_id
- If a table alias/column is referenced, its table MUST be joined.
- Timestamp fields in this dataset are VARCHAR. Always cast before date functions/comparisons:
  DATE_TRUNC('month', CAST(o.order_purchase_timestamp AS TIMESTAMP))
  CAST(o.order_purchase_timestamp AS TIMESTAMP) >= CAST('2017-01-01' AS TIMESTAMP)
- Cast timestamp strings using CAST(... AS TIMESTAMP) when using DATE_TRUNC, DATE, EXTRACT, ORDER BY time, or date filters.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE.
- Include LIMIT {DEFAULT_ROW_LIMIT} unless the question explicitly requests all rows.
- Return SQL only, no explanation.
{retry_context}

Business question:
{question}
"""

    client = _get_client()
    current_prompt = prompt

    for attempt in range(3):
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[{"role": "user", "content": current_prompt}],
        )
        sql = _extract_sql_candidate(response.choices[0].message.content)
        try:
            _validate_readonly_sql(sql)
            return sql
        except ValueError as exc:
            if attempt == 2:
                return _safe_dashboard_sql(question)
            current_prompt = f"""
Your previous SQL violated constraints: {exc}

Return a corrected SQL query now.
Hard constraints:
- One flat SELECT result set only.
- No CROSS JOIN.
- No SELECT (SELECT ...).
- No projection-list subqueries.
- Use explicit joins only.
- Keep DuckDB-compatible syntax.

Business question:
{question}
"""

    return _safe_dashboard_sql(question)


def run_query(sql: str):
    _validate_readonly_sql(sql)
    with _get_connection() as con:
        return con.execute(sql).fetchdf()
