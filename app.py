from pathlib import Path
import time
import io

import duckdb
import pandas as pd
import streamlit as st
from agents.query_agent import generate_sql, run_query
from agents.insight_agent import generate_insight, generate_story
from agents.report_agent import generate_report
from analytics.data_quality import generate_data_quality_report
from analytics.forecast import forecast_revenue
from config import DB_PATH, INSIGHT_PREVIEW_ROWS, REQUIRED_TABLES, SQL_RETRY_LIMIT, has_openai_api_key
from dashboard.charts import create_dashboard


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset+autosize",
    "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "resetScale2d"],
    "responsive": True,
}


def _startup_health_issues():
    issues = []
    db_file = Path(DB_PATH)

    if not has_openai_api_key():
        issues.append("Missing OPENAI_API_KEY. Set it in environment or .env.")

    if not db_file.exists():
        issues.append(f"DuckDB database not found at: {db_file}")
        return issues

    try:
        with duckdb.connect(str(db_file), read_only=True) as con:
            tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
            missing_tables = sorted(set(REQUIRED_TABLES) - tables)
            if missing_tables:
                issues.append(f"Missing required tables: {', '.join(missing_tables)}")
    except Exception as exc:
        issues.append(f"Unable to open DuckDB database: {exc}")

    return issues


def _col_by_keywords(columns, keywords):
    lowered = {col: col.lower() for col in columns}
    for col, low in lowered.items():
        if any(k in low for k in keywords):
            return col
    return None


def _extract_next_questions(report_text: str) -> str:
    if not report_text:
        return "No next questions generated."
    marker = "next questions"
    lower = report_text.lower()
    idx = lower.find(marker)
    if idx == -1:
        return "1. Which segment has the highest growth potential?\n2. What drivers explain top-performing regions?\n3. Which actions can lift revenue next quarter?"
    extracted = report_text[idx:]
    lines = extracted.splitlines()
    if len(lines) > 1:
        return "\n".join(lines[1:]).strip() or "No next questions generated."
    return "No next questions generated."


def _extract_recommended_actions(report_text: str) -> str:
    if not report_text:
        return "No recommended actions generated."
    lower = report_text.lower()
    start = lower.find("recommendations")
    end = lower.find("next questions")
    if start == -1:
        return "No recommended actions generated."
    if end != -1 and end > start:
        section = report_text[start:end]
    else:
        section = report_text[start:]
    lines = section.splitlines()
    if len(lines) > 1:
        actions = "\n".join(lines[1:]).strip()
        return actions or "No recommended actions generated."
    return "No recommended actions generated."


def _is_in_scope_question(question: str) -> bool:
    q = (question or "").lower()
    scope_keys = [
        "sales",
        "revenue",
        "customer",
        "order",
        "product",
        "category",
        "payment",
        "seller",
        "state",
        "trend",
        "forecast",
        "profit",
        "growth",
        "analytics",
    ]
    return any(k in q for k in scope_keys)


def _is_forecast_question(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in ["forecast", "predict", "future", "projection", "next month", "next quarter", "next year"])


def _build_forecast_insight(forecast_values: list[dict]) -> str:
    if not forecast_values:
        return "Forecast data is unavailable for this query."
    try:
        first = float(forecast_values[0].get("revenue", 0.0))
        last = float(forecast_values[-1].get("revenue", 0.0))
    except Exception:
        return "Forecast data is unavailable for this query."
    if first == 0:
        growth_pct = 0.0
    else:
        growth_pct = ((last - first) / abs(first)) * 100.0
    trend = "increase" if growth_pct >= 0 else "decrease"
    return (
        f"Forecast Summary: Revenue is projected to {trend} by approximately {abs(growth_pct):.1f}% "
        "over the next 6 months based on historical trend."
    )


def _chart_preview_for_key(df: pd.DataFrame, chart_key: str) -> str:
    if df is None or df.empty:
        return ""

    revenue_col = _col_by_keywords(df.columns, ["revenue", "payment_value", "price", "sales", "amount", "total"])
    orders_col = _col_by_keywords(df.columns, ["order_id", "orders"])
    customers_col = _col_by_keywords(df.columns, ["customer_id", "customer_unique_id", "customer"])
    category_col = _col_by_keywords(df.columns, ["product_category", "category"])
    state_col = _col_by_keywords(df.columns, ["customer_state", "state", "uf"])
    date_col = _col_by_keywords(df.columns, ["order_month", "month", "order_date", "purchase", "timestamp", "date", "time"])

    working = df.copy()
    if date_col:
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce")

    try:
        if chart_key in ["monthly_revenue_trend", "support_trend_line"] and date_col and revenue_col:
            trend = (
                working.dropna(subset=[date_col])
                .assign(month=working[date_col].dt.to_period("M").dt.to_timestamp())
                .groupby("month", as_index=False)[revenue_col]
                .sum()
                .sort_values("month")
            )
            return trend.head(INSIGHT_PREVIEW_ROWS).to_csv(index=False)

        if chart_key in ["category_treemap", "top_categories"] and category_col and revenue_col:
            cat = (
                working.groupby(category_col, as_index=False)[revenue_col]
                .sum()
                .sort_values(revenue_col, ascending=False)
                .head(INSIGHT_PREVIEW_ROWS)
            )
            return cat.to_csv(index=False)

        if chart_key in ["customer_segmentation", "top_customers"] and customers_col and revenue_col:
            agg_spec = {revenue_col: "sum"}
            if orders_col:
                agg_spec[orders_col] = "nunique"
            cust = (
                working.groupby(customers_col, as_index=False)
                .agg(agg_spec)
                .sort_values(revenue_col, ascending=False)
                .head(INSIGHT_PREVIEW_ROWS)
            )
            return cust.to_csv(index=False)

        if chart_key in ["customer_state_choropleth", "revenue_by_state_map", "state_map"] and state_col:
            if revenue_col:
                st_agg = (
                    working.groupby(state_col, as_index=False)[revenue_col]
                    .sum()
                    .sort_values(revenue_col, ascending=False)
                    .head(INSIGHT_PREVIEW_ROWS)
                )
            elif customers_col:
                st_agg = (
                    working.groupby(state_col, as_index=False)[customers_col]
                    .nunique()
                    .sort_values(customers_col, ascending=False)
                    .head(INSIGHT_PREVIEW_ROWS)
                )
            else:
                st_agg = working[[state_col]].value_counts().head(INSIGHT_PREVIEW_ROWS).reset_index(name="count")
            return st_agg.to_csv(index=False)

        if chart_key in ["revenue_distribution", "customer_revenue_distribution", "support_hist"] and revenue_col:
            return working[[revenue_col]].dropna().head(INSIGHT_PREVIEW_ROWS).to_csv(index=False)
    except Exception:
        pass

    return working.head(INSIGHT_PREVIEW_ROWS).to_csv(index=False)

def _build_full_report_pdf(report_text: str) -> tuple[bytes | None, str | None]:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception as exc:
        return None, f"PDF dependencies missing: {exc}"

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Strategiz Agent - Full Analysis Report")
    y -= 26
    c.setFont("Helvetica", 10)

    for raw_line in report_text.splitlines():
        line = raw_line if raw_line.strip() else " "
        chunks = [line[i:i + 105] for i in range(0, len(line), 105)] or [" "]
        for chunk in chunks:
            if y < 40:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 40
            c.drawString(40, y, chunk)
            y -= 14

    c.save()
    return buffer.getvalue(), None


def _suggest_followups(question: str) -> list[str]:
    q = (question or "").lower()
    if "forecast" in q or "future" in q:
        return [
            "Which categories are expected to drive forecasted growth?",
            "Which states may contribute most to next quarter revenue?",
            "How does forecasted growth compare with last year trends?",
        ]
    if "customer" in q or "state" in q:
        return [
            "Which customer segments contribute the highest revenue?",
            "What is the repeat purchase trend by state?",
            "Which regions show the best growth potential?",
        ]
    if "product" in q or "category" in q:
        return [
            "Which product categories have the highest month-over-month growth?",
            "Which categories drive the strongest customer retention?",
            "How does category performance vary by state?",
        ]
    return [
        "Which states contribute the highest revenue growth?",
        "Which categories are underperforming and need action?",
        "Who are the highest-value repeat customers this period?",
    ]

st.set_page_config(layout="wide", page_title="Strategiz Agent")
st.title("Strategiz Agent")
st.caption("Enterprise AI analytics for natural-language business questions")
st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #1b2a4a 0%, #0b1220 45%, #070b14 100%);
        color: #e5ecff;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.35rem 0 0.7rem 0;
        color: #d9e5ff;
        letter-spacing: .01em;
    }
    .kpi-card {
        background: rgba(17, 24, 39, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 10px 24px rgba(2, 8, 23, 0.35);
        margin-bottom: 10px;
    }
    .kpi-label {
        color: #a7b3cc;
        font-size: 0.83rem;
        margin-bottom: 0.2rem;
    }
    .kpi-value {
        color: #f3f7ff;
        font-size: 1.45rem;
        font-weight: 700;
    }
    .chart-card {
        background: rgba(17, 24, 39, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 14px;
        padding: 10px 10px 2px 10px;
        box-shadow: 0 10px 24px rgba(2, 8, 23, 0.32);
        margin-bottom: 12px;
    }
    .chart-title {
        color: #cbd7f1;
        font-size: 0.92rem;
        font-weight: 600;
        margin: 4px 6px 8px 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)

startup_issues = _startup_health_issues()
if startup_issues:
    for issue in startup_issues:
        st.error(issue)
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chart_explanations" not in st.session_state:
    st.session_state.chart_explanations = {}
if "active_analysis_question" not in st.session_state:
    st.session_state.active_analysis_question = ""

if not st.session_state.chat_history:
    st.markdown(
        """
<div style="
    margin: 8px 0 12px 0;
    padding: 10px 12px;
    border: 1px solid rgba(148,163,184,.24);
    border-radius: 10px;
    background: rgba(15,23,42,.45);
    color: #dbe7ff;
    font-size: 0.98rem;
">
Ask any sales, customer, product, or revenue question. I will build a complete dashboard, highlight key insights, and help you explore deeper with follow-up analysis.
</div>
""",
        unsafe_allow_html=True,
    )


if st.session_state.chat_history:
    st.markdown("### Conversation")
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])
            artifacts = msg.get("artifacts")
            if msg["role"] == "assistant" and artifacts:
                with st.expander("View Full Analysis", expanded=False):
                    st.markdown("**Generated SQL**")
                    st.code(artifacts.get("sql", ""), language="sql")

                    st.markdown("**KPI Snapshot**")
                    st.write(artifacts.get("kpi_summary", ""))

                    if artifacts.get("query_preview_rows"):
                        st.markdown("**Query Result Preview**")
                        st.dataframe(
                            pd.DataFrame(artifacts.get("query_preview_rows", []), columns=artifacts.get("query_preview_columns", [])),
                            use_container_width=True,
                        )

                    st.markdown("**Data Quality Summary**")
                    for line in artifacts.get("dq_lines", []):
                        st.write(f"- {line}")

                    st.markdown("**Business Insight**")
                    st.write(artifacts.get("insight", ""))

                    st.markdown("**Story of the Data**")
                    st.write(artifacts.get("story", ""))

                    st.markdown("**Executive Report**")
                    st.markdown(artifacts.get("report", ""))

                    st.markdown("**Recommended Actions**")
                    st.markdown(artifacts.get("recommended_actions", ""))

user_input = st.chat_input("Ask a business question or continue with a follow-up...")
is_new_question = bool(user_input and user_input.strip())
if is_new_question:
    question = user_input.strip()
    st.session_state.active_analysis_question = question
else:
    question = st.session_state.active_analysis_question

if question:
    try:
        if not _is_in_scope_question(question):
            st.info(
                "This question is outside the scope of the sales analytics dataset. "
                "Please ask a question related to sales, customers, products, or revenue analytics."
            )
            st.stop()

        if is_new_question:
            st.session_state.chat_history.append({"role": "user", "content": question})
        if st.session_state.active_analysis_question != question:
            st.session_state.chart_explanations = {}
            st.session_state.active_analysis_question = question

        thinking_box = st.empty()
        for msg in [
            "Analyzing business question...",
            "Generating SQL query...",
            "Running data analysis...",
            "Building analytics dashboard...",
            "Generating insights...",
        ]:
            thinking_box.info(msg)
            time.sleep(0.35)
        thinking_box.empty()

        sql = generate_sql(question)
        with st.expander("Generated SQL", expanded=False):
            st.code(sql, language="sql")

        try:
            data = run_query(sql)
        except Exception as first_query_error:
            if SQL_RETRY_LIMIT < 1:
                raise
            sql = generate_sql(question, previous_sql=sql, error_text=str(first_query_error))
            st.warning("Initial SQL failed. Retrying once with corrected SQL.")
            st.code(sql, language="sql")
            data = run_query(sql)

        with st.expander("Query Results", expanded=False):
            st.dataframe(data, use_container_width=True)

        state_col = _col_by_keywords(data.columns, ["customer_state", "state", "uf"])
        category_col = _col_by_keywords(data.columns, ["product_category", "category"])
        date_col = _col_by_keywords(data.columns, ["order_date", "order_month", "month", "purchase", "timestamp", "date", "time"])

        st.subheader("Dashboard Filters")
        fcol1, fcol2, fcol3 = st.columns(3)

        with fcol1:
            if state_col:
                state_options = ["All"] + sorted(data[state_col].dropna().astype(str).unique().tolist())
                state_filter = st.selectbox("Customer State", options=state_options, index=0)
            else:
                state_filter = "All"
                st.selectbox("Customer State", options=["All"], index=0, disabled=True)

        with fcol2:
            if category_col:
                cat_options = ["All"] + sorted(data[category_col].dropna().astype(str).unique().tolist())
                category_filter = st.selectbox("Product Category", options=cat_options, index=0)
            else:
                category_filter = "All"
                st.selectbox("Product Category", options=["All"], index=0, disabled=True)

        with fcol3:
            if date_col:
                date_series = pd.to_datetime(data[date_col], errors="coerce").dropna()
                if not date_series.empty:
                    min_date = date_series.min().date()
                    max_date = date_series.max().date()
                    date_filter = st.date_input("Date Range (order_date)", value=(min_date, max_date))
                else:
                    date_filter = None
                    st.date_input("Date Range (order_date)", disabled=True)
            else:
                date_filter = None
                st.date_input("Date Range (order_date)", disabled=True)

        filtered_data = data.copy()
        if state_col and state_filter != "All":
            filtered_data = filtered_data[filtered_data[state_col].astype(str) == state_filter]
        if category_col and category_filter != "All":
            filtered_data = filtered_data[filtered_data[category_col].astype(str) == category_filter]
        if date_col and date_filter is not None:
            filtered_data[date_col] = pd.to_datetime(filtered_data[date_col], errors="coerce")
            if hasattr(date_filter, "__len__") and len(date_filter) == 2:
                start_date, end_date = date_filter[0], date_filter[1]
                filtered_data = filtered_data[
                    (filtered_data[date_col].dt.date >= start_date)
                    & (filtered_data[date_col].dt.date <= end_date)
                ]
            elif hasattr(date_filter, "year"):
                filtered_data = filtered_data[filtered_data[date_col].dt.date == date_filter]

        st.subheader("Visualization")
        dashboard = create_dashboard(filtered_data.copy(), question=question)

        sections = dashboard.setdefault("sections", {})
        forecast = forecast_revenue(dashboard.get("clean_data", filtered_data.copy()))
        if forecast.get("figure") is not None and "revenue_forecast" not in sections:
            sections["revenue_forecast"] = forecast["figure"]

        revenue_col = _col_by_keywords(filtered_data.columns, ["revenue", "payment_value", "price", "sales", "amount", "total"])
        orders_col = _col_by_keywords(filtered_data.columns, ["order_id", "orders"])
        customers_col = _col_by_keywords(filtered_data.columns, ["customer_id", "customer_unique_id", "customer"])

        total_revenue = float(filtered_data[revenue_col].sum()) if revenue_col else 0.0
        total_orders = int(filtered_data[orders_col].nunique()) if orders_col else int(len(filtered_data))
        total_customers = int(filtered_data[customers_col].nunique()) if customers_col else int(len(filtered_data))
        avg_order_value = (total_revenue / total_orders) if total_orders else 0.0

        st.markdown('<div class="section-title">KPI Overview</div>', unsafe_allow_html=True)
        kpi_cols = st.columns(4)
        kpi_items = [
            ("Total Revenue", f"${total_revenue:,.2f}"),
            ("Total Orders", f"{total_orders:,}"),
            ("Average Order Value", f"${avg_order_value:,.2f}"),
            ("Total Customers", f"{total_customers:,}"),
        ]
        for i, (label, value) in enumerate(kpi_items):
            with kpi_cols[i]:
                st.markdown(
                    f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

        def render_chart_card(col, key: str, title: str):
            fig = sections.get(key)
            if fig is None:
                return
            with col:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                if st.button("Explain this chart", key=f"explain_{key}"):
                    with st.spinner("Generating chart explanation..."):
                        chart_preview = _chart_preview_for_key(filtered_data, key)
                        chart_explanation = generate_insight(
                            f"Explain this specific chart: '{title}' (chart_key={key}). "
                            "Use the chart dataset preview only. Start with 'This chart shows ...' and provide 2-3 concise observations with business implications.",
                            "chart_explanation",
                            chart_preview,
                        )
                        st.session_state.chart_explanations[key] = chart_explanation

                if key in st.session_state.chart_explanations:
                    with st.expander("Chart Explanation", expanded=True):
                        st.write(st.session_state.chart_explanations[key])
                        if st.button("Hide explanation", key=f"hide_explain_{key}"):
                            st.session_state.chart_explanations.pop(key, None)
                            st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Analytics Dashboard</div>', unsafe_allow_html=True)

        chart_key_fallbacks = {
            "trend": ["monthly_revenue_trend", "support_trend_line"],
            "treemap": ["category_treemap", "top_categories"],
            "scatter": ["customer_segmentation", "support_scatter", "top_customers"],
            "geo": ["customer_state_choropleth", "revenue_by_state_map", "state_map"],
            "distribution": ["revenue_distribution", "customer_revenue_distribution", "support_hist"],
        }

        def pick_key(group: str):
            for k in chart_key_fallbacks[group]:
                if k in sections:
                    return k
            return None

        trend_key = pick_key("trend")
        treemap_key = pick_key("treemap")
        scatter_key = pick_key("scatter")
        geo_key = pick_key("geo")
        distribution_key = pick_key("distribution")

        row1 = st.columns(2)
        render_chart_card(row1[0], trend_key, "Revenue Trend Over Time")
        render_chart_card(row1[1], treemap_key, "Product Category Revenue")

        row2 = st.columns(2)
        render_chart_card(row2[0], scatter_key, "Customer Segmentation")
        render_chart_card(row2[1], geo_key, "Customer Distribution by State")

        row3 = st.columns(2)
        render_chart_card(row3[0], distribution_key, "Revenue Distribution")
        render_chart_card(row3[1], "top_customers", "Top Customers by Revenue")

        if dashboard["single_chart"] is not None and all(v is None for v in [trend_key, treemap_key, scatter_key, geo_key, distribution_key]):
            st.plotly_chart(dashboard["single_chart"], use_container_width=True, config=PLOTLY_CONFIG)

        dq = generate_data_quality_report(filtered_data)

        preview = filtered_data.head(INSIGHT_PREVIEW_ROWS).to_csv(index=False)
        insight = generate_insight(question, sql, preview)
        story = generate_story(question, preview)
        report = generate_report(question, sql, insight, len(filtered_data))
        recommended_actions = _extract_recommended_actions(report)

        if _is_forecast_question(question):
            forecast_note = _build_forecast_insight(forecast.get("forecast_values", []))
            insight = f"{insight}\n\n{forecast_note}"

        with st.expander("Data Quality Report", expanded=False):
            for line in dq["summary_lines"]:
                st.write(line)
            st.dataframe(dq["missing_by_column_pct"], use_container_width=True)
            st.dataframe(dq["basic_stats"], use_container_width=True)

        with st.expander("Business Insight", expanded=False):
            st.write(insight)

        with st.expander("Story of the Data", expanded=False):
            st.write(story)

        with st.expander("Executive Report", expanded=False):
            st.markdown(report)
        full_report_text = (
            f"Question: {question}\n\n"
            f"Generated SQL:\n{sql}\n\n"
            f"KPI Snapshot:\n"
            f"- Total Revenue: ${total_revenue:,.2f}\n"
            f"- Total Orders: {total_orders:,}\n"
            f"- Average Order Value: ${avg_order_value:,.2f}\n"
            f"- Total Customers: {total_customers:,}\n\n"
            f"Data Quality Summary:\n" + "\n".join(f"- {line}" for line in dq.get("summary_lines", [])) + "\n\n"
            f"Business Insight:\n{insight}\n\n"
            f"Story of the Data:\n{story}\n\n"
            f"Executive Report:\n{report}\n\n"
            f"Recommended Actions:\n{recommended_actions}\n"
        )
        full_report_pdf, full_report_pdf_error = _build_full_report_pdf(full_report_text)
        if is_new_question:
            assistant_summary = (
                f"Completed analysis for: '{question}'. "
                f"Total rows analyzed: {len(filtered_data)}. "
                f"Top insight: {insight[:220]}{'...' if len(insight) > 220 else ''}"
            )
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": assistant_summary,
                    "artifacts": {
                        "sql": sql,
                        "kpi_summary": (
                            f"Total Revenue: ${total_revenue:,.2f} | "
                            f"Total Orders: {total_orders:,} | "
                            f"Average Order Value: ${avg_order_value:,.2f} | "
                            f"Total Customers: {total_customers:,}"
                        ),
                        "query_preview_columns": filtered_data.columns.tolist(),
                        "query_preview_rows": filtered_data.head(20).to_dict(orient="records"),
                        "dq_lines": dq.get("summary_lines", []),
                        "insight": insight,
                        "story": story,
                        "report": report,
                        "recommended_actions": recommended_actions,
                    },
                }
            )

        st.subheader("Download")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                label="Download CSV",
                data=filtered_data.to_csv(index=False),
                file_name="strategiz_query_results.csv",
                mime="text/csv",
            )
        with dcol2:
            if full_report_pdf is not None:
                st.download_button(
                    label="Download Full Report (PDF)",
                    data=full_report_pdf,
                    file_name="strategiz_full_report.pdf",
                    mime="application/pdf",
                )
            else:
                st.button("Download Full Report (PDF)", disabled=True)
                if full_report_pdf_error:
                    st.caption(f"PDF export unavailable: {full_report_pdf_error}. Install `reportlab`.")
    except Exception as exc:
        st.error(f"Unable to complete analysis: {exc}")







