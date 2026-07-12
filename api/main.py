from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.insight_agent import generate_insight, generate_story
from agents.query_agent import generate_sql, run_query
from agents.report_agent import generate_report
from analytics.data_quality import generate_data_quality_report
from analytics.forecast import forecast_revenue
from config import DB_PATH, INSIGHT_PREVIEW_ROWS, REQUIRED_TABLES, SQL_RETRY_LIMIT, has_openai_api_key
from dashboard.charts import create_dashboard


class AnalyzeRequest(BaseModel):
    question: str


app = FastAPI(title="Strategiz Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _extract_next_questions(report_text: str) -> str:
    if not report_text:
        return "No next questions generated."
    marker = "next questions"
    lower = report_text.lower()
    idx = lower.find(marker)
    if idx == -1:
        return "1. Which segment has the highest growth potential?\\n2. What drivers explain top-performing regions?\\n3. Which actions can lift revenue next quarter?"
    extracted = report_text[idx:]
    lines = extracted.splitlines()
    if len(lines) > 1:
        return "\\n".join(lines[1:]).strip() or "No next questions generated."
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
        actions = "\\n".join(lines[1:]).strip()
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


def _to_jsonable_df(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": df.columns.tolist(),
        "rows": df.to_dict(orient="records"),
        "row_count": int(len(df)),
    }


def _serialize_sections(sections: dict[str, Any]) -> dict[str, Any]:
    serialized = {}
    for key, fig in sections.items():
        if hasattr(fig, "to_plotly_json"):
            serialized[key] = fig.to_plotly_json()
    return serialized


@app.get("/health")
def health():
    issues = _startup_health_issues()
    return {"ok": len(issues) == 0, "issues": issues}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    issues = _startup_health_issues()
    if issues:
        raise HTTPException(status_code=500, detail={"health_issues": issues})

    if not _is_in_scope_question(question):
        return {
            "in_scope": False,
            "message": "This question is outside the scope of the sales analytics dataset. Please ask a question related to sales, customers, products, or revenue analytics.",
        }

    try:
        sql = generate_sql(question)
        try:
            data = run_query(sql)
        except Exception as first_query_error:
            if SQL_RETRY_LIMIT < 1:
                raise
            sql = generate_sql(question, previous_sql=sql, error_text=str(first_query_error))
            data = run_query(sql)

        dashboard = create_dashboard(data.copy(), question=question)
        sections = dashboard.setdefault("sections", {})
        forecast = forecast_revenue(dashboard.get("clean_data", data.copy()))
        if forecast.get("figure") is not None and "revenue_forecast" not in sections:
            sections["revenue_forecast"] = forecast["figure"]

        preview = data.head(INSIGHT_PREVIEW_ROWS).to_csv(index=False)
        insight = generate_insight(question, sql, preview)
        story = generate_story(question, preview)
        report = generate_report(question, sql, insight, len(data))

        dq = generate_data_quality_report(data)

        return {
            "in_scope": True,
            "question": question,
            "sql": sql,
            "query_result": _to_jsonable_df(data),
            "dashboard": {
                "layout_rows": dashboard.get("layout_rows", []),
                "sections": _serialize_sections(sections),
            },
            "insight": insight,
            "story": story,
            "report": report,
            "recommended_actions": _extract_recommended_actions(report),
            "next_questions": _extract_next_questions(report),
            "data_quality": {
                "summary_lines": dq.get("summary_lines", []),
                "missing_by_column_pct": dq.get("missing_by_column_pct", pd.DataFrame()).to_dict(orient="records"),
                "basic_stats": dq.get("basic_stats", pd.DataFrame()).to_dict(orient="records"),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
