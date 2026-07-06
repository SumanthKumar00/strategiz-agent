from openai import OpenAI

from config import OPENAI_MODEL


def generate_report(question: str, sql: str, insight: str, row_count: int) -> str:
    prompt = f"""
Create an executive-ready analytics summary.

Question: {question}
SQL: {sql}
Rows returned: {row_count}
Insights:
{insight}

Return markdown with these sections:
- Summary
- Recommendations
- Next Questions

Keep it short and actionable for leadership.
"""

    response = OpenAI().chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()
