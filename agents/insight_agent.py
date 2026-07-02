from openai import OpenAI

from config import OPENAI_MODEL


def _get_client() -> OpenAI:
    return OpenAI()


def generate_insight(question: str, sql: str, data_preview: str) -> str:
    prompt = f"""
You are a business strategy analyst.
The user asked: {question}
SQL used: {sql}
Data preview:
{data_preview}

Write a concise response for a business user with:
1) What the data indicates
2) Why it matters
3) 2-3 concrete recommendations

Use plain language and avoid technical jargon.
"""

    response = _get_client().chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()


def generate_story(question: str, data_preview: str) -> str:
    prompt = f"""
You are a BI storyteller.
Create a concise narrative called "Story of the Data" for business users.
Use this format:

Step 1
...

Step 2
...

Step 3
...

Question:
{question}

Data preview:
{data_preview}

Focus on trends, category performance, and geography when available.
Use simple language and concrete business implications.
"""

    response = _get_client().chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()
