import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("STRATEGIZ_DB_PATH", str(BASE_DIR / "strategiz.db"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_ROW_LIMIT = int(os.getenv("DEFAULT_ROW_LIMIT", "500"))
INSIGHT_PREVIEW_ROWS = int(os.getenv("INSIGHT_PREVIEW_ROWS", "30"))
SQL_RETRY_LIMIT = int(os.getenv("SQL_RETRY_LIMIT", "1"))

REQUIRED_TABLES = (
    "orders",
    "customers",
    "order_items",
    "payments",
    "products",
    "sellers",
)


def has_openai_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))
