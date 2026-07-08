import pandas as pd


def generate_data_quality_report(df: pd.DataFrame):
    if df is None:
        df = pd.DataFrame()

    working = df.copy()
    working = working.replace(["undefined", "None", "", "null", "NULL", "nan", "NaN"], pd.NA)
    total_rows = len(working)
    total_cols = len(working.columns)

    if total_rows == 0:
        missing_pct = pd.DataFrame(columns=["column", "missing_pct"])
    else:
        missing_pct = (
            (working.isna().sum() / total_rows * 100)
            .round(2)
            .reset_index()
            .rename(columns={"index": "column", 0: "missing_pct"})
            .sort_values("missing_pct", ascending=False)
        )

    duplicate_rows = int(working.duplicated().sum()) if total_rows else 0

    categorical_cols = list(working.select_dtypes(include=["object", "string", "category"]).columns)
    if categorical_cols:
        null_categories = int(working[categorical_cols].isna().sum().sum())
    else:
        null_categories = 0

    basic_stats = working.describe(include="all").transpose().reset_index().rename(columns={"index": "column"})

    summary_lines = []
    state_col = next((c for c in working.columns if "state" in c.lower()), None)
    cat_col = next((c for c in working.columns if "category" in c.lower()), None)

    if total_rows:
        if cat_col:
            miss_cat = float((working[cat_col].isna().sum() / total_rows) * 100)
            summary_lines.append(f"Missing product categories: {miss_cat:.1f}%")
        if state_col:
            miss_state = float((working[state_col].isna().sum() / total_rows) * 100)
            summary_lines.append(f"Missing states: {miss_state:.1f}%")
    summary_lines.append(f"Duplicate rows: {duplicate_rows}")

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "missing_by_column_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "null_categories_count": null_categories,
        "basic_stats": basic_stats,
        "summary_lines": summary_lines,
    }
