import pandas as pd
import plotly.graph_objects as go


def _find_column(columns, keywords):
    lowered = {col: col.lower() for col in columns}
    for col, low in lowered.items():
        if any(k in low for k in keywords):
            return col
    return None


def forecast_revenue(df: pd.DataFrame):
    empty_result = {"forecast_values": [], "figure": None, "error": None}
    if df is None or df.empty:
        return empty_result

    date_col = _find_column(df.columns, ["order_date", "purchase", "timestamp", "date", "time"])
    revenue_col = _find_column(df.columns, ["revenue", "payment_value", "price", "sales", "amount", "total"])
    if date_col is None or revenue_col is None:
        return {**empty_result, "error": "Forecast requires a date column and a revenue column."}

    working = df[[date_col, revenue_col]].copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working[revenue_col] = pd.to_numeric(working[revenue_col], errors="coerce")
    working = working.dropna()
    if working.empty:
        return {**empty_result, "error": "Insufficient clean data for forecasting."}

    monthly = (
        working.assign(month=working[date_col].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)[revenue_col]
        .sum()
        .rename(columns={revenue_col: "revenue"})
        .sort_values("month")
    )

    if len(monthly) < 3:
        return {**empty_result, "error": "Need at least 3 monthly points for forecasting."}

    ExponentialSmoothing = None
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ExponentialSmoothing

        ExponentialSmoothing = _ExponentialSmoothing
    except Exception:
        ExponentialSmoothing = None

    series = monthly.set_index("month")["revenue"].asfreq("MS")
    try:
        if ExponentialSmoothing is not None:
            model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit(optimized=True)
        else:
            fit = None
    except Exception:
        # Fallback to simple trend if model fit fails.
        fit = None

    future_index = pd.date_range(start=series.index.max() + pd.offsets.MonthBegin(1), periods=3, freq="MS")
    if fit is not None:
        forecast = fit.forecast(3)
    else:
        step = (series.iloc[-1] - series.iloc[0]) / max(len(series) - 1, 1)
        forecast = pd.Series(
            [max(series.iloc[-1] + step * i, 0.0) for i in range(1, 4)],
            index=future_index,
        )

    forecast_df = pd.DataFrame({"month": future_index, "revenue": forecast.values})

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["revenue"],
            mode="lines+markers",
            name="Historical Revenue",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["month"],
            y=forecast_df["revenue"],
            mode="lines+markers",
            name="Forecast Revenue (3M)",
            line=dict(color="#d62728", dash="dash"),
        )
    )
    fig.update_layout(title="Revenue Trend and Forecast", template="plotly_white", height=360)

    return {
        "forecast_values": forecast_df.to_dict("records"),
        "figure": fig,
        "error": None,
    }
