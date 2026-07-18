import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


BR_STATE_COORDS = {
    "AC": (-8.77, -70.55),
    "AL": (-9.62, -36.82),
    "AM": (-3.47, -65.10),
    "AP": (1.41, -51.77),
    "BA": (-12.96, -38.51),
    "CE": (-3.72, -38.54),
    "DF": (-15.79, -47.88),
    "ES": (-19.19, -40.34),
    "GO": (-16.67, -49.25),
    "MA": (-2.55, -44.30),
    "MG": (-19.92, -43.94),
    "MS": (-20.45, -54.61),
    "MT": (-15.60, -56.10),
    "PA": (-1.46, -48.49),
    "PB": (-7.12, -34.88),
    "PE": (-8.05, -34.88),
    "PI": (-5.09, -42.80),
    "PR": (-25.42, -49.27),
    "RJ": (-22.90, -43.17),
    "RN": (-5.80, -35.21),
    "RO": (-8.76, -63.90),
    "RR": (2.82, -60.67),
    "RS": (-30.03, -51.23),
    "SC": (-27.59, -48.55),
    "SE": (-10.91, -37.07),
    "SP": (-23.55, -46.63),
    "TO": (-10.25, -48.33),
}


def _col_by_keywords(columns, keywords):
    lowered = {col: col.lower() for col in columns}
    for col, low in lowered.items():
        if any(key in low for key in keywords):
            return col
    return None


def _detect_datetime_column(df: pd.DataFrame):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            low_name = col.lower()
            if not any(token in low_name for token in ["date", "time", "timestamp", "dt"]):
                continue
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            except TypeError:
                parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.8 and parsed.notna().sum() > 1:
                df[col] = parsed
                return col
    return None


def _clean_for_visualization(df: pd.DataFrame):
    cleaned = df.copy()
    cleaned = cleaned.replace(["undefined", "None", "", "null", "NULL", "nan", "NaN"], pd.NA)
    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)
    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].astype("string").str.strip()
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.dropna()
    return cleaned


def _typed_columns(df: pd.DataFrame):
    working = df.copy()
    datetime_col = _detect_datetime_column(working)
    numeric_cols = [c for c in working.columns if pd.api.types.is_numeric_dtype(working[c])]
    categorical_cols = [c for c in working.columns if c not in numeric_cols and c != datetime_col]
    return working, numeric_cols, categorical_cols, datetime_col


def _indicator_figure(title: str, value: float, prefix: str = "", suffix: str = "", fmt: str = ",.2f"):
    fig = go.Figure(
        go.Indicator(
            mode="number",
            value=float(value),
            title={"text": title},
            number={"prefix": prefix, "suffix": suffix, "valueformat": fmt},
        )
    )
    fig.update_layout(height=160, margin=dict(l=12, r=12, t=46, b=12), template="plotly_dark")
    return fig


def _build_supporting_charts(df: pd.DataFrame, datetime_col, numeric_cols, categorical_cols):
    supplemental = {}
    if datetime_col and numeric_cols:
        supplemental["support_trend_line"] = px.line(
            df.sort_values(datetime_col),
            x=datetime_col,
            y=numeric_cols[0],
            markers=True,
            title=f"{numeric_cols[0]} over Time",
            template="plotly_dark",
        )
    if len(numeric_cols) >= 2:
        supplemental["support_scatter"] = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
            template="plotly_dark",
        )
    if numeric_cols:
        supplemental["support_hist"] = px.histogram(
            df,
            x=numeric_cols[0],
            nbins=30,
            title=f"{numeric_cols[0]} Distribution",
            template="plotly_dark",
        )
    if categorical_cols and numeric_cols:
        grouped = (
            df.groupby(categorical_cols[0], as_index=False)[numeric_cols[0]]
            .sum()
            .sort_values(numeric_cols[0], ascending=False)
            .head(10)
        )
        supplemental["support_bar"] = px.bar(
            grouped,
            x=categorical_cols[0],
            y=numeric_cols[0],
            title=f"{numeric_cols[0]} by {categorical_cols[0]}",
            template="plotly_dark",
        )
    return supplemental


def create_chart(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    df = _clean_for_visualization(df)
    if df.empty:
        return None

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime_col = _detect_datetime_column(df)
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != datetime_col]

    if datetime_col and numeric_cols:
        y_col = numeric_cols[0]
        fig = px.line(df.sort_values(datetime_col), x=datetime_col, y=y_col, markers=True)
        fig.update_layout(title=f"{y_col} over time", template="plotly_dark")
        return fig

    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        if df[cat_col].nunique() <= 8 and len(df) <= 20:
            fig = px.pie(df, names=cat_col, values=num_col, title=f"{num_col} by {cat_col}", template="plotly_dark")
            return fig
        fig = px.bar(df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}", template="plotly_dark")
        return fig

    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="Relationship view", template="plotly_dark")
        return fig

    if len(numeric_cols) == 1:
        fig = px.histogram(df, x=numeric_cols[0], nbins=30, title=f"Distribution of {numeric_cols[0]}", template="plotly_dark")
        return fig

    value_col = df.columns[0]
    counts = df[value_col].astype(str).value_counts().head(20).reset_index()
    counts.columns = [value_col, "count"]
    fig = px.bar(counts, x=value_col, y="count", title=f"Top values in {value_col}", template="plotly_dark")
    return fig


def _is_forecast_question(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in ["forecast", "predict", "future", "projection", "next month", "next quarter", "next year"])


def create_dashboard(data: pd.DataFrame, question: str | None = None):
    empty_payload = {
        "single_chart": None,
        "kpis": [],
        "charts": [],
        "sections": {},
        "layout_rows": [],
        "clean_data": pd.DataFrame(),
    }
    if data is None or data.empty:
        return empty_payload

    df = _clean_for_visualization(data)
    if df.empty:
        return empty_payload

    df, numeric_cols, categorical_cols, datetime_col = _typed_columns(df)
    is_forecast = _is_forecast_question(question or "")

    revenue_col = _col_by_keywords(df.columns, ["revenue", "payment_value", "price", "sales", "amount", "total"])
    orders_col = _col_by_keywords(df.columns, ["order_id", "orders"])
    category_col = _col_by_keywords(df.columns, ["product_category", "category"])
    payment_col = _col_by_keywords(df.columns, ["payment_type", "payment_method", "payment"])
    state_col = _col_by_keywords(df.columns, ["customer_state", "state", "uf"])
    customer_col = _col_by_keywords(df.columns, ["customer_id", "customer_unique_id", "customer"])

    if revenue_col is None and numeric_cols:
        revenue_col = numeric_cols[0]

    total_orders = float(df[orders_col].nunique()) if orders_col else float(len(df))
    total_revenue = float(df[revenue_col].sum()) if revenue_col else 0.0
    avg_order_value = total_revenue / total_orders if total_orders else 0.0
    total_customers = float(df[customer_col].nunique()) if customer_col else float(len(df))

    sections = {
        "kpi_total_orders": _indicator_figure("Total Orders", total_orders, fmt=",.0f"),
        "kpi_total_revenue": _indicator_figure("Total Revenue", total_revenue, prefix="$"),
        "kpi_avg_order_value": _indicator_figure("Average Order Value", avg_order_value, prefix="$"),
        "kpi_total_customers": _indicator_figure("Total Customers", total_customers, fmt=",.0f"),
    }

    # 5) Geographic Analytics - customer distribution by state (choropleth).
    if state_col:
        state_customers = (
            df.groupby(state_col, as_index=False)
            .agg(customers=(customer_col, "nunique") if customer_col else (state_col, "count"))
            .sort_values("customers", ascending=False)
        )
        state_customers[state_col] = state_customers[state_col].astype(str).str.upper().str.strip()
        if not state_customers.empty:
            try:
                geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
                br_state_name = {
                    "AC": "Acre",
                    "AL": "Alagoas",
                    "AP": "Amapa",
                    "AM": "Amazonas",
                    "BA": "Bahia",
                    "CE": "Ceara",
                    "DF": "Distrito Federal",
                    "ES": "Espirito Santo",
                    "GO": "Goias",
                    "MA": "Maranhao",
                    "MT": "Mato Grosso",
                    "MS": "Mato Grosso do Sul",
                    "MG": "Minas Gerais",
                    "PA": "Para",
                    "PB": "Paraiba",
                    "PR": "Parana",
                    "PE": "Pernambuco",
                    "PI": "Piaui",
                    "RJ": "Rio de Janeiro",
                    "RN": "Rio Grande do Norte",
                    "RS": "Rio Grande do Sul",
                    "RO": "Rondonia",
                    "RR": "Roraima",
                    "SC": "Santa Catarina",
                    "SP": "Sao Paulo",
                    "SE": "Sergipe",
                    "TO": "Tocantins",
                }
                choropleth_df = state_customers.copy()
                choropleth_df["state_name"] = choropleth_df[state_col].map(br_state_name)
                choropleth_df = choropleth_df.dropna(subset=["state_name"])
                sections["customer_state_choropleth"] = px.choropleth(
                    choropleth_df,
                    geojson=geojson_url,
                    locations="state_name",
                    featureidkey="properties.name",
                    color="customers",
                    hover_name="state_name",
                    hover_data={"customers": ":,.0f"},
                    title="Customer Distribution by State",
                    color_continuous_scale="Blues",
                    template="plotly_dark",
                )
                sections["customer_state_choropleth"].update_geos(fitbounds="locations", visible=False)
            except Exception:
                fallback = state_customers[state_customers[state_col].isin(BR_STATE_COORDS.keys())].copy()
                fallback["lat"] = fallback[state_col].map(lambda x: BR_STATE_COORDS[x][0])
                fallback["lon"] = fallback[state_col].map(lambda x: BR_STATE_COORDS[x][1])
                if not fallback.empty:
                    sections["customer_state_choropleth"] = px.scatter_geo(
                        fallback,
                        lat="lat",
                        lon="lon",
                        size="customers",
                        color="customers",
                        hover_name=state_col,
                        title="Customer Distribution by State",
                        template="plotly_dark",
                    )
                    sections["customer_state_choropleth"].update_geos(fitbounds="locations", visible=False)

    # 3) Product Analytics - treemap + donut.
    if category_col and revenue_col:
        cat_rev = (
            df.groupby(category_col, as_index=False)[revenue_col]
            .sum()
            .sort_values(revenue_col, ascending=False)
        )
        sections["category_treemap"] = px.treemap(
            cat_rev,
            path=[category_col],
            values=revenue_col,
            title="Revenue by Product Category",
            template="plotly_dark",
        )
        sections["category_share_donut"] = px.pie(
            cat_rev.head(10),
            names=category_col,
            values=revenue_col,
            hole=0.45,
            title="Category Share",
            template="plotly_dark",
        )

    # Customer analytics - top customers + customer revenue distribution.
    if customer_col and revenue_col:
        top_customers = (
            df.groupby(customer_col, as_index=False)[revenue_col]
            .sum()
            .sort_values(revenue_col, ascending=False)
            .head(10)
        )
        if not top_customers.empty:
            sections["top_customers"] = px.bar(
                top_customers.sort_values(revenue_col, ascending=True),
                x=revenue_col,
                y=customer_col,
                orientation="h",
                title="Top Customers by Revenue",
                template="plotly_dark",
            )

        customer_rev = df.groupby(customer_col, as_index=False)[revenue_col].sum()
        sections["customer_revenue_distribution"] = px.histogram(
            customer_rev,
            x=revenue_col,
            nbins=30,
            title="Customer Revenue Distribution",
            template="plotly_dark",
        )

    # 4) Additional customer segmentation bubble.
    if customer_col and orders_col and revenue_col:
        cust = (
            df.groupby(customer_col, as_index=False)
            .agg(
                orders=(orders_col, "nunique"),
                total_revenue=(revenue_col, "sum"),
            )
        )
        cust["avg_order_value"] = cust["total_revenue"] / cust["orders"].replace(0, np.nan)
        cust = cust.dropna().sort_values("total_revenue", ascending=False).head(1000)
        if not cust.empty:
            sections["customer_segmentation"] = px.scatter(
                cust,
                x="orders",
                y="total_revenue",
                size="avg_order_value",
                hover_name=customer_col,
                title="Customer Value Segmentation",
                template="plotly_dark",
            )
    elif len(numeric_cols) >= 2:
        sections["customer_segmentation"] = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title="Customer Value Segmentation",
            template="plotly_dark",
        )

    # 2) Trend analytics - revenue over time (line) + orders over time (area).
    if datetime_col:
        time_df = df.copy()
        time_df[datetime_col] = pd.to_datetime(time_df[datetime_col], errors="coerce")
        time_df = time_df.dropna(subset=[datetime_col])
        if not time_df.empty:
            time_df["month"] = time_df[datetime_col].dt.to_period("M").dt.to_timestamp()
            trend = None
            if revenue_col:
                trend = (
                    time_df.groupby("month", as_index=False)[revenue_col]
                    .sum()
                    .rename(columns={revenue_col: "value"})
                )
            else:
                trend = time_df.groupby("month", as_index=False).size().rename(columns={"size": "value"})
            sections["monthly_revenue_trend"] = px.line(
                trend.sort_values("month"),
                x="month",
                y="value",
                markers=True,
                title="Revenue over Time" if revenue_col else "Revenue/Orders over Time",
                template="plotly_dark",
            )

            if orders_col:
                order_trend = (
                    time_df.groupby("month", as_index=False)[orders_col]
                    .nunique()
                    .rename(columns={orders_col: "orders"})
                )
            else:
                order_trend = time_df.groupby("month", as_index=False).size().rename(columns={"size": "orders"})

            sections["monthly_orders_area"] = px.area(
                order_trend.sort_values("month"),
                x="month",
                y="orders",
                title="Orders over Time",
                template="plotly_dark",
            )

            # 5) Revenue Forecast (next 6 months) with simple linear trend.
            hist = trend.sort_values("month").copy()
            hist["t"] = np.arange(len(hist))
            if len(hist) >= 2:
                slope, intercept = np.polyfit(hist["t"], hist["value"], 1)
            else:
                slope, intercept = 0.0, float(hist["value"].iloc[0])
            last_t = int(hist["t"].iloc[-1])
            future_months = pd.date_range(
                start=hist["month"].iloc[-1] + pd.offsets.MonthBegin(1),
                periods=6,
                freq="MS",
            )
            forecast_vals = [max(slope * (last_t + i + 1) + intercept, 0.0) for i in range(6)]

            forecast_fig = go.Figure()
            forecast_fig.add_trace(
                go.Scatter(
                    x=hist["month"],
                    y=hist["value"],
                    mode="lines+markers",
                    name="Historical Revenue",
                    line=dict(color="#4dd0e1"),
                )
            )
            forecast_fig.add_trace(
                go.Scatter(
                    x=future_months,
                    y=forecast_vals,
                    mode="lines+markers",
                    name="Forecast Revenue",
                    line=dict(color="#ffb74d", dash="dash"),
                )
            )
            upper = [v * 1.1 for v in forecast_vals]
            lower = [max(v * 0.9, 0.0) for v in forecast_vals]
            forecast_fig.add_trace(
                go.Scatter(
                    x=list(future_months) + list(future_months[::-1]),
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor="rgba(255,183,77,0.15)",
                    line=dict(color="rgba(255,183,77,0)"),
                    hoverinfo="skip",
                    name="Forecast Band",
                )
            )
            forecast_fig.update_layout(title="Revenue Forecast", template="plotly_dark")
            sections["revenue_forecast"] = forecast_fig

    # 5) Geographic Analytics - revenue by region/state ranking.
    if state_col and revenue_col:
        state_revenue = (
            df.groupby(state_col, as_index=False)[revenue_col]
            .sum()
            .sort_values(revenue_col, ascending=False)
            .head(15)
        )
        if not state_revenue.empty:
            sections["revenue_by_region_bar"] = px.bar(
                state_revenue.sort_values(revenue_col, ascending=True),
                x=revenue_col,
                y=state_col,
                orientation="h",
                title="Revenue by Region",
                template="plotly_dark",
            )

    base_row = ["kpi_total_orders", "kpi_total_revenue", "kpi_avg_order_value", "kpi_total_customers"]
    layout_rows = [
        base_row,
        ["monthly_revenue_trend", "monthly_orders_area"],
        ["category_treemap", "category_share_donut"],
        ["top_customers", "customer_revenue_distribution"],
        ["customer_state_choropleth", "revenue_by_region_bar"],
    ]
    if is_forecast:
        layout_rows.append(["revenue_forecast"])

    visible_order = []
    for row in layout_rows[1:]:
        for key in row:
            if key not in visible_order:
                visible_order.append(key)
    for fallback_key in ["monthly_revenue_trend", "category_treemap", "top_customers", "customer_revenue_distribution", "customer_state_choropleth"]:
        if fallback_key not in visible_order:
            visible_order.append(fallback_key)

    charts = [sections[key] for key in visible_order if key in sections][:7]
    # Charts are capped to keep a professional dashboard density.
    kpis = [
        sections["kpi_total_orders"],
        sections["kpi_total_revenue"],
        sections["kpi_avg_order_value"],
        sections["kpi_total_customers"],
    ]

    if len(charts) < 2:
        supplemental = _build_supporting_charts(df, datetime_col, numeric_cols, categorical_cols)
        for key, fig in supplemental.items():
            if key not in sections:
                sections[key] = fig
                visible_order.append(key)
            if len([k for k in visible_order if k in sections]) >= 4:
                break
        layout_rows.extend(
            [
                ["support_trend_line", "support_scatter"],
                ["support_hist", "support_bar"],
            ]
        )
        charts = [sections[key] for key in visible_order if key in sections][:7]

    if len(charts) == 0:
        return {
            **empty_payload,
            "single_chart": create_chart(df),
            "kpis": kpis,
            "sections": sections,
            "layout_rows": layout_rows,
            "clean_data": df,
        }

    return {
        "single_chart": None,
        "kpis": kpis,
        "charts": charts,
        "sections": sections,
        "layout_rows": layout_rows,
        "clean_data": df,
    }
