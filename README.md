# Strategiz Agent
**Enterprise AI Analytics for Natural-Language Business Questions**

Strategiz Agent is an AI-powered business intelligence application built with **Python, Streamlit, Plotly, DuckDB, and OpenAI**.  
It allows users to ask analytics questions in plain English and automatically generates:

- SQL queries
- Query results
- Multi-chart dashboards
- KPI summaries
- Business insights
- Story-driven analysis
- Executive-style reporting

The goal is to deliver a copilot-like BI experience where business users can explore sales, customers, products, and revenue without writing SQL manually.

---

## Demo Video

- [Project Demo](https://github.com/user-attachments/assets/0535d794-1bc6-4fb9-b6b4-75c4e24a0e20)

---

## What This Project Solves

Most analytics tools require users to:

1. Understand database schema  
2. Write valid SQL  
3. Build charts manually  
4. Interpret results independently

Strategiz Agent removes that friction by turning one natural-language question into a full analysis workflow:

1. Understands user business intent  
2. Generates SQL grounded in available e-commerce tables  
3. Executes query safely in read-only mode  
4. Builds an interactive BI-style dashboard  
5. Produces narrative insights and recommendations

This makes it useful for:

- Founders and business teams
- Operations and growth managers
- Data analysts who want faster first-pass insights
- Portfolio/demo scenarios for AI + analytics products

---

## Core Features

### 1) Natural Language to SQL
Users can ask questions like:

> “Build a sales dashboard for delivered orders in 2017 with KPIs, category performance, customer distribution by state, and top customers.”

The query agent generates executable SQL with joins across:

- `orders`
- `customers`
- `order_items`
- `products`
- `payments`
- `sellers`

### 2) Read-Only Query Execution + Retry Logic
- Read-only query enforcement
- SQL safety checks
- Retry once with LLM correction if first SQL fails
- Error messaging for out-of-scope analytics requests

### 3) AI-Powered Multi-Chart Dashboard
Instead of showing one chart, the app generates a full dashboard layout with:

- KPI cards (Total Revenue, Orders, AOV, Customers)
- Trend charts (monthly revenue/orders)
- Category and distribution visuals
- Customer and geographic views

Plotly interactivity includes:

- Zoom
- Pan
- Box select
- Reset axes

### 4) Data Quality Monitoring
- Missing values by column (%)
- Duplicate count
- Null category checks
- Basic descriptive statistics

### 5) AI Insight + Story + Executive Report
For each query, the app generates:

- **Business Insight**
- **Story of the Data**
- **Executive Report**

### 6) Chart Explanation On Demand
Users can click **“Explain this chart”** under each visualization to receive chart-level interpretation.

### 7) Conversation-Style UX
Chat-like flow allows follow-up questions in the same context.

### 8) Full Report Export
Users can download:

- Query results as CSV
- Full narrative report as PDF

---

## Project Architecture

```text
strategiz-agent/
│
├── app.py                          # Streamlit orchestration + UI + chat experience
├── config.py                       # Central configuration (model, DB path, limits)
├── requirements.txt
│
├── agents/
│   ├── query_agent.py              # NL -> SQL generation + SQL retry/correction path
│   ├── insight_agent.py            # Business insight + chart explanation + story generation
│   └── report_agent.py             # Executive report generation
│
├── analytics/
│   ├── data_quality.py             # Missing/duplicate/stat profiling
│   └── forecast.py                 # Revenue forecasting helper
│
├── dashboard/
│   └── charts.py                   # Auto dashboard chart creation logic (Plotly)
│
├── database/
│   └── load_data.py                # Data loading/prep script(s)
│
├── data/                           # Olist e-commerce CSV data sources
│
├── api/                            # Experimental FastAPI scaffold
└── frontend/                       # Experimental React scaffold
```

Tech Stack

    Frontend/UI: Streamlit
    Visualization: Plotly
    LLM: OpenAI API
    Database: DuckDB
    Data: Olist e-commerce dataset
    Language: Python 3.x

Dataset Context
The dashboard supports e-commerce analytics with fields such as:

    customer_state
    product_category_name
    order_purchase_timestamp
    payment_value
    customer_id
    order_id

Common business questions supported:

    Sales trend analysis
    Product/category performance
    Customer segmentation
    Geographic distribution
    Revenue pattern analysis

Local Setup

    1) Clone
    git clone https://github.com/shanmathy-ravisankaran/strategiz-agent.git
    cd strategiz-agent
    
    2) Create virtual environment
    python -m venv .venv
    Windows:
    .venv\Scripts\activate
    
    3) Install dependencies
    python -m pip install -r requirements.txt
    
    4) Configure environment
    Create .env in project root:
    env
    OPENAI_API_KEY=your_openai_api_key_here
    
    5) Run app
    python -m streamlit run app.py

Reliability & Guardrails

    Read-only query execution policy
    Blocking of risky SQL patterns
    Flat-table SQL expectation for dashboard compatibility
    SQL retry path after failed execution
    Out-of-scope handling for unrelated questions

UI/UX Highlights

    Dark modern BI-style layout
    KPI cards first, charts second, reports below
    Collapsible analysis sections
    Interactive Plotly modebar
    Conversation history for iterative analytics

Future Enhancements

    Stronger semantic schema grounding
    SQL parser-level validation
    Richer filter persistence
    Branded multi-page report export
    Production auth and user/session management
    Full React + FastAPI deployment mode

Security Notes

    .env is local-only and should never be committed
    Rotate API keys immediately if exposed
    Prefer managed secret storage for production

License

    This project is licensed under the MIT License.

<!-- auto-update 2026-08-23 16:18:01 -->

<!-- auto-update 2026-08-25 16:18:01 -->

<!-- auto-update 2026-08-27 16:18:01 -->

<!-- auto-update 2026-08-29 16:18:01 -->

<!-- auto-update 2026-08-31 16:18:01 -->

<!-- auto-update 2026-09-02 16:18:01 -->

<!-- auto-update 2026-09-04 16:18:01 -->
