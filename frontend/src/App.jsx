import { useMemo, useState } from 'react'
import Plot from 'react-plotly.js'

const API_BASE = 'http://127.0.0.1:8000'

const plotConfig = {
  displayModeBar: true,
  displaylogo: false,
  scrollZoom: true,
  responsive: true,
}

function ChartCard({ title, figure }) {
  if (!figure) return null
  return (
    <div className="card">
      <h3>{title}</h3>
      <Plot data={figure.data} layout={{ ...figure.layout, autosize: true }} config={plotConfig} style={{ width: '100%', height: '100%' }} useResizeHandler />
    </div>
  )
}

export default function App() {
  const [question, setQuestion] = useState('Show a complete business performance dashboard with total revenue, total orders, average order value, total customers, monthly revenue trend, monthly orders trend, top product categories, customer distribution by state, top customers by revenue, and revenue distribution.')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [payload, setPayload] = useState(null)

  const sectionTitle = useMemo(() => ({
    kpi_total_orders: 'Total Orders',
    kpi_total_revenue: 'Total Revenue',
    kpi_avg_order_value: 'Average Order Value',
    kpi_total_customers: 'Total Customers',
    monthly_revenue_trend: 'Revenue Trend',
    monthly_orders_area: 'Orders Trend',
    category_treemap: 'Category Revenue Treemap',
    category_share_donut: 'Category Share',
    top_customers: 'Top Customers',
    customer_revenue_distribution: 'Customer Revenue Distribution',
    customer_state_choropleth: 'Customers by State',
    revenue_by_region_bar: 'Revenue by Region',
    revenue_forecast: 'Revenue Forecast',
  }), [])

  const analyze = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(typeof data?.detail === 'string' ? data.detail : 'Request failed')
      if (!data.in_scope) {
        setPayload(null)
        setError(data.message)
      } else {
        setPayload(data)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const rows = payload?.dashboard?.layout_rows || []
  const sections = payload?.dashboard?.sections || {}

  return (
    <div className="page">
      <header>
        <h1>Strategiz Agent BI</h1>
        <p>Enterprise AI analytics with React + FastAPI</p>
      </header>

      <div className="query-box">
        <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a business question" />
        <button onClick={analyze} disabled={loading}>{loading ? 'Analyzing...' : 'Analyze'}</button>
      </div>

      {error && <div className="error">{error}</div>}

      {payload && (
        <>
          <details className="panel"><summary>Generated SQL</summary><pre>{payload.sql}</pre></details>

          {rows.map((row, idx) => {
            const visible = row.filter((key) => sections[key])
            if (!visible.length) return null
            return (
              <section className={`grid cols-${Math.min(visible.length, 4)}`} key={idx}>
                {visible.map((key) => <ChartCard key={key} title={sectionTitle[key] || key} figure={sections[key]} />)}
              </section>
            )
          })}

          <details className="panel"><summary>Business Insight</summary><p>{payload.insight}</p></details>
          <details className="panel"><summary>Story of the Data</summary><p>{payload.story}</p></details>
          <details className="panel"><summary>Executive Report</summary><pre>{payload.report}</pre></details>
          <details className="panel"><summary>Recommended Actions</summary><pre>{payload.recommended_actions}</pre></details>
          <details className="panel"><summary>Next Questions</summary><pre>{payload.next_questions}</pre></details>
        </>
      )}
    </div>
  )
}
