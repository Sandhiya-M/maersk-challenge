import { useEffect, useMemo, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const initialForm = {
  shipment_id: 'SHP-UI-001',
  booking_time: new Date().toISOString().slice(0, 16),
  port_of_loading: 'SGSIN',
  port_of_discharge: 'USLAX',
  cargo_type: 'Electronics',
  container_count: 12,
  weight_tons: 1450,
}

const metricLabels = {
  precision: 'Precision',
  recall: 'Recall',
  f1: 'F1 score',
  auc: 'ROC-AUC',
}

const cargoOptions = ['Apparel', 'Automotive Parts', 'Chemicals', 'Consumer goods', 'Electronics', 'Food & beverage', 'Furniture', 'Machinery', 'Pharmaceuticals', 'Raw Materials', 'Textiles']

const lowRiskPresets = [
  {
    id: 'jpyok-esvlc',
    label: 'JPYOK → ESVLC · 10% predicted risk',
    values: { shipment_id: 'SHP-03061', booking_time: '2025-06-18T20:00', port_of_loading: 'JPYOK', port_of_discharge: 'ESVLC', cargo_type: 'Electronics', container_count: 32, weight_tons: 1125.3 },
  },
  {
    id: 'sgsin-pkkar',
    label: 'SGSIN → PKKAR · 11% predicted risk',
    values: { shipment_id: 'SHP-04087', booking_time: '2025-03-17T17:00', port_of_loading: 'SGSIN', port_of_discharge: 'PKKAR', cargo_type: 'Pharmaceuticals', container_count: 1, weight_tons: 2752.03 },
  },
  {
    id: 'beanr-usnyc',
    label: 'BEANR → USNYC · 11% predicted risk',
    values: { shipment_id: 'SHP-04969', booking_time: '2024-09-07T09:00', port_of_loading: 'BEANR', port_of_discharge: 'USNYC', cargo_type: 'Furniture', container_count: 39, weight_tons: 578.45 },
  },
]

const formatPercent = (value, digits = 1) => `${(Number(value || 0) * 100).toFixed(digits)}%`
const formatScore = (value) => Number(value || 0).toFixed(3)
const parseJsonResponse = async (response) => {
  const text = await response.text()
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return { detail: text }
  }
}
const labelPort = (code, portMap = {}) => {
  if (!code) return 'Unknown port'
  const label = portMap[code]?.trim()
  if (label) return `${label} (${code})`
  return code
}

function Logo() {
  return (
    <svg viewBox="0 0 48 48" aria-hidden="true" className="logo-mark">
      <rect x="4" y="4" width="40" height="40" rx="10" fill="currentColor" />
      <path d="M14 34V15h4l8 16 8-16h4v19h-4V22l-8 16h-2L18 22v12h-4Z" fill="#ffffff" />
      <path d="M20 12h8l-3.5 7h-1z" fill="#f59e0b" opacity="0.95" />
    </svg>
  )
}

function ArrowIcon() {
  return <span className="arrow-icon" aria-hidden="true">→</span>
}

function App() {
  const [form, setForm] = useState(initialForm)
  const [prediction, setPrediction] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [metricsError, setMetricsError] = useState('')
  const [predictionError, setPredictionError] = useState('')
  const [loadingPrediction, setLoadingPrediction] = useState(false)
  const [loadingMetrics, setLoadingMetrics] = useState(true)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [predictionHistory, setPredictionHistory] = useState([])
  const [ports, setPorts] = useState([])
  const [selectedPreset, setSelectedPreset] = useState('')
  const [shipments, setShipments] = useState([])
  const [shipmentSearch, setShipmentSearch] = useState('')
  const [qualityReport, setQualityReport] = useState(null)
  const [routeQuery, setRouteQuery] = useState({ origin: 'SGSIN', destination: 'USLAX' })
  const [routeStats, setRouteStats] = useState(null)
  const [routeError, setRouteError] = useState('')
  const [pipelineFiles, setPipelineFiles] = useState({ shipments: null, ports: null, port_events: null })
  const [pipelineMessage, setPipelineMessage] = useState('No pipeline run in this session.')
  const [pipelineState, setPipelineState] = useState('idle')
  const [assistantMessages, setAssistantMessages] = useState([
    { role: 'assistant', content: 'Ask about shipment risk, route performance, or the latest operational signals.' },
  ])
  const [assistantInput, setAssistantInput] = useState('')
  const [assistantLoading, setAssistantLoading] = useState(false)
  const [activePage, setActivePage] = useState('predictor')

  const loadMetrics = async () => {
    try {
      setLoadingMetrics(true)
      setMetricsError('')
      const response = await fetch(`${API_BASE}/predict/metrics`)
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'Model metrics are unavailable.')
      setMetrics(payload)
      setLastUpdated(new Date())
    } catch (error) {
      setMetrics(null)
      setMetricsError(error.message || 'Unable to load model metrics.')
    } finally {
      setLoadingMetrics(false)
    }
  }

  const loadOperations = async () => {
    try {
      const [shipmentsResponse, qualityResponse] = await Promise.all([
        fetch(`${API_BASE}/shipments?offset=0&limit=5000`),
        fetch(`${API_BASE}/data-quality`),
      ])
      if (shipmentsResponse.ok) {
        const payload = await shipmentsResponse.json()
        const items = Array.isArray(payload) ? payload : payload.items || []
        setShipments(items)
      }
      if (qualityResponse.ok) {
        const payload = await qualityResponse.json()
        setQualityReport(payload.report || payload)
      }
    } catch {
      setShipments([])
      setQualityReport(null)
    }
  }

  useEffect(() => {
    loadMetrics()
    loadOperations()
  }, [])

  useEffect(() => {
    const loadPorts = async () => {
      try {
        const response = await fetch(`${API_BASE}/ports?offset=0&limit=100`)
        if (!response.ok) return
        const payload = await response.json()
        const items = Array.isArray(payload) ? payload : payload.items || []
        const normalized = items
          .filter((item) => item?.port_code)
          .map((item) => ({
            code: String(item.port_code).trim().toUpperCase(),
            name: String(item.port_name || item.port_code || '').trim(),
          }))
        setPorts(normalized)
      } catch {
        setPorts([])
      }
    }

    loadPorts()
  }, [])

  const handleChange = (event) => {
    const { name, value } = event.target
    setForm((current) => ({
      ...current,
      [name]: ['container_count', 'weight_tons'].includes(name) ? (value === '' ? '' : Number(value)) : value,
    }))
  }

  const handlePresetChange = (event) => {
    const presetId = event.target.value
    setSelectedPreset(presetId)
    const preset = lowRiskPresets.find((item) => item.id === presetId)
    if (!preset) return
    setForm(preset.values)
    setPrediction(null)
    setPredictionError('')
  }

  const handleRefresh = async () => {
    await Promise.all([loadMetrics(), loadOperations()])
  }

  const loadRouteStats = async (event) => {
    event?.preventDefault()
    try {
      setRouteError('')
      const response = await fetch(`${API_BASE}/routes/${encodeURIComponent(routeQuery.origin)}/${encodeURIComponent(routeQuery.destination)}/stats`)
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'Route statistics are unavailable.')
      setRouteStats(payload)
    } catch (error) {
      setRouteStats(null)
      setRouteError(error.message || 'Route statistics are unavailable.')
    }
  }

  const handlePipelineFile = (event) => {
    const { name, files } = event.target
    setPipelineFiles((current) => ({ ...current, [name]: files?.[0] || null }))
  }

  const runPipeline = async () => {
    const formData = new FormData()
    Object.entries(pipelineFiles).forEach(([name, file]) => {
      if (file) formData.append(name, file)
    })
    if ([...formData.keys()].length === 0) {
      setPipelineState('error')
      setPipelineMessage('Choose at least one CSV file before running the pipeline.')
      return
    }
    try {
      setPipelineState('running')
      setPipelineMessage('Uploading files and running quality checks…')
      const uploadResponse = await fetch(`${API_BASE}/pipeline/upload`, { method: 'POST', body: formData })
      const uploadPayload = await uploadResponse.json()
      if (!uploadResponse.ok) throw new Error(uploadPayload.detail || 'Upload failed.')
      const runResponse = await fetch(`${API_BASE}/pipeline/run`, { method: 'POST' })
      const runPayload = await runResponse.json()
      if (!runResponse.ok) throw new Error(runPayload.detail || 'Pipeline failed.')
      setPipelineState('success')
      setPipelineMessage(`Pipeline completed. ${Object.keys(uploadPayload.saved || {}).length} file(s) refreshed.`)
      setQualityReport(runPayload.report || null)
      await loadOperations()
    } catch (error) {
      setPipelineState('error')
      setPipelineMessage(error.message || 'Pipeline failed.')
    }
  }

  const handleAssistantChat = async (event) => {
    event.preventDefault()
    const query = assistantInput.trim()
    if (!query) return

    setAssistantMessages((current) => [...current, { role: 'user', content: query }])
    setAssistantInput('')
    setAssistantLoading(true)

    try {
      const response = await fetch(`${API_BASE}/assistant/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, max_tool_calls: 5 }),
      })
      const payload = await parseJsonResponse(response)
      if (!response.ok) {
        throw new Error(payload?.detail || payload?.message || 'The assistant could not answer that request.')
      }
      setAssistantMessages((current) => [
        ...current,
        { role: 'assistant', content: payload?.response || 'No answer was returned.', tool_calls: payload?.tool_calls || [] },
      ])
    } catch (error) {
      setAssistantMessages((current) => [
        ...current,
        { role: 'assistant', content: error.message || 'Assistant request failed.', is_error: true },
      ])
    } finally {
      setAssistantLoading(false)
    }
  }

  const handlePredict = async (event) => {
    event.preventDefault()
    try {
      setLoadingPrediction(true)
      setPredictionError('')
      const response = await fetch(`${API_BASE}/predict-delay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'Prediction could not be completed.')
      setPrediction(payload)
      setPredictionHistory((current) => [payload, ...current.filter((item) => item.shipment_id !== payload.shipment_id)].slice(0, 4))
    } catch (error) {
      setPrediction(null)
      setPredictionError(error.message || 'Prediction could not be completed.')
    } finally {
      setLoadingPrediction(false)
    }
  }

  const risk = useMemo(() => {
    if (!prediction) return null
    const probability = Number(prediction.probability_gt_24h_late || 0)
    const highRisk = prediction.is_more_than_24h_late || probability >= Number(metrics?.threshold || 0.5)
    return {
      highRisk,
      probability,
      label: highRisk ? 'High delay risk' : 'Within threshold',
      description: highRisk
        ? 'Plan an operational review before the shipment is released.'
        : 'Current booking inputs are below the model’s delay threshold.',
    }
  }, [metrics?.threshold, prediction])

  const comparisonRows = metrics ? [
    { label: 'Selected model', value: metrics.auc, tone: 'blue' },
    { label: 'Benchmark', value: metrics.baseline_auc, tone: 'slate' },
    { label: 'Challenger model', value: metrics.model_b_auc, tone: 'mint' },
  ] : []

  const portMap = useMemo(() => Object.fromEntries(ports.map((port) => [port.code, port.name])), [ports])
  const portOptions = useMemo(() => {
    const base = new Map()
    ports.forEach((port) => {
      if (port?.code) base.set(port.code, port.name || port.code)
    })
    ;['SGSIN', 'USLAX', 'JPYOK', 'BEANR', 'PKKAR', 'ESVLC', 'USNYC'].forEach((code) => {
      if (!base.has(code)) base.set(code, code)
    })
    return [...base.entries()].map(([code, name]) => ({ value: code, label: `${name} (${code})` }))
  }, [ports])
  const calibration = Array.isArray(metrics?.calibration_table) ? metrics.calibration_table : []
  const normalizedShipmentSearch = shipmentSearch.trim().toLowerCase()
  const filteredShipments = shipments.filter((shipment) => {
    if (!normalizedShipmentSearch) return true
    const searchableValues = [
      shipment.shipment_id,
      shipment.origin_port,
      shipment.destination_port,
      shipment.port_of_loading,
      shipment.port_of_discharge,
      shipment.cargo_type,
      shipment.status,
      shipment.booking_time,
      shipment.booking_date,
      labelPort(shipment.port_of_loading || shipment.origin_port, portMap),
      labelPort(shipment.port_of_discharge || shipment.destination_port, portMap),
    ]
    return searchableValues.some((value) => String(value ?? '').toLowerCase().includes(normalizedShipmentSearch))
  })
  const qualityItems = Object.entries(qualityReport?.raw_quality || qualityReport?.curated_quality || {})
  const routeShipments = Array.isArray(routeStats?.shipments) ? routeStats.shipments : Array.isArray(routeStats?.scheduled_shipments) ? routeStats.scheduled_shipments : []

  const renderPage = () => {
    const baseLayout = (
      <main className="page-shell">
        <section className="hero" aria-labelledby="page-title">
          <div className="hero-copy">
            <p className="eyebrow">Predictive operations</p>
            <h1 id="page-title">See delay risk before it disrupts the route.</h1>
            <p className="hero-text">Score a new booking in seconds and review the validation signals behind each decision.</p>
          </div>
          <div className="model-status">
            <span className={`status-dot ${metrics ? 'online' : 'offline'}`} />
            <div>
              <span>Model status</span>
              <strong>{metrics ? 'Ready for scoring' : 'Metrics unavailable'}</strong>
            </div>
            <small>{metrics?.model || 'Train a model to begin'}</small>
          </div>
        </section>
      </main>
    )

    switch (activePage) {
      case 'model-health':
        return (
          <main className="page-shell">
            <section className="page-hero">
              <div>
                <p className="section-kicker">Model governance</p>
                <h2>Validation at a glance</h2>
                <p>Test-set performance for the current production model.</p>
              </div>
              <div className="updated-label">
                <span>Last refreshed</span>
                <strong>{lastUpdated ? lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'}</strong>
              </div>
            </section>

            {metricsError ? (
              <div className="metrics-message">{metricsError}</div>
            ) : loadingMetrics && !metrics ? (
              <div className="metrics-message">Loading model validation data…</div>
            ) : metrics && (
              <>
                <div className="kpi-grid">
                  {Object.entries(metricLabels).map(([key, label]) => (
                    <article className="kpi-card" key={key}>
                      <span>{label}</span>
                      <strong>{formatScore(metrics[key])}</strong>
                      <div className="score-track"><i style={{ width: formatPercent(metrics[key]) }} /></div>
                    </article>
                  ))}
                  <article className="kpi-card threshold-card">
                    <span>Decision threshold</span>
                    <strong>{formatPercent(metrics.threshold, 0)}</strong>
                    <small>Probability required to flag a shipment</small>
                  </article>
                </div>

                <div className="analytics-grid">
                  <article className="analytics-card comparison-card">
                    <div className="card-title">
                      <div>
                        <span className="card-kicker">Model comparison</span>
                        <h3>Discrimination performance</h3>
                      </div>
                      <span className="metric-chip">ROC-AUC</span>
                    </div>
                    <div className="comparison-list">
                      {comparisonRows.map((row) => (
                        <div className="comparison-row" key={row.label}>
                          <div><span className={`legend-dot ${row.tone}`} />{row.label}</div>
                          <div className="comparison-bar"><i className={row.tone} style={{ width: formatPercent(row.value) }} /></div>
                          <strong>{formatScore(row.value)}</strong>
                        </div>
                      ))}
                    </div>
                    <p className="card-note">A score of 0.500 represents random discrimination. Higher is better.</p>
                  </article>

                  <article className="analytics-card confusion-card">
                    <div className="card-title">
                      <div>
                        <span className="card-kicker">Test set</span>
                        <h3>Decision outcomes</h3>
                      </div>
                      <span className="metric-chip">Threshold {formatPercent(metrics.threshold, 0)}</span>
                    </div>
                    <div className="confusion-grid">
                      <div className="axis-label blank" />
                      <div className="axis-label">Predicted on-time</div>
                      <div className="axis-label">Predicted late</div>
                      <div className="axis-label vertical">Actual on-time</div>
                      <div className="confusion-cell"><strong>{metrics.confusion_matrix?.[0] ?? 0}</strong><span>True negative</span></div>
                      <div className="confusion-cell alert"><strong>{metrics.confusion_matrix?.[1] ?? 0}</strong><span>False positive</span></div>
                      <div className="axis-label vertical">Actual late</div>
                      <div className="confusion-cell alert"><strong>{metrics.confusion_matrix?.[2] ?? 0}</strong><span>False negative</span></div>
                      <div className="confusion-cell success"><strong>{metrics.confusion_matrix?.[3] ?? 0}</strong><span>True positive</span></div>
                    </div>
                  </article>
                </div>

                <article className="analytics-card calibration-card">
                  <div className="card-title">
                    <div>
                      <span className="card-kicker">Reliability check</span>
                      <h3>Predicted vs. observed delay rate</h3>
                    </div>
                    <span className="metric-chip">{metrics.calibration_table?.reduce((sum, item) => sum + Number(item.count || 0), 0) || 0} records</span>
                  </div>
                  <div className="calibration-chart">
                    {calibration.map((bucket) => (
                      <div className="calibration-bar" key={bucket.bin}>
                        <div className="bar-columns">
                          <i className="predicted" style={{ height: formatPercent(bucket.avg_predicted_probability) }} title={`Predicted ${formatPercent(bucket.avg_predicted_probability)}`} />
                          <i className="observed" style={{ height: formatPercent(bucket.observed_rate) }} title={`Observed ${formatPercent(bucket.observed_rate)}`} />
                        </div>
                        <span>{bucket.bin.replace(/[\[\]]/g, '')}</span>
                      </div>
                    ))}
                  </div>
                  <div className="chart-legend"><span><i className="legend-dot blue" />Model prediction</span><span><i className="legend-dot mint" />Observed outcome</span></div>
                </article>
              </>
            )}
          </main>
        )

      case 'assistant':
        return (
          <main className="page-shell">
            <section className="page-hero">
              <div>
                <p className="section-kicker">Decision support</p>
                <h2>Analyst assistant</h2>
                <p>Ask for route trends, risk reviews, or shipment summaries using the same data and tool-calling flow as the CLI assistant.</p>
              </div>
            </section>

            <div className="assistant-panel">
              <div className="assistant-messages" aria-live="polite">
                {assistantMessages.map((message, index) => (
                  <div key={`${message.role}-${index}`} className={`assistant-message ${message.role} ${message.is_error ? 'error' : ''}`}>
                    <span className="assistant-role">{message.role === 'user' ? 'You' : 'Assistant'}</span>
                    <p>{message.content}</p>
                    {message.tool_calls?.length > 0 && (
                      <small>Tools used: {message.tool_calls.map((tool) => tool.name).join(', ')}</small>
                    )}
                  </div>
                ))}
              </div>

              <form className="assistant-form" onSubmit={handleAssistantChat}>
                <input
                  value={assistantInput}
                  onChange={(event) => setAssistantInput(event.target.value)}
                  placeholder="Ask: Which routes had the worst delay this quarter?"
                  aria-label="Assistant query"
                  disabled={assistantLoading}
                />
                <button type="submit" disabled={assistantLoading || !assistantInput.trim()}>
                  {assistantLoading ? 'Thinking…' : 'Ask assistant'}
                </button>
              </form>
            </div>
          </main>
        )

      case 'operations':
        return (
          <main className="page-shell">
            <section className="page-hero">
              <div>
                <p className="section-kicker">Operational control</p>
                <h2>Network activity and data readiness</h2>
                <p>Search curated shipments, investigate a route, and review the latest quality gate.</p>
              </div>
              <span className="metric-chip">{shipments.length} shipments loaded</span>
            </section>

            <div className="operations-grid">
              <article className="operations-card shipment-card">
                <div className="card-title">
                  <div>
                    <span className="card-kicker">Curated registry</span>
                    <h3>Shipment details</h3>
                  </div>
                  <input className="search-input" value={shipmentSearch} onChange={(event) => setShipmentSearch(event.target.value)} placeholder="Search ID, route, cargo…" aria-label="Search shipments" />
                </div>
                <div className="shipment-table-wrap">
                  <table className="shipment-table">
                    <thead><tr><th>Shipment</th><th>Route</th><th>Cargo</th><th>Status</th></tr></thead>
                    <tbody>
                      {filteredShipments.slice(0, 50).map((shipment, index) => (
                        <tr key={shipment.shipment_id || index}>
                          <td><strong>{shipment.shipment_id || '—'}</strong><small>{shipment.booking_time || shipment.booking_date || 'No booking date'}</small></td>
                          <td>{labelPort(shipment.port_of_loading || shipment.origin_port, portMap)} <span>→</span> {labelPort(shipment.port_of_discharge || shipment.destination_port, portMap)}</td>
                          <td>{shipment.cargo_type || '—'}</td>
                          <td><span className={`status-pill ${String(shipment.status || '').toLowerCase()}`}>{shipment.status || 'UNKNOWN'}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {!filteredShipments.length && <p className="empty-row">No shipments match this search.</p>}
                </div>
              </article>

              <article className="operations-card route-card">
                <div className="card-title">
                  <div><span className="card-kicker">Route intelligence</span><h3>Route performance</h3></div>
                </div>
                <form className="route-form" onSubmit={loadRouteStats}>
                  <div className="field"><label htmlFor="route-origin">Origin</label><select id="route-origin" value={routeQuery.origin} onChange={(event) => setRouteQuery((current) => ({ ...current, origin: event.target.value }))}>{portOptions.map((port) => <option key={port.value} value={port.value}>{port.label}</option>)}</select></div>
                  <div className="field"><label htmlFor="route-destination">Destination</label><select id="route-destination" value={routeQuery.destination} onChange={(event) => setRouteQuery((current) => ({ ...current, destination: event.target.value }))}>{portOptions.map((port) => <option key={port.value} value={port.value}>{port.label}</option>)}</select></div>
                  <button className="route-button" type="submit">View route</button>
                </form>
                {routeError && <p className="form-error">{routeError}</p>}
                {routeStats ? (
                  <div className="route-results">
                    <div><span>Average delay</span><strong>{Number(routeStats.average_delay_hours || 0).toFixed(1)}h</strong></div>
                    <div><span>On-time rate</span><strong>{Number(routeStats.on_time_rate_percent || 0).toFixed(1)}%</strong></div>
                    <div><span>Shipments</span><strong>{routeStats.shipment_count || 0}</strong></div>
                    {routeShipments.length > 0 && (
                      <div className="route-shipments">
                        <h4>Route shipment details</h4>
                        <div className="shipment-table-wrap route-shipment-table-wrap">
                          <table className="shipment-table route-shipment-table">
                            <thead>
                              <tr>
                                <th>Shipment</th>
                                <th>Cargo</th>
                                <th>Status</th>
                                <th>Delay (h)</th>
                                <th>Departure</th>
                                <th>Arrival</th>
                              </tr>
                            </thead>
                            <tbody>
                              {routeShipments.map((shipment, index) => (
                                <tr key={shipment.shipment_id || `${shipment.origin || routeQuery.origin}-${shipment.destination || routeQuery.destination}-${index}`}>
                                  <td>
                                    <strong>{shipment.shipment_id || '—'}</strong>
                                    <small>{shipment.port_of_loading || shipment.origin_port || routeQuery.origin} → {shipment.port_of_discharge || shipment.destination_port || routeQuery.destination}</small>
                                  </td>
                                  <td>{shipment.cargo_type || '—'}</td>
                                  <td><span className={`status-pill ${String(shipment.status || '').toLowerCase()}`}>{shipment.status || 'UNKNOWN'}</span></td>
                                  <td>{Number(shipment.delay_hours || 0) > 0 ? `${Number(shipment.delay_hours).toFixed(1)}h` : '0.0h'}</td>
                                  <td>{shipment.planned_departure || shipment.actual_departure || '—'}</td>
                                  <td>{shipment.planned_arrival || shipment.actual_arrival || '—'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                ) : <p className="card-note">Choose a route to see its reliability and planned shipment volume.</p>}
              </article>

              <article className="operations-card quality-card">
                <div className="card-title"><div><span className="card-kicker">Quality gate</span><h3>Data quality</h3></div><span className={`quality-state ${qualityReport?.status === 'ok' ? 'good' : ''}`}>{qualityReport?.status === 'ok' ? 'Ready' : 'Not run'}</span></div>
                {qualityItems.length ? <div className="quality-list">{qualityItems.slice(0, 4).map(([name, summary]) => <div key={name}><span>{name.replaceAll('_', ' ')}</span><strong>{summary?.passed === false ? 'Needs review' : 'Passed'}</strong><small>{summary?.total_rows ?? 0} rows checked</small></div>)}</div> : <p className="card-note">Run the pipeline to generate validation checks and quarantine counts.</p>}
              </article>
            </div>
          </main>
        )

      case 'pipeline':
        return (
          <main className="page-shell">
            <section className="page-hero">
              <div>
                <p className="section-kicker">Raw-to-curated processing</p>
                <h2>Pipeline and quality checks</h2>
                <p>Upload source files, run validation, and refresh the operational data layer.</p>
              </div>
            </section>

            <div className="pipeline-card">
              <div className="file-grid">
                <label className="file-input"><span>Shipments CSV</span><input name="shipments" type="file" accept=".csv" onChange={handlePipelineFile} /><small>{pipelineFiles.shipments?.name || 'Choose file'}</small></label>
                <label className="file-input"><span>Ports CSV</span><input name="ports" type="file" accept=".csv" onChange={handlePipelineFile} /><small>{pipelineFiles.ports?.name || 'Choose file'}</small></label>
                <label className="file-input"><span>Port events CSV</span><input name="port_events" type="file" accept=".csv" onChange={handlePipelineFile} /><small>{pipelineFiles.port_events?.name || 'Choose file'}</small></label>
              </div>
              <div className={`pipeline-status ${pipelineState}`}><div><span>Pipeline status</span><strong>{pipelineState === 'running' ? 'Running checks' : pipelineState === 'success' ? 'Completed' : pipelineState === 'error' ? 'Action needed' : 'Ready'}</strong></div><p>{pipelineMessage}</p><button type="button" className="score-button" onClick={runPipeline} disabled={pipelineState === 'running'}>{pipelineState === 'running' ? 'Processing…' : 'Run pipeline'}<ArrowIcon /></button></div>
            </div>
          </main>
        )

      default:
        return (
          <main className="page-shell">
            <section className="hero" aria-labelledby="page-title">
              <div className="hero-copy">
                <p className="eyebrow">Predictive operations</p>
                <h1 id="page-title">See delay risk before it disrupts the route.</h1>
                <p className="hero-text">Score a new booking in seconds and review the validation signals behind each decision.</p>
              </div>
              <div className="model-status">
                <span className={`status-dot ${metrics ? 'online' : 'offline'}`} />
                <div>
                  <span>Model status</span>
                  <strong>{metrics ? 'Ready for scoring' : 'Metrics unavailable'}</strong>
                </div>
                <small>{metrics?.model || 'Train a model to begin'}</small>
              </div>
            </section>

            <section className="workspace" id="predictor">
              <div className="prediction-panel">
                <div className="section-heading">
                  <div>
                    <p className="section-kicker">Shipment assessment</p>
                    <h2>Delay risk predictor</h2>
                  </div>
                  <span className="step-label">01 / Score booking</span>
                </div>

                <form onSubmit={handlePredict} className="predict-form">
                  <div className="field field-wide preset-field">
                    <label htmlFor="low-risk-preset">Load a verified low-risk example</label>
                    <select id="low-risk-preset" value={selectedPreset} onChange={handlePresetChange}>
                      <option value="">Choose a sample booking</option>
                      {lowRiskPresets.map((preset) => <option key={preset.id} value={preset.id}>{preset.label}</option>)}
                    </select>
                    <small>Examples score below the current {formatPercent(metrics?.threshold, 0)} alert threshold.</small>
                  </div>
                  <div className="field field-wide">
                    <label htmlFor="shipment_id">Shipment reference</label>
                    <input id="shipment_id" name="shipment_id" value={form.shipment_id} onChange={handleChange} required />
                  </div>
                  <div className="field">
                    <label htmlFor="booking_time">Booking time</label>
                    <input id="booking_time" type="datetime-local" name="booking_time" value={form.booking_time} onChange={handleChange} required />
                  </div>
                  <div className="field">
                    <label htmlFor="cargo_type">Cargo type</label>
                    <select id="cargo_type" name="cargo_type" value={form.cargo_type} onChange={handleChange} required>
                      {!cargoOptions.includes(form.cargo_type) && <option value={form.cargo_type}>{form.cargo_type}</option>}
                      {cargoOptions.map((cargo) => <option key={cargo} value={cargo}>{cargo}</option>)}
                    </select>
                  </div>
                  <div className="route-fields field-wide">
                    <div className="field">
                      <label htmlFor="port_of_loading">Origin port</label>
                      <select id="port_of_loading" name="port_of_loading" value={form.port_of_loading} onChange={handleChange} required>
                        {!portOptions.some((port) => port.value === form.port_of_loading) && <option value={form.port_of_loading}>{labelPort(form.port_of_loading, portMap)}</option>}
                        {portOptions.map((port) => <option key={port.value} value={port.value}>{port.label}</option>)}
                      </select>
                    </div>
                    <ArrowIcon />
                    <div className="field">
                      <label htmlFor="port_of_discharge">Destination port</label>
                      <select id="port_of_discharge" name="port_of_discharge" value={form.port_of_discharge} onChange={handleChange} required>
                        {!portOptions.some((port) => port.value === form.port_of_discharge) && <option value={form.port_of_discharge}>{labelPort(form.port_of_discharge, portMap)}</option>}
                        {portOptions.map((port) => <option key={port.value} value={port.value}>{port.label}</option>)}
                      </select>
                    </div>
                  </div>
                  <div className="field">
                    <label htmlFor="container_count">Container count</label>
                    <input id="container_count" min="0" type="number" name="container_count" value={form.container_count} onChange={handleChange} />
                  </div>
                  <div className="field">
                    <label htmlFor="weight_tons">Weight (tons)</label>
                    <input id="weight_tons" min="0" step="0.1" type="number" name="weight_tons" value={form.weight_tons} onChange={handleChange} />
                  </div>
                  <button className="score-button field-wide" type="submit" disabled={loadingPrediction}>
                    {loadingPrediction ? 'Scoring booking…' : 'Run prediction'}
                    {!loadingPrediction && <ArrowIcon />}
                  </button>
                </form>
              </div>

              <aside className={`decision-card ${risk?.highRisk ? 'risk-high' : risk ? 'risk-low' : ''}`} aria-live="polite">
                <div className="decision-header">
                  <span>Prediction output</span>
                  {prediction && <small>{prediction.shipment_id}</small>}
                </div>
                {risk ? (
                  <>
                    <div className="risk-ring" style={{ '--score': `${Math.min(risk.probability * 100, 100)}%` }}>
                      <div className="risk-ring-inner">
                        <strong>{formatPercent(risk.probability, 0)}</strong>
                        <span>delay probability</span>
                      </div>
                    </div>
                    <div className="risk-copy">
                      <span className="risk-badge">{risk.highRisk ? 'Attention needed' : 'On track'}</span>
                      <h2>{risk.label}</h2>
                      <p>{risk.description}</p>
                    </div>
                    <div className="decision-detail">
                      <span>Estimated delay</span>
                      <strong>{Number(prediction.predicted_delay_hours || 0).toFixed(1)} hours</strong>
                    </div>
                  </>
                ) : (
                  <div className="empty-decision">
                    <div className="empty-orbit" />
                    <h2>Ready when you are</h2>
                    <p>Enter shipment details to generate an AI-backed delay assessment.</p>
                  </div>
                )}
                {predictionError && <p className="form-error">{predictionError}</p>}
              </aside>
            </section>

            {predictionHistory.length > 0 && (
              <section className="history-section">
                <div className="section-heading compact">
                  <div>
                    <p className="section-kicker">Current session</p>
                    <h2>Recently scored shipments</h2>
                  </div>
                </div>
                <div className="history-grid">
                  {predictionHistory.map((item) => (
                    <article className="history-item" key={item.shipment_id}>
                      <div>
                        <span>{item.shipment_id}</span>
                        <strong>{item.is_more_than_24h_late ? 'Review route' : 'Within threshold'}</strong>
                      </div>
                      <b>{formatPercent(item.probability_gt_24h_late, 0)}</b>
                    </article>
                  ))}
                </div>
              </section>
            )}
          </main>
        )
    }
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <button className="brand" type="button" onClick={() => setActivePage('predictor')}>
          <Logo />
          <span>
            <strong>Maersk</strong>
            <small>Supply chain intelligence</small>
          </span>
        </button>
        <nav className="top-nav" aria-label="Primary navigation">
          <button type="button" className={activePage === 'predictor' ? 'active' : ''} onClick={() => setActivePage('predictor')}>Risk predictor</button>
          <button type="button" className={activePage === 'model-health' ? 'active' : ''} onClick={() => setActivePage('model-health')}>Model health</button>
          <button type="button" className={activePage === 'operations' ? 'active' : ''} onClick={() => setActivePage('operations')}>Operations</button>
          <button type="button" className={activePage === 'assistant' ? 'active' : ''} onClick={() => setActivePage('assistant')}>Assistant</button>
          <button type="button" className={activePage === 'pipeline' ? 'active' : ''} onClick={() => setActivePage('pipeline')}>Pipeline</button>
        </nav>
        <button className="refresh-button" type="button" onClick={handleRefresh} disabled={loadingMetrics}>
          <span className={loadingMetrics ? 'refresh-symbol spinning' : 'refresh-symbol'} aria-hidden="true">↻</span>
          Refresh data
        </button>
      </header>

      {renderPage()}
    </div>
  )
}

export default App
