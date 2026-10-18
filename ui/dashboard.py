from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.common.config import CURATED_DIR, RAW_DIR, REPORTS_DIR
from src.ml.predict import predict_delay

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Supply Chain Platform",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_report() -> dict:
    report_path = REPORTS_DIR / "pipeline_report.json"
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


@st.cache_data
def load_shipments() -> pd.DataFrame:
    curated = CURATED_DIR / "shipments.csv"
    if curated.exists():
        return load_csv(curated)
    return load_csv(RAW_DIR / "shipments.csv")


@st.cache_data
def load_port_events() -> pd.DataFrame:
    curated = CURATED_DIR / "port_events.csv"
    if curated.exists():
        return load_csv(curated)
    return load_csv(RAW_DIR / "port_events.csv")


@st.cache_data
def load_ports() -> pd.DataFrame:
    curated = CURATED_DIR / "ports.csv"
    if curated.exists():
        return load_csv(curated)
    return load_csv(RAW_DIR / "ports.csv")


def render_metric(label: str, value: str, delta: str | None = None, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta, help=help_text)


def format_quality_status(report: dict) -> tuple[str, str]:
    if not report:
        return "No pipeline report yet", "warning"

    raw_summary = report.get("raw_quality", {})
    curated_summary = report.get("curated_quality", {})
    failing = [name for name, summary in curated_summary.items() if not summary.get("passed", False)]
    if failing:
        return f"Needs attention: {', '.join(failing)}", "warning"
    return "Healthy curated data", "positive"


page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Shipments", "Port Events", "Ports & Routes", "Data Quality", "Delay Prediction"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Supply Chain Platform")

if page == "Overview":
    shipments = load_shipments()
    events = load_port_events()
    ports = load_ports()
    report = load_report()

    st.title("🚢 Supply Chain Command Center")
    st.caption("Operational DQ monitoring, curated data, route visibility, and delay analytics.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric("Shipments", f"{len(shipments):,}", help_text="Curated shipment records")
    with col2:
        render_metric("Port Events", f"{len(events):,}", help_text="Latest event volume")
    with col3:
        render_metric("Ports", f"{len(ports):,}", help_text="Active port catalog")
    with col4:
        quality_status, tone = format_quality_status(report)
        st.metric("Data Quality", quality_status, delta="" if tone == "positive" else "Review")

    st.markdown("---")

    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Shipment status distribution")
        if "status" in shipments.columns:
            status_counts = shipments["status"].fillna("UNKNOWN").astype(str).value_counts().head(10)
            st.bar_chart(status_counts)
        else:
            st.info("No status field is available in the loaded shipment dataset.")

    with right:
        st.subheader("Route overview")
        if {"port_code", "region"}.issubset(ports.columns):
            port_region = ports.groupby("region").size().sort_values(ascending=False)
            st.dataframe(port_region.rename_axis("region").reset_index(name="port_count"), use_container_width=True)
        else:
            st.info("Port region data is not available in the loaded dataset.")

    st.subheader("Operational highlights")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        if "weight_tons" in shipments.columns:
            st.metric("Avg weight", f"{shipments['weight_tons'].mean():,.1f} tons")
    with summary_cols[1]:
        if "delay_minutes" in events.columns:
            st.metric("Avg delay", f"{events['delay_minutes'].fillna(0).mean():,.1f} min")
    with summary_cols[2]:
        if "event_type" in events.columns:
            st.metric("Event types", f"{events['event_type'].nunique():,}")

elif page == "Shipments":
    shipments = load_shipments()
    st.title("🧾 Shipments")

    if shipments.empty:
        st.warning("No shipment data is available yet.")
        st.stop()

    filters = st.columns(3)
    with filters[0]:
        status_filter = st.selectbox("Status", ["ALL", *sorted(shipments["status"].fillna("UNKNOWN").astype(str).unique().tolist())])
    with filters[1]:
        cargo_filter = st.selectbox("Cargo type", ["ALL", *sorted(shipments["cargo_type"].fillna("UNKNOWN").astype(str).unique().tolist())])
    with filters[2]:
        search_value = st.text_input("Search shipment ID")

    filtered = shipments.copy()
    if status_filter != "ALL":
        filtered = filtered[filtered["status"].fillna("UNKNOWN").astype(str) == status_filter]
    if cargo_filter != "ALL":
        filtered = filtered[filtered["cargo_type"].fillna("UNKNOWN").astype(str) == cargo_filter]
    if search_value:
        filtered = filtered[filtered["shipment_id"].astype(str).str.contains(search_value, case=False, na=False)]

    page_size = st.slider("Rows per page", min_value=10, max_value=100, value=25, step=5)
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start = (current_page - 1) * page_size
    end = start + page_size
    page_data = filtered.iloc[start:end]
    st.dataframe(page_data, use_container_width=True)
    st.caption(f"Showing {len(page_data)} of {len(filtered)} shipments | Page {current_page} of {total_pages}")

elif page == "Port Events":
    events = load_port_events()
    st.title("📍 Port Events")

    if events.empty:
        st.warning("No port event data is available yet.")
        st.stop()

    filters = st.columns(3)
    with filters[0]:
        event_type_filter = st.selectbox("Event type", ["ALL", *sorted(events["event_type"].fillna("UNKNOWN").astype(str).unique().tolist())])
    with filters[1]:
        port_filter = st.selectbox("Port", ["ALL", *sorted(events["port_code"].fillna("UNKNOWN").astype(str).unique().tolist())])
    with filters[2]:
        search_value = st.text_input("Search event ID")

    filtered = events.copy()
    if event_type_filter != "ALL":
        filtered = filtered[filtered["event_type"].fillna("UNKNOWN").astype(str) == event_type_filter]
    if port_filter != "ALL":
        filtered = filtered[filtered["port_code"].fillna("UNKNOWN").astype(str) == port_filter]
    if search_value:
        filtered = filtered[filtered["event_id"].astype(str).str.contains(search_value, case=False, na=False)]

    page_size = st.slider("Rows per page", min_value=10, max_value=100, value=25, step=5)
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start = (current_page - 1) * page_size
    end = start + page_size
    page_data = filtered.iloc[start:end]
    st.dataframe(page_data, use_container_width=True)
    st.caption(f"Showing {len(page_data)} of {len(filtered)} events | Page {current_page} of {total_pages}")

elif page == "Ports & Routes":
    ports = load_ports()
    st.title("🧭 Ports & Routes")
    if ports.empty:
        st.warning("No port data is available yet.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric("Ports", f"{len(ports):,}")
    with col2:
        if "avg_congestion_score" in ports.columns:
            render_metric("Avg congestion", f"{ports['avg_congestion_score'].mean():.2f}")
    with col3:
        if "country" in ports.columns:
            render_metric("Countries", f"{ports['country'].nunique():,}")

    st.dataframe(ports, use_container_width=True)

elif page == "Data Quality":
    report = load_report()
    st.title("✅ Data Quality Overview")
    if not report:
        st.warning("No pipeline report is available yet. Run the ingestion pipeline first.")
        st.stop()

    st.subheader("Quality summary")
    for name, summary in report.get("curated_quality", {}).items():
        passed = summary.get("passed", False)
        st.write(f"### {name.title()}")
        st.success(f"Passed: {passed}") if passed else st.warning(f"Passed: {passed}")
        if summary.get("issues"):
            st.json(summary["issues"])
        else:
            st.write("No issues detected.")

elif page == "Delay Prediction":
    st.title("🤖 Delay Prediction")
    st.caption("Estimate expected delay hours from booking context.")

    with st.form("delay_prediction_form"):
        shipment_id = st.text_input("Shipment ID", value="SHP-001")
        booking_time = st.datetime_input("Booking time", value=pd.Timestamp("2024-01-15 12:00:00"))
        port_of_loading = st.text_input("Port of loading", value="SGSIN")
        port_of_discharge = st.text_input("Port of discharge", value="USLAX")
        submitted = st.form_submit_button("Predict delay")

    if submitted:
        payload = {
            "shipment_id": shipment_id,
            "booking_time": booking_time.isoformat(),
            "port_of_loading": port_of_loading,
            "port_of_discharge": port_of_discharge,
        }
        prediction = predict_delay(payload)
        st.metric("Predicted delay hours", f"{prediction:.2f}")
        st.json(payload)
