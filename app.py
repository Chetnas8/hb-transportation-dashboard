"""Transportation Incident Operations

st.divider()
st.caption(
    "Built with Python, pandas, Plotly and Streamlit" )

Run from this folder with: python -m streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent / "transportation_incidents_sample.csv"
REQUIRED_COLUMNS = {
    "incident_id", "reported_at", "responded_at", "cleared_at", "roadway",
    "direction", "incident_type", "severity", "lanes_closed",
    "response_minutes", "clearance_minutes", "latitude", "longitude",
    "data_source", "reporting_region",
}
SEVERITY_ORDER = ["Low", "Medium", "High"]
SEVERITY_COLORS = {"Low": "#36a69b", "Medium": "#e8ac4f", "High": "#d65e64"}
BLUE = "#3778ba"

st.set_page_config(
    page_title="Transportation Incident Operations | Portfolio Demo",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
    div[data-testid="stMetric"] {background: #f3f7fa; border: 1px solid #dce6ef;
        padding: 17px 19px; border-radius: 12px;}
    div[data-testid="stMetricLabel"] {color: #345269;}
    div[data-testid="stMetricValue"] {color: #16344d;}
    @media (prefers-color-scheme: dark) {
      div[data-testid="stMetric"] {background: #172b39; border-color: #355060;}
      div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] {color: #f0f6fa;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Read the original CSV without assuming the current working directory."""
    df = pd.read_csv(path, parse_dates=["reported_at", "responded_at", "cleared_at"])
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("The incident dataset contains no records.")
    for col in ("response_minutes", "clearance_minutes", "latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def quality_issues(df: pd.DataFrame) -> list[str]:
    """Simple source-data safeguards; do not silently hide problematic records."""
    issues = []
    missing = int(df[list(REQUIRED_COLUMNS)].isna().sum().sum())
    if missing:
        issues.append(f"{missing} missing required field values")
    dupe = int(df["incident_id"].duplicated().sum())
    if dupe:
        issues.append(f"{dupe} duplicate incident IDs")
    bad_times = ((df["reported_at"] > df["responded_at"]) |
                 (df["responded_at"] > df["cleared_at"])).sum()
    if bad_times:
        issues.append(f"{bad_times} timestamp-order errors")
    response_calc = (df["responded_at"] - df["reported_at"]).dt.total_seconds() / 60
    clearance_calc = (df["cleared_at"] - df["reported_at"]).dt.total_seconds() / 60
    bad_response = (~(response_calc - df["response_minutes"]).abs().le(0.01)).sum()
    bad_clearance = (~(clearance_calc - df["clearance_minutes"]).abs().le(0.01)).sum()
    if bad_response or bad_clearance:
        issues.append(f"{bad_response} response and {bad_clearance} clearance duration mismatches")
    bad_coords = (~df["latitude"].between(-90, 90) | ~df["longitude"].between(-180, 180)).sum()
    if bad_coords:
        issues.append(f"{bad_coords} invalid coordinates")
    return issues


def filter_data(
    df: pd.DataFrame,
    start_date,
    end_date,
    roadways: list[str],
    incident_types: list[str],
    severities: list[str],
) -> pd.DataFrame:
    """Inclusive calendar-day range, applied to incident reported_at."""
    date_mask = df["reported_at"].dt.date.between(start_date, end_date)
    return df.loc[
        date_mask
        & df["roadway"].isin(roadways)
        & df["incident_type"].isin(incident_types)
        & df["severity"].isin(severities)
    ].copy()


def make_chart(fig, *, height: int = 355):
    fig.update_layout(
        height=height, margin=dict(l=5, r=10, t=30, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


try:
    source = load_data(str(DATA_PATH))
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Could not load the CSV: {exc}")
    st.info("Place transportation_incidents_sample.csv in the same folder as app.py.")
    st.stop()

st.title("🚦 Transportation Incident Operations")
st.caption("Long Island | Interactive analytics portfolio")

with st.sidebar:
    st.header("Filters")
    st.caption("Choose the reporting period and incident characteristics.")
    minimum = source["reported_at"].min().date()
    maximum = source["reported_at"].max().date()
    date_selection = st.date_input(
        "Incident reported date", value=(minimum, maximum),
        min_value=minimum, max_value=maximum,
    )
    if isinstance(date_selection, tuple) and len(date_selection) == 2:
        start_date, end_date = date_selection
    else:
        st.warning("Select both start and end dates to continue.")
        st.stop()
    if start_date > end_date:
        st.error("The start date must not be later than the end date.")
        st.stop()
    roadways = st.multiselect("Roadway", sorted(source["roadway"].dropna().unique()),
                              default=sorted(source["roadway"].dropna().unique()))
    incident_types = st.multiselect(
        "Incident type", sorted(source["incident_type"].dropna().unique()),
        default=sorted(source["incident_type"].dropna().unique()),
    )
    severities = st.multiselect("Severity", SEVERITY_ORDER,
                                default=SEVERITY_ORDER)
    st.divider()
    st.caption(f"Source records: {len(source):,} | Coverage: {minimum} through {maximum}")

issues = quality_issues(source)
if issues:
    st.warning("Data-quality issues detected: " + "; ".join(issues) + ". Please investigate the source.")

filtered = filter_data(source, start_date, end_date, roadways, incident_types, severities)

st.subheader("Executive overview")
if filtered.empty:
    st.warning("No incident records match the current filters. Widen your date range or select more categories.")
    st.stop()

incident_count = filtered["incident_id"].nunique()
mean_response = filtered["response_minutes"].mean()
mean_clearance = filtered["clearance_minutes"].mean()
within_target = filtered["response_minutes"].le(15).mean() * 100
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total incidents", f"{incident_count:,}")
k2.metric("Avg. response", f"{mean_response:.1f} min")
k3.metric("Avg. clearance", f"{mean_clearance:.1f} min")
k4.metric("Response ≤ 15 min", f"{within_target:.1f}%", help="Demonstration threshold only")

st.caption(
        "All KPIs respond to the filters."
)

st.subheader("Incident patterns")
filtered["month"] = filtered["reported_at"].dt.to_period("M").dt.to_timestamp()
monthly = filtered.groupby("month").size().reset_index(name="Incidents")
road_counts = filtered.groupby("roadway").size().reset_index(name="Incidents").sort_values("Incidents", ascending=True)
a, b = st.columns(2)
with a:
    st.markdown("**Monthly incident trend**")
    fig = px.line(monthly, x="month", y="Incidents", markers=True,
                  labels={"month": "Reporting month"})
    fig.update_traces(line_color=BLUE, line_width=3, marker_size=6)
    fig.update_xaxes(tickformat="%b %Y")
    make_chart(fig)
with b:
    st.markdown("**Incidents by roadway**")
    fig = px.bar(road_counts, x="Incidents", y="roadway", orientation="h",
                 labels={"roadway": ""}, color_discrete_sequence=[BLUE])
    make_chart(fig)

st.subheader("Geographic incident distribution")
map_df = filtered.dropna(subset=["latitude", "longitude"]).copy()
if not map_df.empty:
    fig = px.scatter_map(
        map_df, lat="latitude", lon="longitude", color="severity",
        color_discrete_map=SEVERITY_COLORS,
        category_orders={"severity": SEVERITY_ORDER},
        hover_name="incident_id",
        hover_data={"roadway": True, "incident_type": True,
                    "response_minutes": True, "latitude": False, "longitude": False},
        zoom=8.5, center={"lat": 40.77, "lon": -73.18},
        opacity=0.6,
    )
    fig.update_traces(marker={"size": 8})
    fig.update_layout(map_style="open-street-map", legend=dict(orientation="h", y=1.02, x=0))
    make_chart(fig, height=440)
else:
    st.warning("No valid coordinates to map for the current selection.")

st.subheader("Operational breakdown")
c, d = st.columns(2)
with c:
    st.markdown("**Incident type distribution**")
    type_counts = filtered.groupby("incident_type").size().reset_index(name="Incidents")
    type_counts = type_counts.sort_values("Incidents", ascending=True)
    fig = px.bar(type_counts, x="Incidents", y="incident_type", orientation="h",
                 labels={"incident_type": ""}, color_discrete_sequence=["#36a69b"])
    make_chart(fig, height=325)
with d:
    st.markdown("**Reported incidents by hour of day**")
    hourly = filtered.groupby(filtered["reported_at"].dt.hour).size().reindex(range(24), fill_value=0)
    hourly_df = pd.DataFrame({"Hour": hourly.index, "Incidents": hourly.values})
    fig = px.bar(hourly_df, x="Hour", y="Incidents", color_discrete_sequence=[BLUE])
    fig.update_xaxes(dtick=3)
    make_chart(fig, height=325)

