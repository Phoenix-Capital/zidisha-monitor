import os
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="Phoenix Capital • Loan Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------
# Helpers: File discovery and data loading
# -------------------------------------------------------------
def find_excel_files(search_dir: str = ".") -> List[str]:
    excel_paths = glob.glob(os.path.join(search_dir, "*.xlsx"))
    excel_paths += glob.glob(os.path.join(search_dir, "*.xls"))
    return sorted(excel_paths)


@st.cache_data(show_spinner=False)
def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    # Normalize column names: strip, consistent spacing
    df.columns = [str(c).strip() for c in df.columns]

    # Expected columns from the user
    expected_cols = {
        "Branch Name": ["Branch Name", "Branch", "BranchName"],
        "Client Name": ["Client Name", "Client", "Customer Name"],
        "Disbursed On Date": ["Disbursed On Date", "Disbursement Date", "Disbursed Date"],
        "Loan ID": ["Loan ID", "LoanID", "Loan Number"],
        "Loan Officer Name": ["Loan Officer Name", "Officer", "Loan Officer"],
        "Principal Amount": ["Principal Amount", "Principal", "Amount Disbursed"],
        "Total Expected Repayment": [
            "Total Expected Repayment",
            "Expected Repayment",
            "Expected",
            "Total Expected Repayment Derived",
        ],
        "Total Outstanding": [
            "Total Outstanding",
            "Outstanding",
            "Balance",
            "Total Outstanding Derived",
        ],
        "Total Repayment": [
            "Total Repayment",
            "Repaid",
            "Total Repaid",
            "Total Repayment Derived",
        ],
    }

    # Build a mapping from found alias -> canonical name (case-insensitive)
    actual_by_lower = {str(col).lower(): col for col in df.columns}
    alias_to_canonical = {}
    for canonical, aliases in expected_cols.items():
        for alias in aliases:
            alias_lower = str(alias).lower()
            if alias_lower in actual_by_lower:
                actual_name = actual_by_lower[alias_lower]
                alias_to_canonical[actual_name] = canonical
                break

    # Additional heuristic: map any "<Canonical> Derived" to canonical
    for canonical in list(expected_cols.keys()):
        derived_name_lower = f"{canonical} Derived".lower()
        if derived_name_lower in actual_by_lower:
            actual_name = actual_by_lower[derived_name_lower]
            alias_to_canonical[actual_name] = canonical

    # Rename columns to canonical where possible
    if alias_to_canonical:
        df = df.rename(columns=alias_to_canonical)

    # Ensure all required columns exist
    missing = [c for c in expected_cols.keys() if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            ". Found columns: " + ", ".join(df.columns)
        )

    # Type conversions
    date_col = "Disbursed On Date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    amount_cols = [
        "Principal Amount",
        "Total Expected Repayment",
        "Total Outstanding",
        "Total Repayment",
    ]
    for c in amount_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived fields
    df["Month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    # Repayment performance ratio: repaid vs (repaid + outstanding)
    denom = (df["Total Repayment"].fillna(0) + df["Total Outstanding"].fillna(0))
    df["Repayment Performance"] = np.where(denom > 0, df["Total Repayment"].fillna(0) / denom, np.nan)

    return df


def kpi_value_fmt(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}"


def pct_fmt(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1%}"


# -------------------------------------------------------------
# Sidebar: Data source selection & Filters
# -------------------------------------------------------------
st.sidebar.title("Data Source")

default_files = find_excel_files(".")
selected_file = st.sidebar.selectbox(
    "Select dataset file",
    options=default_files,
    index=default_files.index(default_files[-1]) if default_files else 0,
    placeholder="Choose an Excel file",
)

uploaded = st.sidebar.file_uploader("...or upload an Excel file", type=["xlsx", "xls"])

data_file_to_use = selected_file
if uploaded is not None:
    data_file_to_use = uploaded

if not data_file_to_use:
    st.error("No Excel file found. Please place the dataset in this folder or upload it from the sidebar.")
    st.stop()

with st.spinner("Loading dataset..."):
    df = load_dataset(data_file_to_use)


st.sidebar.title("Filters")

# Date range filter
min_date = pd.to_datetime(df["Disbursed On Date"]).min()
max_date = pd.to_datetime(df["Disbursed On Date"]).max()

date_range: Tuple[pd.Timestamp, pd.Timestamp] = st.sidebar.date_input(
    "Disbursement date range",
    value=(min_date.to_pydatetime() if pd.notna(min_date) else None,
           max_date.to_pydatetime() if pd.notna(max_date) else None),
)

branches = sorted([b for b in df["Branch Name"].dropna().unique()])
officers = sorted([o for o in df["Loan Officer Name"].dropna().unique()])

selected_branches = st.sidebar.multiselect("Branch(es)", options=branches, default=branches)
selected_officers = st.sidebar.multiselect("Loan Officer(s)", options=officers, default=officers)


# Apply filters
filtered = df.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    filtered = filtered[(filtered["Disbursed On Date"] >= start_date) & (filtered["Disbursed On Date"] <= end_date)]

if selected_branches:
    filtered = filtered[filtered["Branch Name"].isin(selected_branches)]

if selected_officers:
    filtered = filtered[filtered["Loan Officer Name"].isin(selected_officers)]


# Compute "same period last month" window and apply to all visuals
today = pd.Timestamp.today().normalize()
first_this_month = today.replace(day=1)
last_prev_month = first_this_month - pd.Timedelta(days=1)
first_prev_month = last_prev_month.replace(day=1)
target_day = min(today.day, last_prev_month.day)
start_window = first_prev_month
end_window = first_prev_month.replace(day=target_day)

mask_period = (
    (filtered["Disbursed On Date"] >= pd.to_datetime(start_window)) &
    (filtered["Disbursed On Date"] <= pd.to_datetime(end_window))
)
period_df = filtered.loc[mask_period].copy()

## Sidebar control: day performance date (default to computed same-day-last-month)
st.sidebar.markdown("---")
st.sidebar.markdown("### Day Performance Date")
_default_day = end_window.to_pydatetime()
st.sidebar.date_input(
    "Select day (default: same day last month)",
    value=_default_day,
    key="day_perf_date",
)
_sidebar_val = st.session_state.get("day_perf_date", _default_day)


# -------------------------------------------------------------
# Header & KPI Cards
# -------------------------------------------------------------
st.markdown("## 📊 Phoenix Capital — Loan Performance Dashboard")
st.caption(
    f"Interactive analytics for last month's same period: "
    f"{start_window.strftime('%d %b %Y')} → {end_window.strftime('%d %b %Y')}"
)

total_loans_disbursed = period_df["Principal Amount"].sum()
total_amount_repaid = period_df["Total Repayment"].sum()
total_outstanding = period_df["Total Outstanding"].sum()
total_expected = period_df["Total Expected Repayment"].sum()
overall_repayment_rate = (total_amount_repaid / total_expected) if total_expected and not pd.isna(total_expected) and total_expected != 0 else np.nan

# Best Branch and Officer by repayment performance ratio
branch_perf = (
    period_df.groupby("Branch Name", dropna=True)[["Total Repayment", "Total Outstanding"]]
    .sum()
    .assign(Perf=lambda x: np.where(
        (x["Total Repayment"] + x["Total Outstanding"]) > 0,
        x["Total Repayment"] / (x["Total Repayment"] + x["Total Outstanding"]),
        np.nan,
    ))
    .sort_values("Perf", ascending=False)
)
best_branch = branch_perf.index[0] if len(branch_perf) else "-"

officer_perf = (
    filtered.groupby("Loan Officer Name", dropna=True)[["Total Repayment", "Total Outstanding"]]
    .sum()
    .assign(Perf=lambda x: np.where(
        (x["Total Repayment"] + x["Total Outstanding"]) > 0,
        x["Total Repayment"] / (x["Total Repayment"] + x["Total Outstanding"]),
        np.nan,
    ))
    .sort_values("Perf", ascending=False)
)
best_officer = officer_perf.index[0] if len(officer_perf) else "-"


kpi_cols = st.columns(5)
with kpi_cols[0]:
    st.metric("Total Loans Disbursed", kpi_value_fmt(total_loans_disbursed))
with kpi_cols[1]:
    st.metric("Total Amount Repaid", kpi_value_fmt(total_amount_repaid))
with kpi_cols[2]:
    st.metric("Total Outstanding", kpi_value_fmt(total_outstanding))
with kpi_cols[3]:
    st.metric("Total Expected Repayment", kpi_value_fmt(total_expected))
with kpi_cols[4]:
    st.metric("Repayment Rate (All Branches)", pct_fmt(overall_repayment_rate))


# -------------------------------------------------------------
# 1) Branch Performance Overview
# -------------------------------------------------------------
st.markdown("### 1) Branch Performance Overview")

# Clustered bars: Expected vs Total Repayment by branch
branch_disbursed = (
    period_df.groupby("Branch Name", dropna=True)[["Total Expected Repayment", "Total Repayment"]]
    .sum()
    .reset_index()
)
branch_disbursed_long = branch_disbursed.melt(
    id_vars="Branch Name", var_name="Metric", value_name="Amount"
)
fig_branch_disbursed = px.bar(
    branch_disbursed_long,
    x="Branch Name",
    y="Amount",
    color="Metric",
    barmode="group",
    title="Expected vs Total Repayment by Branch",
    color_discrete_map={
        "Total Expected Repayment": "#ff7f0e",
        "Total Repayment": "#2ca02c",
    },
)
fig_branch_disbursed.update_layout(xaxis_title="Branch", yaxis_title="Amount")


# Total repayments vs outstanding per branch (stacked)
branch_rep_vs_out = (
    period_df.groupby("Branch Name", dropna=True)[["Total Repayment", "Total Outstanding"]].sum().reset_index()
)
branch_rep_vs_out_long = branch_rep_vs_out.melt(id_vars="Branch Name", var_name="Metric", value_name="Amount")
fig_branch_rep_vs_out = px.bar(
    branch_rep_vs_out_long,
    x="Branch Name",
    y="Amount",
    color="Metric",
    barmode="stack",
    title="Repayments vs Outstanding by Branch",
    color_discrete_map={
        "Total Repayment": "#2ca02c",
        "Total Outstanding": "#d62728",
    },
)
fig_branch_rep_vs_out.update_layout(xaxis_title="Branch", yaxis_title="Amount")


## Ranking as cards: percentage = Total Repayment / Total Expected Repayment
branch_expected = (
    period_df.groupby("Branch Name", dropna=True)[["Total Repayment", "Total Expected Repayment"]]
    .sum()
    .reset_index()
)
branch_expected["Repayment %"] = np.where(
    branch_expected["Total Expected Repayment"] > 0,
    branch_expected["Total Repayment"] / branch_expected["Total Expected Repayment"],
    np.nan,
)
branch_expected = branch_expected.sort_values("Repayment %", ascending=False)


col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_branch_disbursed, use_container_width=True)
with col2:
    st.plotly_chart(fig_branch_rep_vs_out, use_container_width=True)

# Card grid for branch repayment percentage
if not branch_expected.empty:
    st.markdown("#### Branch repayment rate (Repaid / Expected)")
    num_cols = 6
    rows = int(np.ceil(len(branch_expected) / num_cols))
    idx = 0
    for _ in range(rows):
        cols = st.columns(num_cols)
        for c in cols:
            if idx >= len(branch_expected):
                break
            row = branch_expected.iloc[idx]
            c.metric(
                label=str(row["Branch Name"]),
                value=f"{(row['Repayment %']*100):.1f}%" if pd.notna(row["Repayment %"]) else "-",
                help=f"Repaid: {row['Total Repayment']:,.0f} / Expected: {row['Total Expected Repayment']:,.0f}"
            )
            idx += 1


# -------------------------------------------------------------
# 2) Loan Repayment Performance
# -------------------------------------------------------------
st.markdown("### 2) Loan Repayment Performance")

total_repaid = period_df["Total Repayment"].sum()
total_out = period_df["Total Outstanding"].sum()
rep_out_df = pd.DataFrame({
    "Category": ["Repaid", "Outstanding"],
    "Amount": [total_repaid, total_out],
})
fig_donut = px.pie(
    rep_out_df,
    names="Category",
    values="Amount",
    hole=0.5,
    title="Repaid vs Outstanding (All Branches)",
    color="Category",
    color_discrete_map={"Repaid": "#2ca02c", "Outstanding": "#d62728"},
)

# Trend: monthly disbursements vs repayments
monthly = (
    period_df.groupby("Month")[
        ["Principal Amount", "Total Repayment"]
    ].sum().reset_index().sort_values("Month")
)
monthly_long = monthly.melt(id_vars="Month", var_name="Metric", value_name="Amount")
fig_trend = px.line(
    monthly_long,
    x="Month",
    y="Amount",
    color="Metric",
    markers=True,
    line_shape="spline",
    title="Monthly Disbursements vs Repayments",
    color_discrete_map={"Principal Amount": "#1f77b4", "Total Repayment": "#2ca02c"},
)
fig_trend.update_layout(xaxis_title="Month", yaxis_title="Amount")

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(fig_donut, use_container_width=True)
with col4:
    st.plotly_chart(fig_trend, use_container_width=True)


# -------------------------------------------------------------
# 3) Loan Officer Performance
# -------------------------------------------------------------
## Same-period last month repayment vs expected (daily)
daily_same_period = (
    period_df.assign(Date=period_df["Disbursed On Date"].dt.floor("D"))
    .groupby("Date")[
        ["Total Repayment", "Total Expected Repayment", "Principal Amount"]
    ]
    .sum()
    .reset_index()
    .sort_values("Date")
)

daily_long = daily_same_period.melt(
    id_vars="Date", var_name="Metric", value_name="Amount"
)

fig_same_period = px.line(
    daily_long,
    x="Date",
    y="Amount",
    color="Metric",
    markers=True,
    line_shape="spline",
    title=f"Same-period last month (" \
          f"{start_window.strftime('%d %b %Y')} → {end_window.strftime('%d %b %Y')})",
    color_discrete_map={
        "Total Repayment": "#2ca02c",
        "Total Expected Repayment": "#ff7f0e",
        "Principal Amount": "#1f77b4",
    },
)
fig_same_period.update_layout(xaxis_title="Date", yaxis_title="Amount")

st.plotly_chart(fig_same_period, use_container_width=True)

# Day performance by branch: Repaid vs Expected (selectable date)
st.markdown("#### Day Performance — selected date")
st.date_input(
    "Change day here (or via sidebar)",
    value=st.session_state.get("day_perf_date", _default_day),
    key="day_perf_date_inline",
    help="Defaults to the same day last month; adjust as needed.",
)
# Determine effective date without mutating existing widget state
_inline_val = st.session_state.get("day_perf_date_inline")
if _inline_val is not None:
    day_date = pd.to_datetime(_inline_val).normalize()
else:
    day_date = pd.to_datetime(_sidebar_val).normalize()
day_df = (
    period_df.assign(Date=period_df["Disbursed On Date"].dt.floor("D"))
    .query("Date == @day_date")
)

# Ensure all currently selected branches appear (with zeros if no activity on that day)
all_branches = sorted(period_df["Branch Name"].dropna().unique().tolist())
baseline = pd.DataFrame({
    "Branch Name": all_branches,
})

day_branch = (
    day_df.groupby("Branch Name", dropna=True)[["Total Repayment", "Total Expected Repayment"]]
    .sum()
    .reset_index()
)

day_branch = baseline.merge(day_branch, on="Branch Name", how="left").fillna(0)

table_day = day_branch.copy()
table_day["Repayment %"] = np.where(
    table_day["Total Expected Repayment"] > 0,
    table_day["Total Repayment"] / table_day["Total Expected Repayment"],
    np.nan,
)
table_day = table_day.sort_values("Repayment %", ascending=False)

# Display as a table with formatted columns
st.dataframe(
    table_day.assign(**{
        "Total Expected Repayment": table_day["Total Expected Repayment"].map(lambda v: f"{v:,.0f}"),
        "Total Repayment": table_day["Total Repayment"].map(lambda v: f"{v:,.0f}"),
        "Repayment %": table_day["Repayment %"].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "-"),
    }),
    use_container_width=True,
)

st.markdown("### 3) Loan Officer Performance")

officer_counts = (
    period_df.groupby("Loan Officer Name", dropna=True)["Loan ID"].nunique().reset_index(name="Loans Disbursed")
)
fig_officer_loans = px.bar(
    officer_counts,
    x="Loan Officer Name",
    y="Loans Disbursed",
    title="Loans Disbursed per Officer",
    color="Loans Disbursed",
    color_continuous_scale="Blues",
)
fig_officer_loans.update_layout(xaxis_title="Officer", yaxis_title="Count", coloraxis_showscale=False)

officer_repaid = (
    period_df.groupby("Loan Officer Name", dropna=True)["Total Repayment"].sum().reset_index()
)
fig_officer_repaid = px.bar(
    officer_repaid,
    x="Loan Officer Name",
    y="Total Repayment",
    title="Total Repayment Collected per Officer",
    color="Total Repayment",
    color_continuous_scale="Greens",
)
fig_officer_repaid.update_layout(xaxis_title="Officer", yaxis_title="Amount", coloraxis_showscale=False)

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(fig_officer_loans, use_container_width=True)
with col6:
    st.plotly_chart(fig_officer_repaid, use_container_width=True)


# -------------------------------------------------------------
# 4) Client Loan Distribution
# -------------------------------------------------------------
st.markdown("### 4) Client Loan Distribution")

fig_hist = px.histogram(
    period_df,
    x="Principal Amount",
    nbins=30,
    title="Distribution of Principal Loan Amounts",
    color_discrete_sequence=["#1f77b4"],
)
fig_hist.update_layout(xaxis_title="Principal Amount", yaxis_title="Count")

fig_scatter = px.scatter(
    period_df,
    x="Principal Amount",
    y="Total Repayment",
    color="Branch Name",
    hover_data=["Client Name", "Loan Officer Name", "Loan ID"],
    title="Principal vs Total Repayment",
)
fig_scatter.update_layout(xaxis_title="Principal Amount", yaxis_title="Total Repayment")

col7, col8 = st.columns(2)
with col7:
    st.plotly_chart(fig_hist, use_container_width=True)
with col8:
    st.plotly_chart(fig_scatter, use_container_width=True)


# Footer note
st.caption(
    "Data fields used: Branch Name, Client Name, Disbursed On Date, Loan ID, "
    "Loan Officer Name, Principal Amount, Total Expected Repayment, Total Outstanding, Total Repayment."
)


