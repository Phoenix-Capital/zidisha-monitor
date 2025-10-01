import os
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -------------------------------------------------------------
# Date Override (for testing different dates as "today")
# -------------------------------------------------------------
# Set OVERRIDE_DATE to test what the dashboard would show on a different date
# Examples:
OVERRIDE_DATE = None #pd.Timestamp("2024-09-29")  # Yesterday



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
today = OVERRIDE_DATE if OVERRIDE_DATE is not None else pd.Timestamp.today().normalize()
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
    min_value=min_date.to_pydatetime() if pd.notna(min_date) else None,
    max_value=max_date.to_pydatetime() if pd.notna(max_date) else None,
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

# Previous month full-month metrics (loans issued in that month)
prev_month_mask = (
    (filtered["Disbursed On Date"] >= first_prev_month) &
    (filtered["Disbursed On Date"] <= last_prev_month)
)
prev_month_df = filtered.loc[prev_month_mask].copy()

pm_total_disbursed = prev_month_df["Principal Amount"].sum()
pm_total_repaid = prev_month_df["Total Repayment"].sum()
pm_total_outstanding = prev_month_df["Total Outstanding"].sum()
pm_total_expected = prev_month_df["Total Expected Repayment"].sum()
pm_repayment_rate = (pm_total_repaid / pm_total_expected) if pm_total_expected and not pd.isna(pm_total_expected) and pm_total_expected != 0 else np.nan

pm_label = last_prev_month.strftime("%b %Y")

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

# Previous month (full month) KPI cards — loans issued in that month
st.caption(f"Previous month end-of-month snapshot for loans disbursed in {pm_label}")
pm_cols = st.columns(5)
with pm_cols[0]:
    st.metric(f"{pm_label} — Total Disbursed", kpi_value_fmt(pm_total_disbursed))
with pm_cols[1]:
    st.metric(f"{pm_label} — Total Repaid", kpi_value_fmt(pm_total_repaid))
with pm_cols[2]:
    st.metric(f"{pm_label} — Total Outstanding", kpi_value_fmt(pm_total_outstanding))
with pm_cols[3]:
    st.metric(f"{pm_label} — Total Expected", kpi_value_fmt(pm_total_expected))
with pm_cols[4]:
    st.metric(f"{pm_label} — Repayment Rate", pct_fmt(pm_repayment_rate))

## Current month disbursed (to-date) — per-branch cards
cm_mask = (
    (filtered["Disbursed On Date"] >= first_this_month) &
    (filtered["Disbursed On Date"] <= today)
)
cm_df = filtered.loc[cm_mask]
cm_label = today.strftime("%b %Y")

cm_branch = (
    cm_df.groupby("Branch Name", dropna=True)
    .agg({"Principal Amount": "sum", "Loan ID": "nunique"})
    .reset_index()
    .sort_values("Principal Amount", ascending=False)
)

st.markdown(f"#### {cm_label} — Disbursed (to-date) by Branch")
if not cm_branch.empty:
    num_cols = 7
    # Build rendering list and ensure "Overall" lands as the last (rightmost) card
    render_rows = cm_branch.to_dict(orient="records")
    overall_disbursed = float(cm_df["Principal Amount"].sum())
    overall_loans = int(cm_df["Loan ID"].nunique()) if "Loan ID" in cm_df.columns else 0
    total_with_overall = len(render_rows) + 1
    pad_count = (num_cols - (total_with_overall % num_cols)) % num_cols
    if pad_count:
        render_rows.extend([{"_spacer": True} for _ in range(pad_count)])
    render_rows.append({
        "Branch Name": "Overall",
        "Principal Amount": overall_disbursed,
        "Loan ID": overall_loans,
        "_overall": True,
    })

    rows = int(np.ceil(len(render_rows) / num_cols))
    idx = 0
    for _ in range(rows):
        cols = st.columns(num_cols)
        for c in cols:
            if idx >= len(render_rows):
                break
            row = render_rows[idx]
            idx += 1
            if row.get("_spacer"):
                c.write("")
                continue
            label = str(row.get("Branch Name", ""))
            value = kpi_value_fmt(float(row.get("Principal Amount", 0.0)))
            help_txt = f"Loans: {int(row.get('Loan ID', 0))}"
            if row.get("_overall"):
                c.markdown(
                    f"""
<div style="background:#FF7E09; color:white; padding:0.75rem 0.5rem; border-radius:6px; text-align:center;">
  <div style="font-size:0.9rem; font-weight:600;">{label}</div>
  <div style="font-size:1.4rem; font-weight:700;">{value}</div>
  <div style="font-size:0.8rem; opacity:0.95;">{help_txt}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                c.metric(label=label, value=value, help=help_txt)

## (Removed month cohorts comparison table)


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
    st.markdown("#### Branch repayment rate (MTD)")
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

    # Add previous month cohort cards below
    _pm_branch_full = (
        prev_month_df.groupby("Branch Name", dropna=True)[["Total Repayment", "Total Expected Repayment"]]
        .sum()
    )
    _pm_branch = (
        _pm_branch_full
        .assign(
            RepaymentPct=lambda x: np.where(
                x["Total Expected Repayment"] > 0,
                x["Total Repayment"] / x["Total Expected Repayment"],
                np.nan,
            )
        )
        .reset_index()
        .sort_values("RepaymentPct", ascending=False)
    )

    if not _pm_branch.empty:
        st.markdown(f"#### Branch repayment rate (EOM)")
        _num_cols = 6
        _rows = int(np.ceil(len(_pm_branch) / _num_cols))
        _idx = 0
        for _ in range(_rows):
            _cols = st.columns(_num_cols)
            for _c in _cols:
                if _idx >= len(_pm_branch):
                    break
                _row = _pm_branch.iloc[_idx]
                _c.metric(
                    label=str(_row["Branch Name"]),
                    value=f"{(_row['RepaymentPct']*100):.1f}%" if pd.notna(_row["RepaymentPct"]) else "-",
                    help=f"Repaid: {_pm_branch_full.loc[_row['Branch Name'], 'Total Repayment']:,.0f} / Expected: {_pm_branch_full.loc[_row['Branch Name'], 'Total Expected Repayment']:,.0f}"
                )
                _idx += 1


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
    help="Defaults to the same day last month; select any date within available data.",
    min_value=min_date.to_pydatetime() if pd.notna(min_date) else None,
    max_value=max_date.to_pydatetime() if pd.notna(max_date) else None,
)
# Determine effective date without mutating existing widget state
_inline_val = st.session_state.get("day_perf_date_inline")
if _inline_val is not None:
    day_date = pd.to_datetime(_inline_val).normalize()
else:
    day_date = pd.to_datetime(_sidebar_val).normalize()
_base_day_df = filtered.assign(Date=filtered["Disbursed On Date"].dt.floor("D"))
day_df = _base_day_df.query("Date == @day_date")

# Repayment percentage checkpoints for current-month disbursements at 7/14/21/28 days
_cards = st.columns(4)
_filtered_with_date = filtered.assign(__date=pd.to_datetime(filtered["Disbursed On Date"]).dt.floor("D"))
for _idx, _days in enumerate([7, 14, 21, 28]):
    _threshold = (today - pd.Timedelta(days=_days)).normalize()
    _cohort = _filtered_with_date[_filtered_with_date["__date"] == _threshold]
    _exp_sum = pd.to_numeric(_cohort.get("Total Expected Repayment", pd.Series(dtype=float)), errors="coerce").sum()
    _rep_sum = pd.to_numeric(_cohort.get("Total Repayment", pd.Series(dtype=float)), errors="coerce").sum()
    _rate = (_rep_sum / _exp_sum) if _exp_sum and not pd.isna(_exp_sum) and _exp_sum != 0 else np.nan
    with _cards[_idx]:
        st.metric(
            label=f"At {_days} days",
            value=(f"{_rate*100:.1f}%" if pd.notna(_rate) else "-"),
            help=f"Repayment % for loans issued exactly {_days} days ago ({_threshold.strftime('%d %b %Y')})"
)

# Ensure all currently selected branches appear (with zeros if no activity on that day)
all_branches = sorted(filtered["Branch Name"].dropna().unique().tolist())
baseline = pd.DataFrame({
    "Branch Name": all_branches,
})

day_branch = (
    day_df.groupby("Branch Name", dropna=True)[["Total Repayment", "Total Expected Repayment"]]
    .sum()
    .reset_index()
)

day_branch = baseline.merge(day_branch, on="Branch Name", how="left").fillna(0)

# Add aggregated "Zidisha" row = sum of all branches except those matching "advance"
_non_advance_mask = ~day_branch["Branch Name"].str.strip().str.lower().str.contains("advance", na=False)
_agg = day_branch.loc[_non_advance_mask, ["Total Repayment", "Total Expected Repayment"]].sum()
zidisha_row = pd.DataFrame({
    "Branch Name": ["Zidisha"],
    "Total Repayment": [_agg.get("Total Repayment", 0.0)],
    "Total Expected Repayment": [_agg.get("Total Expected Repayment", 0.0)],
})
day_branch = pd.concat([zidisha_row, day_branch], ignore_index=True)

table_day = day_branch.copy()
table_day["Repayment %"] = np.where(
    table_day["Total Expected Repayment"] > 0,
    table_day["Total Repayment"] / table_day["Total Expected Repayment"],
    np.nan,
)
table_day = table_day.sort_values("Repayment %", ascending=False)

# Display as a table with formatted columns
# Ensure "Zidisha" aggregate row is always at the bottom
_is_zidisha = table_day["Branch Name"].astype(str).str.strip().str.lower() == "zidisha"
if _is_zidisha.any():
    _z = table_day.loc[_is_zidisha]
    _o = table_day.loc[~_is_zidisha]
    table_day = pd.concat([_o, _z], ignore_index=True)
st.dataframe(
    table_day.assign(**{
        "Total Expected Repayment": table_day["Total Expected Repayment"].map(lambda v: f"{v:,.0f}"),
        "Total Repayment": table_day["Total Repayment"].map(lambda v: f"{v:,.0f}"),
        "Repayment %": table_day["Repayment %"].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "-"),
    }),
    use_container_width=True,
)

## Officer day performance table (selected date)
st.markdown("#### Day Performance — Officers (selected date)")

# Ensure all currently selected officers appear (with zeros if no activity on that day)
all_officers = sorted(period_df["Loan Officer Name"].dropna().unique().tolist())
officer_baseline = pd.DataFrame({
    "Loan Officer Name": all_officers,
})

officer_day = (
    day_df.groupby("Loan Officer Name", dropna=True)[["Total Repayment", "Total Expected Repayment"]]
    .sum()
    .reset_index()
)
officer_day = officer_baseline.merge(officer_day, on="Loan Officer Name", how="left").fillna(0)

table_officer_day = officer_day.copy()
table_officer_day["Repayment %"] = np.where(
    table_officer_day["Total Expected Repayment"] > 0,
    table_officer_day["Total Repayment"] / table_officer_day["Total Expected Repayment"],
    np.nan,
)
table_officer_day = table_officer_day.sort_values("Repayment %", ascending=False)

st.dataframe(
    table_officer_day.rename(columns={"Loan Officer Name": "Officer"}).assign(**{
        "Total Expected Repayment": table_officer_day["Total Expected Repayment"].map(lambda v: f"{v:,.0f}"),
        "Total Repayment": table_officer_day["Total Repayment"].map(lambda v: f"{v:,.0f}"),
        "Repayment %": table_officer_day["Repayment %"].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "-"),
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


# -------------------------------------------------------------
# 5) Defaulted Amounts — August (Full Month)
# -------------------------------------------------------------
st.markdown("### 5) Defaulted Amounts — August (Full Month)")

_max_d = pd.to_datetime(filtered["Disbursed On Date"]).max()
if pd.isna(_max_d):
    st.info("No data available to compute Aug 1–21 derived repayments.")
else:
    _yr = int(_max_d.year)
    _aug_start = pd.Timestamp(_yr, 8, 1)
    _sep_start = pd.Timestamp(_yr, 9, 1)
    _aug_mask = (
        (pd.to_datetime(filtered["Disbursed On Date"]) >= _aug_start) &
        (pd.to_datetime(filtered["Disbursed On Date"]) < _sep_start)
    )
    _aug_df = filtered.loc[_aug_mask].copy()

    if _aug_df.empty:
        st.info("No August records found under current filters.")
    else:
        # Static defaulted amounts for August (provided)
        _aug_defaulted_static = [
            {"Branch": "Utawala Branch", "Defaulted Amount (August)": 558_384},
            {"Branch": "Pipeline Branch", "Defaulted Amount (August)": 540_488},
            {"Branch": "Kasarani Branch", "Defaulted Amount (August)": 503_011},
            {"Branch": "Kiambu Branch", "Defaulted Amount (August)": 415_887},
            {"Branch": "Adams Branch", "Defaulted Amount (August)": 336_204},
            {"Branch": "Kawangware Branch", "Defaulted Amount (August)": 228_501},
        ]
        _aug_defaulted_df = pd.DataFrame(_aug_defaulted_static)

        # Derived totals from data for August (Expected/Repayment)
        _aug_derived = (
            _aug_df
            .groupby("Branch Name", dropna=True)[["Total Expected Repayment", "Total Repayment"]]
            .sum()
            .reset_index()
            .rename(columns={
                "Branch Name": "Branch",
                "Total Expected Repayment": "Total Expected Repayment Derived",
                "Total Repayment": "Total Repayment Derived",
            })
        )

        # Merge static defaulted with derived totals
        _aug_table = (
            _aug_defaulted_df
            .merge(_aug_derived, on="Branch", how="left")
            .fillna({
                "Total Expected Repayment Derived": 0,
                "Total Repayment Derived": 0,
            })
            .sort_values("Defaulted Amount (August)", ascending=False)
        )

        # Challenge Amount Payed = Static Defaulted - (Expected Derived - Repayment Derived)
        _aug_table["Challenge Amount Payed"] = (
            pd.to_numeric(_aug_table["Defaulted Amount (August)"], errors="coerce")
            - (
                pd.to_numeric(_aug_table["Total Expected Repayment Derived"], errors="coerce")
                - pd.to_numeric(_aug_table["Total Repayment Derived"], errors="coerce")
            )
        )
        _aug_table["Commission (4%)"] = pd.to_numeric(_aug_table["Challenge Amount Payed"], errors="coerce") * 0.04

        st.dataframe(
            _aug_table.assign(**{
                "Defaulted Amount (August)": _aug_table["Defaulted Amount (August)"].map(lambda v: f"{float(v):,.0f}"),
                "Total Expected Repayment Derived": _aug_table["Total Expected Repayment Derived"].map(lambda v: f"{float(v):,.0f}"),
                "Total Repayment Derived": _aug_table["Total Repayment Derived"].map(lambda v: f"{float(v):,.0f}"),
                "Challenge Amount Payed": _aug_table["Challenge Amount Payed"].map(lambda v: f"{float(v):,.0f}"),
                "Commission (4%)": _aug_table["Commission (4%)"].map(lambda v: f"{float(v):,.0f}"),
            }),
            use_container_width=True,
        )


# -------------------------------------------------------------
# 6) New Clients — Zidisha Simba (Current Month)
# -------------------------------------------------------------
st.markdown("### 6) New Clients — Zidisha Simba vs All Products (Current Month)")

# Detect product column
_product_aliases = ["Product Name", "Product", "Loan Product"]

def _normalize_name(n: str) -> str:
    s = str(n).strip().lower()
    s = " ".join(s.replace("_", " ").split())
    return s

_normalized_to_actual = { _normalize_name(c): c for c in df.columns }
_product_col = None
for _cand in _product_aliases:
    _norm_cand = _normalize_name(_cand)
    if _norm_cand in _normalized_to_actual:
        _product_col = _normalized_to_actual[_norm_cand]
        break

if _product_col is None:
    st.info("Product column not found. Expected one of: Product Name, Product, Loan Product.")
elif "Loan Officer Name" not in df.columns:
    st.info("Column 'Loan Officer Name' not found in dataset.")
else:
    _cm_mask_nc = (
        (filtered["Disbursed On Date"] >= first_this_month) &
        (filtered["Disbursed On Date"] <= today)
    )
    _cm_df_nc = filtered.loc[_cm_mask_nc].copy()
    # Baseline: all officers present this month (any product)
    _officers_all = (
        _cm_df_nc.groupby("Loan Officer Name", dropna=True)["Loan ID"].nunique().reset_index().drop(columns=["Loan ID"]).rename(columns={"Loan Officer Name": "Officer"})
    )
    # Filter to product == Zidisha Simba (normalized)
    _cm_df_nc["__prod_norm"] = (
        _cm_df_nc[_product_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ")
        .str.replace("\s+", " ", regex=True)
    )
    _simba = _cm_df_nc[_cm_df_nc["__prod_norm"] == "zidisha simba"].copy()

    _sum_simba = (
        _simba.groupby("Loan Officer Name", dropna=True)["Principal Amount"].sum().reset_index()
        .rename(columns={"Loan Officer Name": "Officer", "Principal Amount": "Principal Amount (Zidisha Simba)"})
    )

    # Total (all products) disbursement per officer in current month
    _sum_all_products = (
        _cm_df_nc.groupby("Loan Officer Name", dropna=True)["Principal Amount"].sum().reset_index()
        .rename(columns={"Loan Officer Name": "Officer", "Principal Amount": "Principal Amount (All Products)"})
    )

    _table_nc = (
        _officers_all
        .merge(_sum_all_products, on="Officer", how="left")
        .merge(_sum_simba, on="Officer", how="left")
        .fillna({
            "Principal Amount (All Products)": 0,
            "Principal Amount (Zidisha Simba)": 0,
        })
        .sort_values("Principal Amount (Zidisha Simba)", ascending=False)
    )

    # Table
    st.dataframe(
        _table_nc.assign(**{
            "Principal Amount (Zidisha Simba)": _table_nc["Principal Amount (Zidisha Simba)"].map(lambda v: f"{float(v):,.0f}"),
            "Principal Amount (All Products)": _table_nc["Principal Amount (All Products)"].map(lambda v: f"{float(v):,.0f}"),
        }),
        use_container_width=True,
    )

    # By Branch table
    st.markdown("#### By Branch — Current Month")
    _branches_all = (
        _cm_df_nc.groupby("Branch Name", dropna=True)["Loan ID"].nunique().reset_index().drop(columns=["Loan ID"]).rename(columns={"Branch Name": "Branch"})
    )
    _sum_all_branch = (
        _cm_df_nc.groupby("Branch Name", dropna=True)["Principal Amount"].sum().reset_index().rename(columns={"Branch Name": "Branch", "Principal Amount": "Principal Amount (All Products)"})
    )
    _sum_simba_branch = (
        _simba.groupby("Branch Name", dropna=True)["Principal Amount"].sum().reset_index().rename(columns={"Branch Name": "Branch", "Principal Amount": "Principal Amount (Zidisha Simba)"})
    )
    _table_branch = (
        _branches_all
        .merge(_sum_all_branch, on="Branch", how="left")
        .merge(_sum_simba_branch, on="Branch", how="left")
        .fillna({
            "Principal Amount (All Products)": 0,
            "Principal Amount (Zidisha Simba)": 0,
        })
        .sort_values("Principal Amount (Zidisha Simba)", ascending=False)
    )
    st.dataframe(
        _table_branch.assign(**{
            "Principal Amount (All Products)": _table_branch["Principal Amount (All Products)"].map(lambda v: f"{float(v):,.0f}"),
            "Principal Amount (Zidisha Simba)": _table_branch["Principal Amount (Zidisha Simba)"].map(lambda v: f"{float(v):,.0f}"),
        }),
        use_container_width=True,
    )

    # By Branch — July vs August (Zidisha Simba) table
    st.markdown("#### By Branch — July vs August (Zidisha Simba)")
    # Use full filtered dataset to capture July/August of the latest year available
    _filtered_all = filtered.copy()
    _filtered_all["__date"] = pd.to_datetime(_filtered_all["Disbursed On Date"]).dt.floor("D")
    _filtered_all["__prod_norm"] = (
        _filtered_all[_product_col]
        .astype(str).str.strip().str.lower().str.replace("_", " ").str.replace("\s+", " ", regex=True)
    )
    _simba_all = _filtered_all[_filtered_all["__prod_norm"] == "zidisha simba"].copy()
    if _simba_all.empty:
        st.info("No 'Zidisha Simba' records available in the current filters to compute July/August table.")
    else:
        _simba_all["__year"] = pd.to_datetime(_simba_all["__date"]).dt.year
        _simba_all["__month"] = pd.to_datetime(_simba_all["__date"]).dt.month
        _candidate_years = sorted(_simba_all.loc[_simba_all["__month"].isin([7, 8]), "__year"].unique())
        if not _candidate_years:
            st.info("No July/August 'Zidisha Simba' records available in the current filters.")
        else:
            _latest_year = int(_candidate_years[-1])
            _july = _simba_all[(_simba_all["__year"] == _latest_year) & (_simba_all["__month"] == 7)]
            _aug = _simba_all[(_simba_all["__year"] == _latest_year) & (_simba_all["__month"] == 8)]

    _july_sum = (
        _july.groupby("Branch Name", dropna=True)["Principal Amount"].sum().reset_index()
        .rename(columns={"Branch Name": "Branch", "Principal Amount": "July Principal (Zidisha Simba)"})
    )
    _july_sum["Branch"] = _july_sum["Branch"].astype(str).str.strip()
    _aug_sum = (
        _aug.groupby("Branch Name", dropna=True)["Principal Amount"].sum().reset_index()
        .rename(columns={"Branch Name": "Branch", "Principal Amount": "August Principal (Zidisha Simba)"})
    )
    _aug_sum["Branch"] = _aug_sum["Branch"].astype(str).str.strip()

    # Compute defaulted for July/Aug (Expected - Repaid, floored at 0)
    _july_er = (
        _july.groupby("Branch Name", dropna=True)[["Total Expected Repayment", "Total Repayment"]]
        .sum().reset_index()
        .rename(columns={
            "Branch Name": "Branch",
            "Total Expected Repayment": "July Expected (Zidisha Simba)",
            "Total Repayment": "July Repaid (Zidisha Simba)",
        })
    )
    _july_er["Branch"] = _july_er["Branch"].astype(str).str.strip()
    _july_er["Defaulted July (Zidisha Simba)"] = (
        pd.to_numeric(_july_er["July Expected (Zidisha Simba)"], errors="coerce")
        - pd.to_numeric(_july_er["July Repaid (Zidisha Simba)"], errors="coerce")
    ).clip(lower=0)

    _aug_er = (
        _aug.groupby("Branch Name", dropna=True)[["Total Expected Repayment", "Total Repayment"]]
        .sum().reset_index()
        .rename(columns={
            "Branch Name": "Branch",
            "Total Expected Repayment": "August Expected (Zidisha Simba)",
            "Total Repayment": "August Repaid (Zidisha Simba)",
        })
    )
    _aug_er["Branch"] = _aug_er["Branch"].astype(str).str.strip()
    _aug_er["Defaulted August (Zidisha Simba)"] = (
        pd.to_numeric(_aug_er["August Expected (Zidisha Simba)"], errors="coerce")
        - pd.to_numeric(_aug_er["August Repaid (Zidisha Simba)"], errors="coerce")
    ).clip(lower=0)

    _branches_union = pd.DataFrame({
        "Branch": sorted(set(_july_sum["Branch"].astype(str)).union(set(_aug_sum["Branch"].astype(str))))
    })
    _branches_union["Branch"] = _branches_union["Branch"].astype(str).str.strip()
    _table_jul_aug = (
        _branches_union
        .merge(_july_sum, on="Branch", how="left")
        .merge(_aug_sum, on="Branch", how="left")
        .merge(_july_er[["Branch", "Defaulted July (Zidisha Simba)"]], on="Branch", how="left")
        .merge(_aug_er[["Branch", "Defaulted August (Zidisha Simba)"]], on="Branch", how="left")
        .fillna({
            "July Principal (Zidisha Simba)": 0,
            "August Principal (Zidisha Simba)": 0,
            "Defaulted July (Zidisha Simba)": 0,
            "Defaulted August (Zidisha Simba)": 0,
        })
        .sort_values("August Principal (Zidisha Simba)", ascending=False)
    )

st.dataframe(
        _table_jul_aug.assign(**{
            "July Principal (Zidisha Simba)": _table_jul_aug["July Principal (Zidisha Simba)"].map(lambda v: f"{float(v):,.0f}"),
            "August Principal (Zidisha Simba)": _table_jul_aug["August Principal (Zidisha Simba)"].map(lambda v: f"{float(v):,.0f}"),
            "Defaulted July (Zidisha Simba)": _table_jul_aug["Defaulted July (Zidisha Simba)"].map(lambda v: f"{float(v):,.0f}"),
            "Defaulted August (Zidisha Simba)": _table_jul_aug["Defaulted August (Zidisha Simba)"].map(lambda v: f"{float(v):,.0f}"),
    }),
    use_container_width=True,
)



