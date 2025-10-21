import os
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Phoenix Capital • Q3 Summary", page_icon="📊", layout="wide", initial_sidebar_state="expanded")


def find_excel_files(search_dir: str = ".") -> List[str]:
    excel_paths = glob.glob(os.path.join(search_dir, "*.xlsx"))
    excel_paths += glob.glob(os.path.join(search_dir, "*.xls"))
    return sorted(excel_paths)


@st.cache_data(show_spinner=False)
def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip() for c in df.columns]

    # Alias normalization (mirror of zidisha.py essentials)
    expected_cols = {
        "Branch Name": ["Branch Name", "Branch", "BranchName"],
        "Client Name": ["Client Name", "Client", "Customer Name"],
        "Disbursed On Date": ["Disbursed On Date", "Disbursement Date", "Disbursed Date"],
        "Expected Matured On Date": [
            "Expected Matured On Date",
            "Expected Maturity Date",
            "Maturity Date",
            "Matured On Date",
            "Due Date",
            "Expected Due Date",
        ],
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
    actual_by_lower = {str(col).lower(): col for col in df.columns}
    alias_to_canonical = {}
    for canonical, aliases in expected_cols.items():
        for alias in aliases:
            al = str(alias).lower()
            if al in actual_by_lower:
                alias_to_canonical[actual_by_lower[al]] = canonical
                break
    # Map any "<Canonical> Derived" as well
    for canonical in list(expected_cols.keys()):
        derived_lower = f"{canonical} Derived".lower()
        if derived_lower in actual_by_lower:
            alias_to_canonical[actual_by_lower[derived_lower]] = canonical
    if alias_to_canonical:
        df = df.rename(columns=alias_to_canonical)

    # Conversions
    if "Disbursed On Date" in df.columns:
        df["Disbursed On Date"] = pd.to_datetime(df["Disbursed On Date"], errors="coerce")
    if "Expected Matured On Date" in df.columns:
        df["Expected Matured On Date"] = pd.to_datetime(df["Expected Matured On Date"], errors="coerce")
    for c in ["Principal Amount", "Total Repayment", "Total Expected Repayment", "Total Outstanding"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def kpi_value_fmt(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}"


def pct_fmt(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1%}"


# Data load
files = find_excel_files(".")
if not files:
    st.error("No Excel file found. Place the dataset in this folder.")
    st.stop()

df = load_dataset(files[-1])

st.markdown("## 7) Q3 Summary — July to September (Latest Year)")

st.caption("Morning, Greg — Q3 (July to September) summary for the last three completed quarters.")

_df_dates = pd.to_datetime(df["Disbursed On Date"], errors="coerce")
_today_norm = pd.Timestamp.today().normalize()


def _quarter_start_end(y: int, q: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if q == 1:
        return pd.Timestamp(y, 1, 1), pd.Timestamp(y, 3, 31)
    if q == 2:
        return pd.Timestamp(y, 4, 1), pd.Timestamp(y, 6, 30)
    if q == 3:
        return pd.Timestamp(y, 7, 1), pd.Timestamp(y, 9, 30)
    return pd.Timestamp(y, 10, 1), pd.Timestamp(y, 12, 31)


_quarters = pd.DataFrame({"year": _df_dates.dt.year, "quarter": _df_dates.dt.quarter}).dropna()
if _quarters.empty:
    st.info("No quarterly data found in the dataset.")
    st.stop()

_uq = (
    _quarters.drop_duplicates()
    .assign(start=lambda d: d.apply(lambda r: _quarter_start_end(int(r["year"]), int(r["quarter"]))[0], axis=1))
    .assign(end=lambda d: d.apply(lambda r: _quarter_start_end(int(r["year"]), int(r["quarter"]))[1], axis=1))
    .sort_values(["year", "quarter"])
)
_uq = _uq[_uq["end"] <= _today_norm]
if _uq.empty:
    st.info("No completed quarters available yet.")
    st.stop()

_last_three = _uq.tail(3)
_frames = []
for _, r in _last_three.iterrows():
    y, q = int(r["year"]), int(r["quarter"])
    qs, qe = _quarter_start_end(y, q)
    m = (_df_dates >= qs) & (_df_dates <= qe)
    qdf = df.loc[m].copy()
    _frames.append({"year": y, "quarter": q, "start": qs, "end": qe, "df": qdf})

_curr = _frames[-1]
_prev = _frames[-2] if len(_frames) >= 2 else None
_prev2 = _frames[-3] if len(_frames) >= 3 else None


def _sum_safe(frame_df: pd.DataFrame, col: str) -> float:
    return pd.to_numeric(frame_df.get(col, pd.Series(dtype=float)), errors="coerce").sum()


def _nunique_safe(frame_df: pd.DataFrame, col: str) -> int:
    return int(frame_df.get(col, pd.Series(dtype=object)).nunique()) if col in frame_df.columns else 0

# Disbursement Performance
_curr_disb = _sum_safe(_curr["df"], "Principal Amount")
_prev_disb = _sum_safe(_prev["df"], "Principal Amount") if _prev else np.nan
_curr_loans = _nunique_safe(_curr["df"], "Loan ID")
_curr_avg = (_curr_disb / _curr_loans) if _curr_loans > 0 else np.nan
_prev_loans = _nunique_safe(_prev["df"], "Loan ID") if _prev else np.nan
_prev_avg = (_prev_disb / _prev_loans) if (pd.notna(_prev_disb) and pd.notna(_prev_loans) and _prev_loans > 0) else np.nan

st.markdown("#### Disbursement Performance")
_cols_d = st.columns(3)
with _cols_d[0]:
    _delta = (f"QoQ: {((_curr_disb - _prev_disb) / _prev_disb * 100):.1f}%" if (pd.notna(_prev_disb) and _prev_disb != 0) else None)
    st.metric("Total Disbursement Volume (KES)", kpi_value_fmt(_curr_disb), delta=_delta)
with _cols_d[1]:
    _delta_loans = (f"QoQ: {((_curr_loans - _prev_loans) / _prev_loans * 100):.1f}%" if (pd.notna(_prev_loans) and _prev_loans != 0) else None)
    st.metric("Number of Loans Disbursed", f"{_curr_loans:,}", delta=_delta_loans)
with _cols_d[2]:
    _delta_avg = (f"QoQ: {((_curr_avg - _prev_avg) / _prev_avg * 100):.1f}%" if (pd.notna(_prev_avg) and _prev_avg != 0) else None)
    st.metric("Average Loan Size", kpi_value_fmt(_curr_avg) if pd.notna(_curr_avg) else "-", delta=_delta_avg)

# 2025 quarterly disbursement + average loan charts
_year_target = 2025
_dates_all = pd.to_datetime(df["Disbursed On Date"], errors="coerce")
_is_2025 = _dates_all.dt.year.eq(_year_target)
_df_2025 = df.loc[_is_2025].copy()
if not _df_2025.empty:
    _df_2025["__quarter"] = _dates_all.loc[_is_2025].dt.quarter.values
    _qbar = (
        _df_2025
        .groupby("__quarter", dropna=True)
        .agg(Disbursed=("Principal Amount", "sum"), Loans=("Loan ID", "nunique"))
        .reset_index()
        .sort_values("__quarter")
        .assign(Quarter=lambda d: d["__quarter"].map(lambda q: f"{_year_target} Q{int(q)}"))
    )
    _qbar["AverageLoan"] = np.where(_qbar["Loans"] > 0, _qbar["Disbursed"] / _qbar["Loans"], np.nan)

    _col_disb, _col_avg = st.columns(2)
    with _col_disb:
        fig_qbar = px.bar(_qbar, x="Quarter", y="Disbursed", title=f"Disbursement Volume by Quarter — {_year_target}", color="Disbursed", color_continuous_scale="Blues")
        fig_qbar.update_traces(text=_qbar["Loans"], textposition="outside")
        fig_qbar.update_layout(xaxis_title="Quarter", yaxis_title="KES")
        st.plotly_chart(fig_qbar, width='stretch')
    with _col_avg:
        fig_avg = px.bar(_qbar, x="AverageLoan", y="Quarter", title=f"Average Loan Size by Quarter — {_year_target}", color="AverageLoan", color_continuous_scale="Oranges", orientation="h")
        fig_avg.update_traces(marker=dict(line=dict(width=0), cornerradius=15), width=0.4)
        fig_avg.update_layout(xaxis_title="KES", yaxis_title="Quarter", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif"))
        st.plotly_chart(fig_avg, width='stretch')

# Collections & Repayments using Expected Matured On Date (mirror zidisha.py)
if "Expected Matured On Date" in df.columns:
    _matured_dates = pd.to_datetime(df["Expected Matured On Date"], errors="coerce")
    _curr_matured_mask = (_matured_dates >= _curr["start"]) & (_matured_dates <= _curr["end"])
    _curr_matured_df = df.loc[_curr_matured_mask].copy()
    _prev_matured_mask = (_matured_dates >= _prev["start"]) & (_matured_dates <= _prev["end"]) if _prev else pd.Series([False] * len(df))
    _prev_matured_df = df.loc[_prev_matured_mask].copy() if _prev else pd.DataFrame()

    _curr_repaid = _sum_safe(_curr_matured_df, "Total Repayment")
    _curr_expected = _sum_safe(_curr_matured_df, "Total Expected Repayment")
    _curr_coll_rate = (_curr_repaid / _curr_expected) if (_curr_expected and _curr_expected != 0) else np.nan
    _prev_repaid = _sum_safe(_prev_matured_df, "Total Repayment") if _prev else np.nan
    _prev_expected = _sum_safe(_prev_matured_df, "Total Expected Repayment") if _prev else np.nan
    _prev_coll_rate = (_prev_repaid / _prev_expected) if (pd.notna(_prev_expected) and _prev_expected != 0) else np.nan
else:
    # Fallback to disbursed-based if matured date missing
    _curr_repaid = _sum_safe(_curr["df"], "Total Repayment")
    _curr_expected = _sum_safe(_curr["df"], "Total Expected Repayment")
    _curr_coll_rate = (_curr_repaid / _curr_expected) if (_curr_expected and _curr_expected != 0) else np.nan
    _prev_repaid = _sum_safe(_prev["df"], "Total Repayment") if _prev else np.nan
    _prev_expected = _sum_safe(_prev["df"], "Total Expected Repayment") if _prev else np.nan
    _prev_coll_rate = (_prev_repaid / _prev_expected) if (pd.notna(_prev_expected) and _prev_expected != 0) else np.nan

st.markdown("#### Collections & Repayments")
_cols_c = st.columns(4)
with _cols_c[0]:
    st.metric("Total Collections (KES)", kpi_value_fmt(_curr_repaid))
with _cols_c[1]:
    st.metric("Collection Rate", pct_fmt(_curr_coll_rate), help=f"Prev: {pct_fmt(_prev_coll_rate)}")
with _cols_c[2]:
    st.metric("Recovery from Past Dues", "-", help="Not available in current dataset")
with _cols_c[3]:
    st.metric("Last Quarter Collection Rate", pct_fmt(_prev_coll_rate), help=f"Previous quarter collection performance")

# Customer metrics with Client ID support
_client_id_col = "Client ID" if "Client ID" in df.columns else None
_name_col = "Client Name" if "Client Name" in df.columns else None

if _client_id_col is not None or _name_col is not None:
    if _client_id_col is not None:
        _curr_clients = _curr["df"][_client_id_col].dropna().astype(str)
        _curr_clients_unique = _curr_clients.unique().tolist()
        _to_series = pd.to_numeric(
            _curr["df"].get("Total Outstanding", pd.Series(index=_curr["df"].index, dtype=float)),
            errors="coerce",
        )
        _to_series = _to_series.reindex(_curr["df"].index)
        _active_mask = _to_series > 0
        _curr_active = _curr["df"].loc[_active_mask, _client_id_col].dropna().astype(str).nunique()
        _first_disbursed = (
            df.assign(__d=_df_dates)
            .dropna(subset=[_client_id_col])
            .sort_values("__d")
            .groupby(_client_id_col, dropna=True)["__d"].first()
        )
    else:
        _curr_clients = _curr["df"][_name_col].dropna().astype(str)
        _curr_clients_unique = _curr_clients.unique().tolist()
        _to_series = pd.to_numeric(
            _curr["df"].get("Total Outstanding", pd.Series(index=_curr["df"].index, dtype=float)),
            errors="coerce",
        )
        _to_series = _to_series.reindex(_curr["df"].index)
        _active_mask = _to_series > 0
        _curr_active = _curr["df"].loc[_active_mask, _name_col].dropna().astype(str).nunique()
        _first_disbursed = (
            df.assign(__d=_df_dates)
            .dropna(subset=[_name_col])
            .sort_values("__d")
            .groupby(_name_col, dropna=True)["__d"].first()
        )

    _new_in_curr = int(((_first_disbursed >= _curr["start"]) & (_first_disbursed <= _curr["end"])) .sum())
    _had_prior = _first_disbursed.loc[_first_disbursed.index.isin(_curr_clients_unique)] < _curr["start"] if len(_curr_clients_unique) else pd.Series([], dtype=bool)
    _repeat_count = int(_had_prior.sum()) if len(_curr_clients_unique) else 0
    _repeat_pct = (_repeat_count / len(_curr_clients_unique)) if len(_curr_clients_unique) > 0 else np.nan

    # Tenure
    _tenure_months = "-"
    if "Expected Matured On Date" in _curr["df"].columns:
        _ten_df = _curr["df"]["Expected Matured On Date"].sub(_curr["df"]["Disbursed On Date"]).dt.days
        _tenure_val = np.nanmean(_ten_df / 30.4375) if len(_ten_df) else np.nan
        _tenure_months = f"{_tenure_val:.1f}" if pd.notna(_tenure_val) else "-"

    st.markdown("#### Customer Metrics")
    _cols_u = st.columns(5)
    with _cols_u[0]:
        st.metric("Active Customers", f"{_curr_active:,}")
    with _cols_u[1]:
        st.metric("New Customers (quarter)", f"{_new_in_curr:,}")
    with _cols_u[2]:
        st.metric("Repeat Borrowers (%)", pct_fmt(_repeat_pct))

    # Churn
    _churn_pct = "-"
    if _prev is not None and len(_curr_clients_unique) > 0:
        if _client_id_col is not None:
            _prev_clients = _prev["df"][_client_id_col].dropna().astype(str).unique().tolist()
            _curr_clients_list = _curr_clients_unique
        else:
            _prev_clients = _prev["df"][_name_col].dropna().astype(str).unique().tolist()
            _curr_clients_list = _curr_clients_unique
        _prev_active = set(_prev_clients)
        _curr_active_set = set(_curr_clients_list)
        _churned = _prev_active - _curr_active_set
        _churn_pct = (len(_churned) / len(_prev_active)) if len(_prev_active) > 0 else 0

    with _cols_u[3]:
        st.metric("Average Borrower Tenure (months)", _tenure_months)
    with _cols_u[4]:
        st.metric("Customer Churn (%)", pct_fmt(_churn_pct) if _churn_pct != "-" else "-")
else:
    st.info("Client-level metrics unavailable (missing 'Client ID' or 'Client Name').")

# Financial Performance — Q3 totals
st.markdown("#### Financial Performance — Q3 Totals (July–September)")
_income_months_q3 = ["July", "August", "September"]
_income_df = pd.DataFrame({
    "Metric": ["Interest Income", "Other Income", "Bad debt", "Other Expenses", "Expenses", "Net Income"],
    "April": [3197957.00, 905457.00, -1494371.00, -2788467.00, -4282838.00, -179424.00],
    "May": [3428770.00, 1017881.00, -1602547.00, -2937995.00, -4540542.00, -93891.00],
    "June": [3582409.00, 1040546.00, -1077794.00, -2665609.00, -3743403.00, 879552.00],
    "July": [3697620.00, 1096006.00, -1265367.00, -2671904.00, -3937271.00, 856355.00],
    "August": [3470127.00, 977790.00, -1395616.00, -2847713.00, -4243329.00, 204588.00],
    "September": [3563648.00, 1014877.00, -1660417.00, -2577687.00, -4238104.00, 340421.00],
})
_q3_totals = _income_df.set_index("Metric")[_income_months_q3].sum(axis=1)

def _fmt_kes(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "-"
    return f"({abs(v):,.0f})" if v < 0 else f"{v:,.0f}"

_cols_f = st.columns(6)
for i, m in enumerate(["Interest Income", "Other Income", "Bad debt", "Other Expenses", "Expenses", "Net Income"]):
    with _cols_f[i]:
        st.metric(m, _fmt_kes(_q3_totals.get(m, float("nan"))))

# Income table (April → September)
_income_months_full = ["April", "May", "June", "July", "August", "September"]
_income_data_full = {
    "Metric": ["Interest Income", "Other Income", "Bad debt", "Other Expenses", "Expenses", "Net Income"],
    "April": [3197957.00, 905457.00, -1494371.00, -2788467.00, -4282838.00, -179424.00],
    "May": [3428770.00, 1017881.00, -1602547.00, -2937995.00, -4540542.00, -93891.00],
    "June": [3582409.00, 1040546.00, -1077794.00, -2665609.00, -3743403.00, 879552.00],
    "July": [3697620.00, 1096006.00, -1265367.00, -2671904.00, -3937271.00, 856355.00],
    "August": [3470127.00, 977790.00, -1395616.00, -2847713.00, -4243329.00, 204588.00],
    "September": [3563648.00, 1014877.00, -1660417.00, -2577687.00, -4238104.00, 340421.00],
    "Growth": ["3%", "4%", "19%", "-9%", "8%", "66%"],
}
_income_df_full = pd.DataFrame(_income_data_full).set_index("Metric")

def _fmt_paren(v):
    try:
        fv = float(v)
    except Exception:
        return v
    return f"({abs(fv):,.0f})" if fv < 0 else f"{fv:,.0f}"


def _row_style(s: pd.Series):
    if s.name == "Net Income":
        return ["background-color:#fff176; font-weight:700; font-style:italic"] * len(s)
    if s.name == "Expenses":
        return ["color:#d32f2f; font-weight:700; font-style:italic"] * len(s)
    return [""] * len(s)

_styler = _income_df_full.style.format(_fmt_paren, subset=_income_months_full).apply(_row_style, axis=1)
st.dataframe(_styler, width='stretch')

# Quarterly aggregation table (Q2 vs Q3) below the monthly table
_q2_cols = ["April", "May", "June"]
_q3_cols = ["July", "August", "September"]
_qtr_df = pd.DataFrame({
    "Q2 (Apr–Jun)": _income_df_full[_q2_cols].sum(axis=1),
    "Q3 (Jul–Sep)": _income_df_full[_q3_cols].sum(axis=1),
})

_qtr_styler = _qtr_df.style.format(_fmt_paren).apply(_row_style, axis=1)

# Layout: left = quarterly table, right = monthly growth line chart
_col_tbl, _col_chart = st.columns(2)
with _col_tbl:
    st.markdown("#### Quarterly Financials (Q2 vs Q3)")
    st.dataframe(_qtr_styler, width='stretch')

with _col_chart:
    _growth_metrics = ["Expenses", "Net Income", "Interest Income", "Other Income"]
    _growth_rows = []
    for mtr in _growth_metrics:
        if mtr in _income_df_full.index:
            _series = _income_df_full.loc[mtr, _income_months_full].astype(float)
            for mo, val in _series.items():
                # Make Expenses positive for chart visualization
                if mtr == "Expenses":
                    val = abs(val)
                _growth_rows.append({"Month": mo, "Metric": mtr, "Amount": val})
    _growth_df = pd.DataFrame(_growth_rows)
    # Keep month ordering
    _growth_df["Month"] = pd.Categorical(_growth_df["Month"], categories=_income_months_full, ordered=True)
    fig_growth = px.line(
        _growth_df.sort_values(["Month", "Metric"]),
        x="Month",
        y="Amount",
        color="Metric",
        markers=True,
        line_shape="spline",
        title="Monthly Values — Selected Metrics",
    )
    fig_growth.update_layout(yaxis_title="KES", xaxis_title="Month")
    st.plotly_chart(fig_growth, width='stretch')
