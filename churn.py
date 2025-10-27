import os
import glob
from typing import List

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="Phoenix Capital • Client Churn Analysis",
    page_icon="📈",
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

    # Expected columns with aliases
    expected_cols = {
        "Branch Name": ["Branch Name", "Branch", "BranchName"],
        "Client Name": ["Client Name", "Client", "Customer Name"],
        "Client ID": ["Client ID", "Customer ID", "ClientID"],
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

    # Type conversions
    date_cols = ["Disbursed On Date", "Expected Matured On Date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    amount_cols = [
        "Principal Amount",
        "Total Expected Repayment",
        "Total Outstanding",
        "Total Repayment",
    ]
    for c in amount_cols:
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


# -------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------
st.markdown("## 📈 Phoenix Capital — Client Churn Analysis")

default_files = find_excel_files(".")
if not default_files:
    st.error("No Excel file found. Please place the dataset in this folder.")
    st.stop()

with st.spinner("Loading dataset..."):
    df = load_dataset(default_files[-1])  # Use the latest file

# Check for required columns
required_cols = ["Client Name", "Expected Matured On Date", "Total Repayment", "Total Expected Repayment", "Total Outstanding"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns: {', '.join(missing_cols)}")
    st.info("Available columns: " + ", ".join(df.columns))
    st.stop()

if "Expected Matured On Date" not in df.columns:
    st.error("Expected Matured On Date column not found. This analysis requires maturity date information.")
    st.stop()

# -------------------------------------------------------------
# Churn Analysis
# -------------------------------------------------------------
st.markdown("### Client Churn Analysis — October Matured Loans")

# Filter for loans that matured in October
df["MaturityMonth"] = pd.to_datetime(df["Expected Matured On Date"]).dt.month
df["MaturityYear"] = pd.to_datetime(df["Expected Matured On Date"]).dt.year

# Get the latest year with October data
october_years = sorted(df[df["MaturityMonth"] == 10]["MaturityYear"].dropna().unique())
if not october_years:
    st.warning("No loans found that matured in October.")
    st.stop()

latest_october_year = int(october_years[-1])
st.caption(f"Analyzing loans that matured in October {latest_october_year}")

# Filter for October matured loans
october_loans = df[
    (df["MaturityMonth"] == 10) & 
    (df["MaturityYear"] == latest_october_year)
].copy()

if october_loans.empty:
    st.warning(f"No loans found that matured in October {latest_october_year}.")
    st.stop()

# Identify fully paid loans (Total Outstanding = 0 OR Total Repayment >= Total Expected Repayment)
october_loans["IsFullyPaid"] = (
    (october_loans["Total Outstanding"].fillna(0) == 0) |
    (october_loans["Total Repayment"].fillna(0) >= october_loans["Total Expected Repayment"].fillna(0))
)

fully_paid_october = october_loans[october_loans["IsFullyPaid"]].copy()

if fully_paid_october.empty:
    st.warning("No fully paid loans found that matured in October.")
    st.stop()

st.info(f"Found {len(fully_paid_october)} fully paid loans that matured in October {latest_october_year}")

# For each client with a fully paid October loan, check if they took another loan after their October loan
churned_clients = []

for _, loan in fully_paid_october.iterrows():
    client_name = loan["Client Name"]
    client_id = loan.get("Client ID", client_name)  # Use Client ID if available, fallback to name
    
    # Get the maturity date of this loan
    loan_maturity_date = pd.to_datetime(loan["Expected Matured On Date"])
    
    # Find all loans for this client
    if "Client ID" in df.columns and pd.notna(loan.get("Client ID")):
        # Use Client ID for matching if available
        client_loans = df[df["Client ID"] == loan["Client ID"]].copy()
    else:
        # Fallback to Client Name matching
        client_loans = df[df["Client Name"] == client_name].copy()
    
    # Check if client took any loan after the October loan's maturity date
    client_loans["Disbursed On Date"] = pd.to_datetime(client_loans["Disbursed On Date"])
    subsequent_loans = client_loans[client_loans["Disbursed On Date"] > loan_maturity_date]
    
    # If no subsequent loans, this client has churned
    if subsequent_loans.empty:
        churned_clients.append({
            "Client Name": client_name,
            "Client ID": client_id,
            "Branch Name": loan.get("Branch Name", "Unknown"),
            "Loan Officer Name": loan.get("Loan Officer Name", "Unknown"),
            "October Loan ID": loan["Loan ID"],
            "Principal Amount": loan["Principal Amount"],
            "Total Expected Repayment": loan["Total Expected Repayment"],
            "Total Repayment": loan["Total Repayment"],
            "Total Outstanding": loan["Total Outstanding"],
            "Disbursed On Date": loan["Disbursed On Date"],
            "Expected Matured On Date": loan["Expected Matured On Date"],
            "Days Since Maturity": (pd.Timestamp.today() - loan_maturity_date).days,
        })

# Convert to DataFrame
churn_df = pd.DataFrame(churned_clients)

if churn_df.empty:
    st.success("🎉 Great news! No client churn detected. All clients who fully paid their October-matured loans have taken subsequent loans.")
else:
    st.warning(f"⚠️ Found {len(churn_df)} clients who churned after fully paying their October-matured loans.")
    
    # Summary metrics
    total_churned_amount = churn_df["Principal Amount"].sum()
    avg_churned_amount = churn_df["Principal Amount"].mean()
    avg_days_since_maturity = churn_df["Days Since Maturity"].mean()
    
    # Display summary cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Churned Clients", len(churn_df))
    with col2:
        st.metric("Total Churned Amount", kpi_value_fmt(total_churned_amount))
    with col3:
        st.metric("Avg Churned Amount", kpi_value_fmt(avg_churned_amount))
    with col4:
        st.metric("Avg Days Since Maturity", f"{avg_days_since_maturity:.0f}")
    
    # Display detailed table
    st.markdown("#### Churned Clients Details")
    
    # Format the table for display
    display_df = churn_df.copy()
    display_df["Principal Amount"] = display_df["Principal Amount"].map(lambda v: f"{float(v):,.0f}")
    display_df["Total Expected Repayment"] = display_df["Total Expected Repayment"].map(lambda v: f"{float(v):,.0f}")
    display_df["Total Repayment"] = display_df["Total Repayment"].map(lambda v: f"{float(v):,.0f}")
    display_df["Total Outstanding"] = display_df["Total Outstanding"].map(lambda v: f"{float(v):,.0f}")
    display_df["Disbursed On Date"] = pd.to_datetime(display_df["Disbursed On Date"]).dt.strftime("%Y-%m-%d")
    display_df["Expected Matured On Date"] = pd.to_datetime(display_df["Expected Matured On Date"]).dt.strftime("%Y-%m-%d")
    
    # Rename columns for better display
    display_df = display_df.rename(columns={
        "Client Name": "Client",
        "Client ID": "Client ID",
        "Branch Name": "Branch",
        "Loan Officer Name": "Officer",
        "October Loan ID": "Loan ID",
        "Principal Amount": "Principal",
        "Total Expected Repayment": "Expected",
        "Total Repayment": "Repaid",
        "Total Outstanding": "Outstanding",
        "Disbursed On Date": "Disbursed",
        "Expected Matured On Date": "Matured",
        "Days Since Maturity": "Days Since",
    })
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )
    
    # Branch-wise churn analysis
    if len(churn_df) > 1:
        st.markdown("#### Churn by Branch")
        branch_churn = (
            churn_df.groupby("Branch Name")
            .agg({
                "Client Name": "count",
                "Principal Amount": "sum",
                "Days Since Maturity": "mean"
            })
            .rename(columns={
                "Client Name": "Churned Clients",
                "Principal Amount": "Total Amount",
                "Days Since Maturity": "Avg Days Since"
            })
            .sort_values("Churned Clients", ascending=False)
        )
        
        branch_display = branch_churn.copy()
        branch_display["Total Amount"] = branch_display["Total Amount"].map(lambda v: f"{float(v):,.0f}")
        branch_display["Avg Days Since"] = branch_display["Avg Days Since"].map(lambda v: f"{float(v):.0f}")
        
        st.dataframe(
            branch_display,
            use_container_width=True,
        )

# Footer
st.caption(
    "Analysis identifies clients who fully paid loans that matured in October but did not take subsequent loans. "
    "Fully paid = Total Outstanding = 0 OR Total Repayment ≥ Total Expected Repayment."
)
