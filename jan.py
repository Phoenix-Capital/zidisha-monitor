import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title('Expected vs Collected (Loans maturing 1–21 Jan 2026)')

@st.cache_data
def load_data():
    df = pd.read_excel("sample.xlsx")
    df.columns = df.columns.str.strip()

    # Parse dates
    df["Expected Matured On Date"] = pd.to_datetime(
        df["Expected Matured On Date"],
        errors="coerce",
        dayfirst=True
    )

    # Ensure numeric base columns exist (needed for fallbacks)
    for c in ["Principal Amount", "Principal Outstanding Derived", "Penalties Overdue Derived"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Ensure "Total Expected Repayment Derived" exists; if not, compute a fallback
    if "Total Expected Repayment Derived" not in df.columns:
        df["Total Expected Repayment Derived"] = df["Principal Amount"] + df["Penalties Overdue Derived"]
    else:
        df["Total Expected Repayment Derived"] = pd.to_numeric(
            df["Total Expected Repayment Derived"], errors="coerce"
        ).fillna(0)

    # ✅ Total Repayment Derived (same dataset, same filter will apply later)
    # If the column exists, use it; otherwise compute it from principal - outstanding
    if "Total Repayment Derived" not in df.columns:
        df["Total Repayment Derived"] = df["Principal Amount"] - df["Principal Outstanding Derived"]
    else:
        df["Total Repayment Derived"] = pd.to_numeric(df["Total Repayment Derived"], errors="coerce").fillna(0)

    return df


df = load_data()

# -------------------------------------------------
# 1) Filter: Expected Matured On Date between Jan 1 and Jan 21, 2026 (inclusive)
# -------------------------------------------------
start_date = pd.Timestamp("2026-01-01")
end_date = pd.Timestamp("2026-01-21")

df = df[
    (df["Expected Matured On Date"] >= start_date) &
    (df["Expected Matured On Date"] <= end_date)
].copy()

# -------------------------------------------------
# 2) Exclude Advans Branch permanently
# -------------------------------------------------
df = df[~df["Branch Name"].astype(str).str.strip().str.lower().eq("advans branch")]

# -------------------------------------------------
# 3) Group expected per branch (UNCHANGED) + add Total Repayment Derived (same filter)
# -------------------------------------------------
expected_by_branch = (
    df.groupby("Branch Name", as_index=False)[
        ["Total Expected Repayment Derived", "Total Repayment Derived"]
    ]
    .sum()
    .rename(columns={"Total Expected Repayment Derived": "Expected (maturing 1–21 Jan)"})
)

# -------------------------------------------------
# 4) Add static Collected by 21 (from your screenshot)
# -------------------------------------------------
collected_by_21_map = {
    "Kitengala Branch": 128_600,
    "Kawangware Branch": 1_148_531,
    "Adams Branch": 2_401_437,
    "Pipeline Branch": 2_525_739,
    "Utawala Branch": 1_705_601,
    "Kasarani Branch": 1_681_908,
    "Kiambu Branch": 1_279_769,
}

expected_by_branch["Collected by 21"] = (
    expected_by_branch["Branch Name"].astype(str).str.strip().map(collected_by_21_map).fillna(0)
)

# -------------------------------------------------
# 5) Compute "Collected after 21st" (remaining vs expected)
# -------------------------------------------------
expected_by_branch["Collected after 21"] = (
    expected_by_branch["Expected (maturing 1–21 Jan)"] - expected_by_branch["Collected by 21"]
).clip(lower=0)

# -------------------------------------------------
# 6) Format currency and show ONE TABLE ONLY
# -------------------------------------------------
for c in ["Expected (maturing 1–21 Jan)", "Total Repayment Derived", "Collected by 21", "Collected after 21"]:
    expected_by_branch[c] = expected_by_branch[c].map(lambda x: f"Ksh {x:,.0f}")

st.dataframe(expected_by_branch, use_container_width=True)
