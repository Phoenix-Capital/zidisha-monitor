import pandas as pd
import streamlit as st
from datetime import datetime, date
import calendar

# Page config
st.set_page_config(
    page_title="Dynamic Month Analysis",
    page_icon="",
    layout="wide"
)

st.title("Dynamic Month Analysis")
st.markdown("Records by Expected Matured On Date")

# Load data
df = pd.read_excel('Rate_20241217094924.xlsx')

# Extract month and year from Expected Matured On Date
df['Maturity Month'] = pd.to_datetime(df['Expected Matured On Date']).dt.month
df['Maturity Year'] = pd.to_datetime(df['Expected Matured On Date']).dt.year

# Get current month and year
current_date = datetime.now()
current_month = current_date.month
current_year = current_date.year

# Calculate next month
if current_month == 12:
    next_month = 1
    next_year = current_year + 1
else:
    next_month = current_month + 1
    next_year = current_year

# Create month selection
st.markdown("### Month Selection")

# Get available months and years from data
available_years = sorted(df['Maturity Year'].unique())
available_months = sorted(df['Maturity Month'].unique())

# Create month name mapping
month_names = {i: calendar.month_name[i] for i in range(1, 13)}

# Default to current month and next month
default_first_month = current_month
default_second_month = next_month
default_year = current_year

# Month selection
col1, col2, col3 = st.columns(3)

with col1:
    selected_year = st.selectbox(
        "Select Year",
        options=available_years,
        index=available_years.index(default_year) if default_year in available_years else 0
    )

with col2:
    first_month = st.selectbox(
        "First Month",
        options=available_months,
        format_func=lambda x: month_names[x],
        index=available_months.index(default_first_month) if default_first_month in available_months else 0
    )

with col3:
    second_month = st.selectbox(
        "Second Month", 
        options=available_months,
        format_func=lambda x: month_names[x],
        index=available_months.index(default_second_month) if default_second_month in available_months else 1
    )

# Determine years for each month
first_year = selected_year
if first_month == 12 and second_month == 1:
    second_year = selected_year + 1
elif first_month > second_month:
    second_year = selected_year + 1
else:
    second_year = selected_year

# Filter records for selected months
first_month_df = df[(df['Maturity Month'] == first_month) & (df['Maturity Year'] == first_year)].copy()
second_month_df = df[(df['Maturity Month'] == second_month) & (df['Maturity Year'] == second_year)].copy()

# Find clients that exist in both months (Non-churn)
first_month_client_ids = set(first_month_df['Client Id'].astype(str))
second_month_client_ids = set(second_month_df['Client Id'].astype(str))
non_churn_client_ids = first_month_client_ids.intersection(second_month_client_ids)

# Find clients that exist in first month but NOT in second month (Churned)
churned_client_ids = first_month_client_ids - second_month_client_ids

# Create Non-churn table with all records for clients who exist in both months
non_churn_df = df[df['Client Id'].astype(str).isin(non_churn_client_ids)].copy()

# Create Churned table with all records for clients who exist in first month but not second month
churned_df = df[df['Client Id'].astype(str).isin(churned_client_ids)].copy()

# One-row-per-client view for churned (limit to selected first month and deduplicate by client)
churned_clients_df = (
    first_month_df[first_month_df['Client Id'].astype(str).isin(churned_client_ids)]
    .sort_values(['Client Id', 'Expected Matured On Date'])
    .drop_duplicates(subset=['Client Id'], keep='first')
    .copy()
)

# Calculate churn metrics
total_first_month_clients = len(first_month_client_ids)
total_second_month_clients = len(second_month_client_ids)
churned_clients_count = len(churned_client_ids)
churn_rate = (churned_clients_count / total_first_month_clients * 100) if total_first_month_clients > 0 else 0

# Calculate retention rate
retention_rate = (len(non_churn_client_ids) / total_first_month_clients * 100) if total_first_month_clients > 0 else 0

# Calculate client growth
client_growth = total_second_month_clients - total_first_month_clients

# Calculate new clients (in second month but not in first month)
new_client_ids = second_month_client_ids - first_month_client_ids
new_clients_count = len(new_client_ids)

# Calculate totals for churned clients (one-row-per-client basis)
churned_principal = churned_df['Principal Amount'].sum() if not churned_df.empty else 0
churned_expected = (
    churned_clients_df['Total Expected Repayment Derived'].sum()
    if (not churned_clients_df.empty and 'Total Expected Repayment Derived' in churned_clients_df.columns)
    else 0
)

# Calculate avg loan size for churned vs retained clients
avg_loan_churned = churned_df['Principal Amount'].mean() if not churned_df.empty else 0
avg_loan_retained = non_churn_df['Principal Amount'].mean() if not non_churn_df.empty else 0

# Calculate outstanding balance from churned clients
churned_outstanding = churned_df['Total Outstanding Derived'].sum() if not churned_df.empty else 0

# Calculate average churn over past 5 months
def calculate_past_5_months_churn(df, current_year, current_month):
    """Calculate average churn rate over the past 5 months"""
    churn_rates = []
    
    for i in range(5):
        # Calculate month and year for i months ago
        if current_month - i <= 0:
            month = current_month - i + 12
            year = current_year - 1
        else:
            month = current_month - i
            year = current_year
        
        # Skip if we don't have data for this month
        if year not in df['Maturity Year'].values or month not in df[df['Maturity Year'] == year]['Maturity Month'].values:
            continue
            
        # Get clients for this month
        month_df = df[(df['Maturity Month'] == month) & (df['Maturity Year'] == year)]
        month_client_ids = set(month_df['Client Id'].astype(str))
        
        # Get clients for next month
        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year
            
        # Skip if we don't have data for next month
        if next_year not in df['Maturity Year'].values or next_month not in df[df['Maturity Year'] == next_year]['Maturity Month'].values:
            continue
            
        next_month_df = df[(df['Maturity Month'] == next_month) & (df['Maturity Year'] == next_year)]
        next_month_client_ids = set(next_month_df['Client Id'].astype(str))
        
        # Calculate churn rate for this month
        month_churned = month_client_ids - next_month_client_ids
        month_churn_rate = (len(month_churned) / len(month_client_ids) * 100) if len(month_client_ids) > 0 else 0
        churn_rates.append(month_churn_rate)
    
    return sum(churn_rates) / len(churn_rates) if churn_rates else 0

average_churn_5_months = calculate_past_5_months_churn(df, current_year, current_month)

# Calculate retention trend (current vs 5-month average)
average_retention_5_months = 100 - average_churn_5_months
retention_delta = retention_rate - average_retention_5_months

# Calculate churn velocity (change vs average)
churn_velocity = churn_rate - average_churn_5_months

# Display churn metrics cards
st.markdown("### Key Metrics")

# Row 1: Core metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Churn Rate",
        value=f"{churn_rate:.1f}%",
        delta=f"{-churn_velocity:.1f}%" if churn_velocity != 0 else None,
        delta_color="inverse",
        help=f"Percentage of {month_names[first_month]} clients who churned in {month_names[second_month]}"
    )

with col2:
    st.metric(
        label="Retention Rate",
        value=f"{retention_rate:.1f}%",
        delta=f"{retention_delta:.1f}%" if retention_delta != 0 else None,
        help=f"Percentage of {month_names[first_month]} clients retained in {month_names[second_month]}"
    )

with col3:
    st.metric(
        label="Churned Clients",
        value=f"{churned_clients_count:,}",
        help=f"Number of clients who had loans in {month_names[first_month]} but not in {month_names[second_month]}"
    )

with col4:
    st.metric(
        label="New Clients",
        value=f"{new_clients_count:,}",
        help=f"Number of clients who appeared in {month_names[second_month]} but not in {month_names[first_month]}"
    )

# Row 2: Growth and historical metrics
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        label="Client Growth",
        value=f"{client_growth:+,}" if client_growth >= 0 else f"{client_growth:,}",
        delta=f"{client_growth:,} clients",
        delta_color="normal" if client_growth >= 0 else "inverse",
        help=f"Net change in clients from {month_names[first_month]} to {month_names[second_month]}"
    )

with col6:
    st.metric(
        label="Avg Churn (5 months)",
        value=f"{average_churn_5_months:.1f}%",
        help="Average churn rate over the past 5 months"
    )

with col7:
    st.metric(
        label="Avg Retention (5 months)",
        value=f"{average_retention_5_months:.1f}%",
        help="Average retention rate over the past 5 months"
    )

with col8:
    st.metric(
        label="Churn Velocity",
        value=f"{churn_velocity:+.1f}%" if churn_velocity >= 0 else f"{churn_velocity:.1f}%",
        delta="Worse" if churn_velocity > 0 else "Better" if churn_velocity < 0 else "Stable",
        delta_color="inverse" if churn_velocity > 0 else "normal" if churn_velocity < 0 else "off",
        help="Change in churn rate vs 5-month average"
    )

# Row 3: Financial metrics
st.markdown("### Financial Impact")
col9, col10, col11, col12 = st.columns(4)

with col9:
    st.metric(
        label="Expected Repayment (Churned)",
        value=f"KES {churned_expected:,.0f}",
        help="Total expected repayment for churned clients"
    )

with col10:
    st.metric(
        label="Outstanding (Churned)",
        value=f"KES {churned_outstanding:,.0f}",
        help="Total outstanding balance from churned clients"
    )

with col11:
    st.metric(
        label="Avg Loan (Churned)",
        value=f"KES {avg_loan_churned:,.0f}",
        help="Average loan size for churned clients"
    )

with col12:
    st.metric(
        label="Avg Loan (Retained)",
        value=f"KES {avg_loan_retained:,.0f}",
        delta=f"{((avg_loan_retained - avg_loan_churned) / avg_loan_churned * 100):.1f}%" if avg_loan_churned > 0 else None,
        help="Average loan size for retained clients"
    )

# Churn rate over the past 5 months plot
st.markdown("### Churn Trend and Branch Breakdown")

left_col, right_col = st.columns(2)

# Build monthly churn series (oldest → latest), only when both month and next month exist
labels, rates = [], []
for offset in range(5, 0, -1):
    # Determine month/year 'offset' months ago from current month
    if current_month - offset <= 0:
        m = current_month - offset + 12
        y = current_year - 1
    else:
        m = current_month - offset
        y = current_year

    # Next month/year
    if m == 12:
        nm, ny = 1, y + 1
    else:
        nm, ny = m + 1, y

    # Ensure data exists for both months
    has_m = ((df['Maturity Year'] == y) & (df['Maturity Month'] == m)).any()
    has_nm = ((df['Maturity Year'] == ny) & (df['Maturity Month'] == nm)).any()
    if not (has_m and has_nm):
        continue

    month_df = df[(df['Maturity Year'] == y) & (df['Maturity Month'] == m)]
    next_df = df[(df['Maturity Year'] == ny) & (df['Maturity Month'] == nm)]
    month_clients = set(month_df['Client Id'].astype(str))
    next_clients = set(next_df['Client Id'].astype(str))
    churned = month_clients - next_clients
    rate = (len(churned) / len(month_clients) * 100) if len(month_clients) > 0 else 0.0
    labels.append(f"{calendar.month_abbr[m]} {y}")
    rates.append(rate)

with left_col:
    if labels and rates:
        churn_series_df = pd.DataFrame({"Month": labels, "Churn Rate (%)": rates})
        st.bar_chart(churn_series_df.set_index("Month"))
    else:
        st.info("Insufficient data to plot churn trend for the past 5 months.")

# Per-branch churn table for the selected months (first → second)
with right_col:
    # Compute churn per branch based on client presence: clients in first month by branch
    branch_rows = []
    if not first_month_df.empty:
        branches = (
            first_month_df["Branch Name"].dropna().astype(str).str.strip().unique()
        )
        for br in sorted(branches):
            br_first = first_month_df[first_month_df["Branch Name"].astype(str).str.strip() == br]
            clients_first = set(br_first["Client Id"].astype(str))
            churned_br = clients_first - second_month_client_ids
            total_clients = len(clients_first)
            churned_cnt = len(churned_br)
            churn_rate_br = (churned_cnt / total_clients * 100) if total_clients > 0 else 0.0
            branch_rows.append({
                "Branch": br,
                "First Month Clients": total_clients,
                "Churned Clients": churned_cnt,
                "Churn Rate (%)": churn_rate_br,
            })

    if branch_rows:
        branch_churn_df = (
            pd.DataFrame(branch_rows)
            .sort_values("Churn Rate (%)", ascending=False)
            .reset_index(drop=True)
        )

        # Display formatted table
        st.markdown("#### Churn Rate by Branch (Selected Months)")
        st.dataframe(
            branch_churn_df.assign(**{
                "First Month Clients": branch_churn_df["First Month Clients"].map(lambda v: f"{int(v):,}"),
                "Churned Clients": branch_churn_df["Churned Clients"].map(lambda v: f"{int(v):,}"),
                "Churn Rate (%)": branch_churn_df["Churn Rate (%)"].map(lambda v: f"{float(v):.1f}%"),
            }),
            use_container_width=True,
        )
    else:
        st.info("No branch data available for the selected months.")

# Display first month table (collapsed by default)
with st.expander(f"{month_names[first_month]} {first_year} Records", expanded=False):
    st.markdown(f"**Total {month_names[first_month]} records:** {len(first_month_df):,}")

    if not first_month_df.empty:
        # Format numeric columns for display
        first_month_display = first_month_df.copy()
        first_month_display['Principal Amount'] = first_month_display['Principal Amount'].map(lambda x: f"{x:,}")
        first_month_display['Total Expected Repayment Derived'] = first_month_display['Total Expected Repayment Derived'].map(lambda x: f"{x:,.0f}")
        first_month_display['Total Repayment Derived'] = first_month_display['Total Repayment Derived'].map(lambda x: f"{x:,.0f}")
        first_month_display['Total Outstanding Derived'] = first_month_display['Total Outstanding Derived'].map(lambda x: f"{x:,}")
        first_month_display['Penalties Overdue Derived'] = first_month_display['Penalties Overdue Derived'].map(lambda x: f"{x:,.0f}")
        
        # Format dates
        first_month_display['Disbursed On Date'] = first_month_display['Disbursed On Date'].dt.strftime('%Y-%m-%d')
        first_month_display['Expected Matured On Date'] = first_month_display['Expected Matured On Date'].dt.strftime('%Y-%m-%d')
        first_month_display['Matured On Date'] = first_month_display['Matured On Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(first_month_display, use_container_width=True)
    else:
        st.info(f"No {month_names[first_month]} records found.")

# Display second month table (collapsed by default)
with st.expander(f"{month_names[second_month]} {second_year} Records", expanded=False):
    st.markdown(f"**Total {month_names[second_month]} records:** {len(second_month_df):,}")

    if not second_month_df.empty:
        # Format numeric columns for display
        second_month_display = second_month_df.copy()
        second_month_display['Principal Amount'] = second_month_display['Principal Amount'].map(lambda x: f"{x:,}")
        second_month_display['Total Expected Repayment Derived'] = second_month_display['Total Expected Repayment Derived'].map(lambda x: f"{x:,.0f}")
        second_month_display['Total Repayment Derived'] = second_month_display['Total Repayment Derived'].map(lambda x: f"{x:,.0f}")
        second_month_display['Total Outstanding Derived'] = second_month_display['Total Outstanding Derived'].map(lambda x: f"{x:,}")
        second_month_display['Penalties Overdue Derived'] = second_month_display['Penalties Overdue Derived'].map(lambda x: f"{x:,.0f}")
        
        # Format dates
        second_month_display['Disbursed On Date'] = second_month_display['Disbursed On Date'].dt.strftime('%Y-%m-%d')
        second_month_display['Expected Matured On Date'] = second_month_display['Expected Matured On Date'].dt.strftime('%Y-%m-%d')
        second_month_display['Matured On Date'] = second_month_display['Matured On Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(second_month_display, use_container_width=True)
    else:
        st.info(f"No {month_names[second_month]} records found.")

# Display Non-churn table (collapsed by default)
with st.expander("Non-churn Records", expanded=False):
    st.markdown(f"**Unique Non-churn clients:** {len(non_churn_client_ids):,}")

    if not non_churn_df.empty:
        # Format numeric columns for display
        non_churn_display = non_churn_df.copy()
        non_churn_display['Principal Amount'] = non_churn_display['Principal Amount'].map(lambda x: f"{x:,}")
        non_churn_display['Total Expected Repayment Derived'] = non_churn_display['Total Expected Repayment Derived'].map(lambda x: f"{x:,.0f}")
        non_churn_display['Total Repayment Derived'] = non_churn_display['Total Repayment Derived'].map(lambda x: f"{x:,.0f}")
        non_churn_display['Total Outstanding Derived'] = non_churn_display['Total Outstanding Derived'].map(lambda x: f"{x:,}")
        non_churn_display['Penalties Overdue Derived'] = non_churn_display['Penalties Overdue Derived'].map(lambda x: f"{x:,.0f}")
        
        # Format dates
        non_churn_display['Disbursed On Date'] = non_churn_display['Disbursed On Date'].dt.strftime('%Y-%m-%d')
        non_churn_display['Expected Matured On Date'] = non_churn_display['Expected Matured On Date'].dt.strftime('%Y-%m-%d')
        non_churn_display['Matured On Date'] = non_churn_display['Matured On Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(non_churn_display, use_container_width=True)
    else:
        st.info("No Non-churn records found.")

# Display Churned table
st.markdown("## Churned Records")
st.markdown(f"**Unique Churned clients:** {len(churned_client_ids):,}")

if not churned_clients_df.empty:
    # Format numeric columns for display
    churned_display = churned_clients_df.copy()
    churned_display['Principal Amount'] = churned_display['Principal Amount'].map(lambda x: f"{x:,}")
    if 'Total Expected Repayment Derived' in churned_display.columns:
        churned_display['Total Expected Repayment Derived'] = churned_display['Total Expected Repayment Derived'].map(lambda x: f"{x:,.0f}")
    if 'Total Repayment Derived' in churned_display.columns:
        churned_display['Total Repayment Derived'] = churned_display['Total Repayment Derived'].map(lambda x: f"{x:,.0f}")
    if 'Total Outstanding Derived' in churned_display.columns:
        churned_display['Total Outstanding Derived'] = churned_display['Total Outstanding Derived'].map(lambda x: f"{x:,}")
    if 'Penalties Overdue Derived' in churned_display.columns:
        churned_display['Penalties Overdue Derived'] = churned_display['Penalties Overdue Derived'].map(lambda x: f"{x:,.0f}")
    
    # Format dates
    churned_display['Disbursed On Date'] = churned_display['Disbursed On Date'].dt.strftime('%Y-%m-%d')
    churned_display['Expected Matured On Date'] = churned_display['Expected Matured On Date'].dt.strftime('%Y-%m-%d')
    churned_display['Matured On Date'] = churned_display['Matured On Date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(churned_display, use_container_width=True)
else:
    st.info("No Churned records found.")