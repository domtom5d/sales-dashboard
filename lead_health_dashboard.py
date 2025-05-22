"""
lead_health_dashboard.py - Lead Scoring Health Dashboard

This dashboard provides an interactive view of your lead scoring model's health,
including score distributions, time decay effects, validation metrics, and regional performance.
"""

import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
from enhanced_lead_scoring import (
    apply_decay, 
    calculate_category_specific_half_life, 
    classify_referral_tiers, 
    identify_high_performing_regions
)

# Page config
st.set_page_config(
    page_title="Lead Scoring Health Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load lead data
@st.cache_data
def load_leads(path="temp_leads.csv"):
    """Load and preprocess lead data"""
    try:
        df = pd.read_csv(path)
        
        # Convert date columns
        date_cols = ['inquiry_date', 'event_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ensure days_since_inquiry exists
        if 'days_since_inquiry' not in df.columns and 'inquiry_date' in df.columns:
            df['days_since_inquiry'] = (datetime.now() - df['inquiry_date']).dt.days
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['inquiry_date', 'event_date', 'days_since_inquiry'])

leads_df = load_leads()

if leads_df.empty:
    st.warning("No lead data available. Please check your data source.")
    st.stop()

# Sidebar: filters and dynamic weight sliders
st.sidebar.header("Filters & Weight Adjustments")

# Date range filter
if 'inquiry_date' in leads_df.columns:
    min_date = leads_df['inquiry_date'].min().date()
    max_date = leads_df['inquiry_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Inquiry Date Range",
        value=(min_date, max_date)
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        date_filter = (
            (leads_df['inquiry_date'].dt.date >= start_date) & 
            (leads_df['inquiry_date'].dt.date <= end_date)
        )
        leads_df = leads_df[date_filter]

# State filter
if 'state' in leads_df.columns:
    all_states = sorted(leads_df['state'].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect(
        "States", 
        options=all_states,
        default=all_states
    )
    
    if selected_states:
        leads_df = leads_df[leads_df['state'].isin(selected_states)]

# Weight sliders
st.sidebar.subheader("Scoring Weights")
weight_guest = st.sidebar.slider("Guest Count Weight", 0.5, 2.0, 1.0, 0.1)
weight_budget = st.sidebar.slider("Budget Weight", 0.5, 2.0, 1.0, 0.1)
weight_decay = st.sidebar.slider("Decay Rate Weight", 0.5, 2.0, 1.0, 0.1)

# Prepare filtered dataframe for scoring
df = leads_df.copy()

# Scoring logic leveraging core functions and dynamic weights
def compute_adjusted_score(row):
    """Compute lead score using enhanced_lead_scoring module functions"""
    # Base scoring: guest count
    guests = row.get("guest_count", 0) or 0
    if pd.isna(guests): guests = 0
    gs = 20 * weight_guest if guests >= 150 else 10 * weight_guest if guests >= 50 else 5 * weight_guest

    # Base scoring: budget
    budget = row.get("budget", 0) or 0
    if pd.isna(budget): budget = 0
    bs = 20 * weight_budget if budget >= 5000 else 10 * weight_budget if budget >= 2000 else 5 * weight_budget

    # Combine base score
    base_score = gs + bs

    # Apply time decay
    days = row.get("days_since_inquiry", 0) or 0
    if pd.isna(days): days = 0
    
    # Use default half-life instead of calculating from data
    half_life = 30 * weight_decay  # Default 30 day half-life
    event_type = row.get("event_type", "default")
    if pd.isna(event_type): event_type = "default"
    
    # Adjust half-life by event type (simplified)
    if event_type and "wedding" in str(event_type).lower():
        half_life = 45 * weight_decay  # Weddings have longer planning cycles
    elif event_type and "corporate" in str(event_type).lower():
        half_life = 15 * weight_decay  # Corporate events have shorter planning cycles
        
    decayed = apply_decay(base_score, days, half_life)

    # Region influence (simplified)
    high_performing_regions = ["California", "New York", "Texas", "Florida"]  # Default high performers
    state = row.get("state", "")
    if pd.isna(state): state = ""
    if state in high_performing_regions:
        decayed *= 1.1

    # Referral influence (simplified)
    top_referrals = ["Website", "Google", "Referral"]  # Default top referrals
    medium_referrals = ["Social Media", "Instagram", "Facebook"]  # Default medium referrals
    referral = row.get("referral_source", "")
    if pd.isna(referral): referral = ""
    
    if referral in top_referrals:
        decayed += 10
    elif referral in medium_referrals:
        decayed += 5
    elif referral and len(referral) > 0:  # Any other referral source that's not empty
        decayed += 2

    # Cap & floor
    return max(0, min(100, decayed))

# Compute scores and categories
if not df.empty:
    df["adjusted_score"] = df.apply(compute_adjusted_score, axis=1)

    # Category mapping
    def categorize(score):
        if score >= 70:
            return "Hot"
        elif score >= 40:
            return "Warm"
        return "Cold"

    df["adjusted_category"] = df["adjusted_score"].apply(categorize)

# Main dashboard
st.title("Lead Scoring Model Health Dashboard")
st.markdown("Monitor your lead scoring model performance with validation metrics")

if df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# KPIs section
st.subheader("Key Performance Indicators")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Leads", len(df))

with col2:
    avg_score = df['adjusted_score'].mean()
    st.metric("Average Score", f"{avg_score:.1f}")

with col3:
    hot_count = df[df['adjusted_category'] == 'Hot'].shape[0]
    warm_count = df[df['adjusted_category'] == 'Warm'].shape[0]
    cold_count = df[df['adjusted_category'] == 'Cold'].shape[0]
    st.metric("Category Counts", f"Hot: {hot_count}, Warm: {warm_count}, Cold: {cold_count}")

# Score Distribution
st.subheader("Score Distribution")
hist = alt.Chart(df).mark_bar().encode(
    alt.X("adjusted_score:Q", bin=alt.Bin(maxbins=20), title="Score"),
    alt.Y("count():Q", title="Count"),
    alt.Color("adjusted_category:N", 
        scale=alt.Scale(
            domain=['Hot', 'Warm', 'Cold'],
            range=['#ff4b4b', '#ffa64b', '#4b83ff']
        )
    ),
    tooltip=["count()", "adjusted_category:N"]
).properties(height=300)
st.altair_chart(hist, use_container_width=True)

# Daily Average Score Trend
if 'inquiry_date' in df.columns:
    st.subheader("Daily Average Score Over Time")
    ts = df.groupby(df["inquiry_date"].dt.date).agg(
        avg_score=("adjusted_score", "mean")
    ).reset_index()
    
    line = alt.Chart(ts).mark_line(point=True).encode(
        x=alt.X("inquiry_date:T", title="Date"),
        y=alt.Y("avg_score:Q", title="Avg Score"),
        tooltip=["inquiry_date:T", alt.Tooltip("avg_score:Q", format=".1f")]
    ).properties(height=300)
    st.altair_chart(line, use_container_width=True)

# Monthly trends
if 'inquiry_date' in df.columns:
    st.subheader("Monthly Score Trends")
    df["month"] = df["inquiry_date"].dt.month_name()
    month_stats = df.groupby("month").agg(
        avg_score=("adjusted_score", "mean"),
        count=("adjusted_score", "count")
    ).reset_index()
    
    month_chart = alt.Chart(month_stats).mark_bar().encode(
        x=alt.X("month:N", sort=None, title="Month"),
        y=alt.Y("avg_score:Q", title="Average Score"),
        color=alt.Color("avg_score:Q", scale=alt.Scale(scheme="viridis")),
        size=alt.Size("count:Q", legend=None),
        tooltip=["month:N", alt.Tooltip("avg_score:Q", format=".1f"), "count:Q"]
    ).properties(height=300)
    st.altair_chart(month_chart, use_container_width=True)

# Category Funnel
st.subheader("Lead Category Funnel")
funnel = df['adjusted_category'].value_counts().reset_index()
funnel.columns = ["category", "count"]
order = ["Hot", "Warm", "Cold"]
funnel['category'] = pd.Categorical(funnel['category'], categories=order, ordered=True)
funnel = funnel.sort_values('category')

funnel_chart = alt.Chart(funnel).mark_bar().encode(
    x="count:Q", 
    y=alt.Y("category:N", sort=order),
    color=alt.Color("category:N", 
        scale=alt.Scale(
            domain=['Hot', 'Warm', 'Cold'],
            range=['#ff4b4b', '#ffa64b', '#4b83ff']
        )
    ),
    tooltip=["category:N", "count:Q"]
).properties(height=200)
st.altair_chart(funnel_chart, use_container_width=True)

# Validation Section
st.header("Model Validation")

# Feature Correlation
st.subheader("Feature Correlations with Score")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'adjusted_score' in numeric_cols:
    numeric_cols.remove('adjusted_score')
    if len(numeric_cols) > 0:
        # Add adjusted_score at the end for better visualization
        corr_features = numeric_cols + ['adjusted_score']
        corr_df = df[corr_features].corr()['adjusted_score'].sort_values(ascending=False)
        
        # Create correlation chart
        corr_chart_df = pd.DataFrame({
            'feature': corr_df.index,
            'correlation': corr_df.values
        })
        corr_chart_df = corr_chart_df[corr_chart_df.feature != 'adjusted_score']
        
        corr_chart = alt.Chart(corr_chart_df).mark_bar().encode(
            x=alt.X('correlation:Q', title='Correlation with Score'),
            y=alt.Y('feature:N', sort='-x', title='Feature'),
            color=alt.Color('correlation:Q', scale=alt.Scale(scheme='blueorange')),
            tooltip=['feature:N', alt.Tooltip('correlation:Q', format='.2f')]
        ).properties(height=300)
        
        st.altair_chart(corr_chart, use_container_width=True)
        
        st.markdown("### Correlation Matrix")
        st.dataframe(df[corr_features].corr().style.background_gradient(cmap='coolwarm'))
    else:
        st.info("No numeric features available for correlation analysis.")
else:
    st.info("Score feature not available for correlation analysis.")

# Conversion Rate Validation
st.subheader("Conversion Rate by Lead Category")
if 'converted' in df.columns:
    conv = df.groupby('adjusted_category')['converted'].mean().reset_index()
    conv.columns = ['Category', 'Conversion Rate']
    
    conv_chart = alt.Chart(conv).mark_bar().encode(
        x=alt.X('Category:N', sort=['Hot', 'Warm', 'Cold']),
        y=alt.Y('Conversion Rate:Q', title='Conversion Rate'),
        color=alt.Color('Category:N', 
            scale=alt.Scale(
                domain=['Hot', 'Warm', 'Cold'],
                range=['#ff4b4b', '#ffa64b', '#4b83ff']
            )
        ),
        tooltip=['Category:N', alt.Tooltip('Conversion Rate:Q', format='.1%')]
    ).properties(height=300)
    
    st.altair_chart(conv_chart, use_container_width=True)
    
    # Ideal pattern explanation
    st.markdown("""
    **Ideal Pattern**: Hot leads should convert at a higher rate than Warm leads, which should convert 
    at a higher rate than Cold leads. If this pattern is not observed, your scoring model may need adjustment.
    """)
else:
    st.info("No 'converted' column found for conversion rate validation.")