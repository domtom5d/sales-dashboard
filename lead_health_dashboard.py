"""
lead_health_dashboard.py - Lead Scoring Health Dashboard

This dashboard provides an interactive view of your lead scoring model's health,
including score distributions, time decay effects, and regional performance.
"""

import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from enhanced_lead_scoring import (
    train_enhanced_lead_scoring_model,
    compute_final_score,
    categorize_score
)

# Page config
st.set_page_config(
    page_title="Lead Scoring Health Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load and cache data
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

# Score all leads
@st.cache_data
def score_all_leads(df):
    """Apply the scoring model to all leads"""
    if df.empty:
        return df
        
    # Train the model on the data
    model, scaler, feature_weights, thresholds, metrics = train_enhanced_lead_scoring_model(df)
    
    # Create a copy to store scores
    scored_df = df.copy()
    
    # Process each lead
    scores = []
    categories = []
    
    for _, row in df.iterrows():
        # Convert row to dict
        lead_data = row.to_dict()
        
        # Score the lead
        result = compute_final_score(lead_data, df, model, scaler, feature_weights, metrics)
        
        scores.append(result['score'] / 100.0)  # Store as 0-1 for consistency
        categories.append(result['category'])
    
    # Add to DataFrame
    scored_df['score'] = scores
    scored_df['category'] = categories
    
    return scored_df

def main():
    # App title
    st.title("Lead Scoring Health Dashboard")
    st.markdown("Monitor your lead scoring model's performance over time")
    
    # Load data
    with st.spinner("Loading leads data..."):
        leads_df = load_leads()
    
    if leads_df.empty:
        st.warning("No lead data available. Please check your data source.")
        st.info("You can upload a CSV file with lead data below:")
        
        uploaded_file = st.file_uploader("Upload leads CSV", type="csv")
        if uploaded_file is not None:
            leads_df = pd.read_csv(uploaded_file)
            # Convert date columns
            date_cols = ['inquiry_date', 'event_date']
            for col in date_cols:
                if col in leads_df.columns:
                    leads_df[col] = pd.to_datetime(leads_df[col], errors='coerce')
            
            # Calculate days since inquiry
            if 'inquiry_date' in leads_df.columns:
                leads_df['days_since_inquiry'] = (datetime.now() - leads_df['inquiry_date']).dt.days
        else:
            st.stop()
    
    # Score all leads
    with st.spinner("Scoring all leads..."):
        scored_df = score_all_leads(leads_df)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    date_col = None
    for col in ['inquiry_date', 'created_date', 'created_at']:
        if col in scored_df.columns:
            date_col = col
            break
            
    if date_col:
        min_date = scored_df[date_col].min().date()
        max_date = scored_df[date_col].max().date()
        
        start_date = st.sidebar.date_input("Start Date", min_date)
        end_date = st.sidebar.date_input("End Date", max_date)
        
        # Apply date filter
        mask = (scored_df[date_col].dt.date >= start_date) & (scored_df[date_col].dt.date <= end_date)
        filtered_df = scored_df.loc[mask]
    else:
        filtered_df = scored_df
    
    # State filter
    state_col = None
    for col in ['state', 'State', 'region', 'Region']:
        if col in scored_df.columns:
            state_col = col
            break
            
    if state_col and not filtered_df.empty:
        available_states = filtered_df[state_col].dropna().unique().tolist()
        selected_states = st.sidebar.multiselect("States", options=available_states, default=available_states)
        
        if selected_states:
            filtered_df = filtered_df[filtered_df[state_col].isin(selected_states)]
    
    # Main dashboard
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        st.stop()
    
    # KPI section
    st.markdown("## Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Leads", len(filtered_df))
    
    with col2:
        avg_score = filtered_df['score'].mean() * 100  # Convert to 0-100 scale
        st.metric("Average Score", f"{avg_score:.1f}")
    
    with col3:
        hot_count = filtered_df[filtered_df['category'] == 'Hot'].shape[0]
        hot_pct = (hot_count / len(filtered_df)) * 100
        st.metric("Hot Leads", f"{hot_count} ({hot_pct:.1f}%)")
    
    with col4:
        warm_count = filtered_df[filtered_df['category'] == 'Warm'].shape[0]
        warm_pct = (warm_count / len(filtered_df)) * 100
        st.metric("Warm Leads", f"{warm_count} ({warm_pct:.1f}%)")
    
    # Additional KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cool_count = filtered_df[filtered_df['category'] == 'Cool'].shape[0] if 'Cool' in filtered_df['category'].unique() else 0
        cool_pct = (cool_count / len(filtered_df)) * 100 if cool_count > 0 else 0
        st.metric("Cool Leads", f"{cool_count} ({cool_pct:.1f}%)")
    
    with col2:
        cold_count = filtered_df[filtered_df['category'] == 'Cold'].shape[0]
        cold_pct = (cold_count / len(filtered_df)) * 100
        st.metric("Cold Leads", f"{cold_count} ({cold_pct:.1f}%)")
    
    with col3:
        conversion_rate = filtered_df['outcome'].mean() * 100 if 'outcome' in filtered_df.columns else None
        if conversion_rate is not None:
            st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    
    with col4:
        if 'budget' in filtered_df.columns:
            avg_budget = filtered_df['budget'].mean()
            st.metric("Average Budget", f"${avg_budget:,.2f}")
    
    # Score Distribution Chart
    st.markdown("## Score Distribution")
    
    # Create histogram with Altair
    score_hist = alt.Chart(filtered_df).mark_bar().encode(
        alt.X('score:Q', bin=alt.Bin(maxbins=20), title='Score'),
        alt.Y('count()', title='Number of Leads'),
        alt.Color('category:N', scale=alt.Scale(
            domain=['Hot', 'Warm', 'Cool', 'Cold'],
            range=['#ff4b4b', '#ffa64b', '#4bcaff', '#4b83ff']
        )),
        tooltip=['category:N', 'count()', alt.Tooltip('score:Q', format='.2f')]
    ).properties(
        height=300
    )
    
    st.altair_chart(score_hist, use_container_width=True)
    
    # Score over time
    if date_col:
        st.markdown("## Score Trends Over Time")
        
        # Prepare time series data
        time_df = filtered_df.copy()
        time_df['date'] = time_df[date_col].dt.date
        
        # Group by date and calculate average score
        daily_scores = time_df.groupby('date')['score'].mean().reset_index()
        daily_scores['score'] = daily_scores['score'] * 100  # Convert to 0-100 scale
        
        # Create line chart with Altair
        score_line = alt.Chart(daily_scores).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('score:Q', title='Average Score', scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('date:T', format='%b %d, %Y'), alt.Tooltip('score:Q', format='.1f')]
        ).properties(
            height=300
        )
        
        st.altair_chart(score_line, use_container_width=True)
    
    # Regional Analysis
    if state_col:
        st.markdown("## Regional Performance")
        
        # Group by state and calculate counts
        state_counts = filtered_df.groupby([state_col, 'category']).size().reset_index(name='count')
        
        # Create stacked bar chart with Altair
        state_chart = alt.Chart(state_counts).mark_bar().encode(
            x=alt.X(f'{state_col}:N', title='State'),
            y=alt.Y('count:Q', title='Number of Leads'),
            color=alt.Color('category:N', scale=alt.Scale(
                domain=['Hot', 'Warm', 'Cool', 'Cold'],
                range=['#ff4b4b', '#ffa64b', '#4bcaff', '#4b83ff']
            )),
            tooltip=[state_col, 'category', 'count']
        ).properties(
            height=300
        )
        
        st.altair_chart(state_chart, use_container_width=True)
    
    # Data Quality Analysis
    st.markdown("## Data Quality Insights")
    
    # Calculate completeness percentage
    completeness = {}
    
    important_fields = [
        'event_type', 'guest_count', 'budget', 'referral_source', 
        'state', 'days_since_inquiry'
    ]
    
    for field in important_fields:
        if field in filtered_df.columns:
            completeness[field] = (filtered_df[field].notna().sum() / len(filtered_df)) * 100
    
    # Create data quality DataFrame
    quality_df = pd.DataFrame({
        'Field': completeness.keys(),
        'Completeness': completeness.values()
    })
    
    # Create bar chart with Altair
    quality_chart = alt.Chart(quality_df).mark_bar().encode(
        x=alt.X('Completeness:Q', title='Completeness (%)'),
        y=alt.Y('Field:N', sort='-x', title='Field'),
        color=alt.Color('Completeness:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['Field', alt.Tooltip('Completeness:Q', format='.1f')]
    ).properties(
        height=300
    )
    
    st.altair_chart(quality_chart, use_container_width=True)
    
    # Raw data table
    st.markdown("## Scored Leads Data")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )

if __name__ == "__main__":
    main()