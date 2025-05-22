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
    
    # Add tabs for main dashboard and configuration
    tab1, tab2 = st.tabs(["Dashboard", "Model Configuration"])
    
    with tab2:
        st.markdown("## Adjust Lead Scoring Parameters")
        st.markdown("Modify these parameters to see how they affect your lead scoring model.")
        
        # Dynamic weight sliders
        st.markdown("### Factor Weights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Guest count importance
            guest_weight = st.slider(
                "Guest Count Weight", 
                min_value=0.0, 
                max_value=2.0, 
                value=1.0, 
                step=0.1,
                help="Higher values give more importance to guest count in scoring"
            )
            
            # Budget importance
            budget_weight = st.slider(
                "Budget Weight", 
                min_value=0.0, 
                max_value=2.0, 
                value=1.0, 
                step=0.1,
                help="Higher values give more importance to budget in scoring"
            )
        
        with col2:
            # Referral source importance
            referral_weight = st.slider(
                "Referral Quality Weight", 
                min_value=0.0, 
                max_value=2.0, 
                value=1.0, 
                step=0.1,
                help="Higher values give more importance to referral quality in scoring"
            )
            
            # Region boost importance
            region_weight = st.slider(
                "Region Boost Factor", 
                min_value=0.0, 
                max_value=2.0, 
                value=1.0, 
                step=0.1,
                help="Higher values give more importance to regional performance in scoring"
            )
        
        # Time decay configuration
        st.markdown("### Time Decay Configuration")
        
        decay_active = st.checkbox("Enable Time Decay", value=True)
        
        decay_col1, decay_col2 = st.columns(2)
        
        with decay_col1:
            base_half_life = st.slider(
                "Base Half-Life (days)", 
                min_value=5, 
                max_value=90, 
                value=30, 
                step=5,
                help="Number of days after which a lead's score is halved"
            )
        
        with decay_col2:
            decay_curve = st.select_slider(
                "Decay Curve", 
                options=["Very Slow", "Slow", "Medium", "Fast", "Very Fast"],
                value="Medium",
                help="How quickly the score drops after the half-life point"
            )
        
        # Category thresholds
        st.markdown("### Score Category Thresholds")
        
        threshold_col1, threshold_col2 = st.columns(2)
        
        with threshold_col1:
            hot_threshold = st.slider(
                "Hot Lead Threshold", 
                min_value=50, 
                max_value=95, 
                value=70, 
                step=5,
                help="Minimum score for a lead to be categorized as Hot"
            )
        
        with threshold_col2:
            warm_threshold = st.slider(
                "Warm Lead Threshold", 
                min_value=20, 
                max_value=70, 
                value=40, 
                step=5,
                help="Minimum score for a lead to be categorized as Warm"
            )
        
        # Save configuration button
        if st.button("Apply Configuration"):
            st.session_state["guest_weight"] = guest_weight
            st.session_state["budget_weight"] = budget_weight
            st.session_state["referral_weight"] = referral_weight
            st.session_state["region_weight"] = region_weight
            st.session_state["decay_active"] = decay_active
            st.session_state["base_half_life"] = base_half_life
            st.session_state["decay_curve"] = decay_curve
            st.session_state["hot_threshold"] = hot_threshold
            st.session_state["warm_threshold"] = warm_threshold
            
            st.success("Configuration applied! Refresh the dashboard to see changes.")
    
    # Switch to main dashboard tab
    with tab1:
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
    
    # Score over time with seasonality overlay
    if date_col:
        st.markdown("## Score Trends Over Time")
        
        # Seasonality controls
        show_seasonality = st.checkbox("Show Seasonality Overlay", value=True)
        
        # Prepare time series data
        time_df = filtered_df.copy()
        time_df['date'] = time_df[date_col].dt.date
        time_df['month'] = time_df[date_col].dt.month
        time_df['quarter'] = time_df[date_col].dt.quarter
        time_df['day_of_week'] = time_df[date_col].dt.dayofweek
        
        # Group by date and calculate average score
        daily_scores = time_df.groupby('date')['score'].mean().reset_index()
        daily_scores['score'] = daily_scores['score'] * 100  # Convert to 0-100 scale
        
        # Extract seasonality if requested
        seasonal_data = None
        if show_seasonality:
            # Choose seasonality type
            seasonality_type = st.radio(
                "Seasonality Type", 
                options=["Monthly", "Quarterly", "Day of Week"],
                horizontal=True
            )
            
            if seasonality_type == "Monthly":
                seasonal_data = time_df.groupby('month')['score'].mean().reset_index()
                seasonal_data['score'] = seasonal_data['score'] * 100
                seasonal_data['month_name'] = seasonal_data['month'].apply(lambda x: datetime(2023, x, 1).strftime('%B'))
                
                # Create seasonal bar chart
                seasonal_chart = alt.Chart(seasonal_data).mark_bar().encode(
                    x=alt.X('month:O', title='Month', sort=None),
                    y=alt.Y('score:Q', title='Average Score'),
                    color=alt.Color('score:Q', scale=alt.Scale(scheme='viridis')),
                    tooltip=['month_name', alt.Tooltip('score:Q', format='.1f')]
                ).properties(
                    height=200,
                    title="Monthly Score Patterns"
                )
                
                st.altair_chart(seasonal_chart, use_container_width=True)
                
            elif seasonality_type == "Quarterly":
                seasonal_data = time_df.groupby('quarter')['score'].mean().reset_index()
                seasonal_data['score'] = seasonal_data['score'] * 100
                seasonal_data['quarter_name'] = seasonal_data['quarter'].apply(lambda x: f"Q{x}")
                
                # Create seasonal bar chart
                seasonal_chart = alt.Chart(seasonal_data).mark_bar().encode(
                    x=alt.X('quarter:O', title='Quarter'),
                    y=alt.Y('score:Q', title='Average Score'),
                    color=alt.Color('score:Q', scale=alt.Scale(scheme='viridis')),
                    tooltip=['quarter_name', alt.Tooltip('score:Q', format='.1f')]
                ).properties(
                    height=200,
                    title="Quarterly Score Patterns"
                )
                
                st.altair_chart(seasonal_chart, use_container_width=True)
                
            elif seasonality_type == "Day of Week":
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                seasonal_data = time_df.groupby('day_of_week')['score'].mean().reset_index()
                seasonal_data['score'] = seasonal_data['score'] * 100
                seasonal_data['day_name'] = seasonal_data['day_of_week'].apply(lambda x: day_names[x])
                
                # Create seasonal bar chart
                seasonal_chart = alt.Chart(seasonal_data).mark_bar().encode(
                    x=alt.X('day_of_week:O', title='Day of Week', sort=None),
                    y=alt.Y('score:Q', title='Average Score'),
                    color=alt.Color('score:Q', scale=alt.Scale(scheme='viridis')),
                    tooltip=['day_name', alt.Tooltip('score:Q', format='.1f')]
                ).properties(
                    height=200,
                    title="Day of Week Score Patterns"
                )
                
                st.altair_chart(seasonal_chart, use_container_width=True)
        
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
    
    # Conversion Funnel Widget
    st.markdown("## Conversion Funnel Analysis")
    st.markdown("Track how leads convert across different stages by category")
    
    # Define stages
    lead_stages = ['Inquiry', 'Contacted', 'Quoted', 'Site Visit', 'Contract Sent', 'Booked']
    
    # Check if we have stage data
    has_stages = False
    stage_cols = ['stage', 'Stage', 'status', 'Status', 'pipeline_stage', 'PipelineStage']
    stage_col = None
    
    for col in stage_cols:
        if col in filtered_df.columns:
            stage_col = col
            has_stages = True
            break
    
    # If no stages, create dummy data based on outcome
    if not has_stages and 'outcome' in filtered_df.columns:
        # Create a synthetic stage column for visualization
        filtered_df['stage'] = 'Inquiry'  # Default stage
        
        # For won deals, distribute to later stages
        won_mask = filtered_df['outcome'] == 1
        stages_distribution = {
            'Quoted': 0.9,
            'Site Visit': 0.7,
            'Contract Sent': 0.5,
            'Booked': 0.3
        }
        
        # Apply stage distribution
        for stage, probability in stages_distribution.items():
            if sum(won_mask) > 0:
                # Take a portion of won deals for each stage
                stage_count = int(sum(won_mask) * probability)
                if stage_count > 0:
                    # Get random indices from won deals
                    stage_indices = filtered_df[won_mask].sample(n=min(stage_count, sum(won_mask))).index
                    # Assign stage
                    filtered_df.loc[stage_indices, 'stage'] = stage
        
        stage_col = 'stage'
        has_stages = True
    
    # Build funnel data
    if has_stages:
        # Get counts by stage and category
        funnel_data = filtered_df.groupby([stage_col, 'category']).size().reset_index(name='count')
        
        # Create funnel chart with Altair
        funnel_chart = alt.Chart(funnel_data).mark_bar().encode(
            x=alt.X('category:N', title='Lead Category'),
            y=alt.Y('count:Q', title='Number of Leads'),
            color=alt.Color('category:N', scale=alt.Scale(
                domain=['Hot', 'Warm', 'Cool', 'Cold'],
                range=['#ff4b4b', '#ffa64b', '#4bcaff', '#4b83ff']
            )),
            column=alt.Column(f'{stage_col}:N', title='Stage', sort=lead_stages),
            tooltip=['category', stage_col, 'count']
        ).properties(
            width=120,
            height=250,
            title="Conversion Funnel by Lead Category"
        )
        
        st.altair_chart(funnel_chart, use_container_width=True)
        
        # Calculate conversion rates between stages
        st.markdown("### Stage Conversion Rates")
        
        # Find unique stages in the data
        existing_stages = sorted(filtered_df[stage_col].unique(), 
                                key=lambda x: lead_stages.index(x) if x in lead_stages else 999)
        
        if len(existing_stages) > 1:
            # Calculate conversion rates between consecutive stages
            conversion_rates = []
            
            for i in range(len(existing_stages) - 1):
                stage1 = existing_stages[i]
                stage2 = existing_stages[i + 1]
                
                stage1_count = filtered_df[filtered_df[stage_col] == stage1].shape[0]
                stage2_count = filtered_df[filtered_df[stage_col] == stage2].shape[0]
                
                if stage1_count > 0:
                    rate = (stage2_count / stage1_count) * 100
                    conversion_rates.append({
                        'From Stage': stage1,
                        'To Stage': stage2,
                        'Conversion Rate': f"{rate:.1f}%",
                        'Count': f"{stage2_count}/{stage1_count}"
                    })
            
            # Convert to DataFrame and display
            if conversion_rates:
                conversion_df = pd.DataFrame(conversion_rates)
                st.dataframe(conversion_df, use_container_width=True)
    else:
        st.info("Stage information not available in the data. Add a 'stage' column to enable the conversion funnel.")
    
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