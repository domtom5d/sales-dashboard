"""
conversion_analysis.py - Comprehensive conversion analysis functionality

This module provides robust data processing and visualization functions 
for the Conversion Analysis dashboard tab.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import datetime

def normalize_data(df_leads, df_ops=None):
    """
    Normalize and clean uploaded data
    
    Args:
        df_leads (DataFrame): The leads data 
        df_ops (DataFrame, optional): The operations data
        
    Returns:
        DataFrame: Normalized and processed dataframe
    """
    # 1) Standardize column names
    df = df_leads.copy()
    df.columns = (
        df.columns
          .astype(str)
          .str.strip()
          .str.lower()
          .str.replace(r'\s+','_', regex=True)
    )
    
    # 2) Parse & merge outcomes
    df['status'] = df['status'].astype(str).str.strip().str.lower() if 'status' in df.columns else 'unknown'
    wins = {'definite', 'tentative'}
    losses = {'lost'}
    df['outcome'] = df['status'].map(lambda s: 1 if s in wins else 0 if s in losses else np.nan)
    df = df.dropna(subset=['outcome'])
    df['outcome'] = df['outcome'].astype(int)
    
    # 3) Ensure date fields are datetime
    date_cols = ['inquiry_date', 'event_date', 'created']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # 4) Merge deal value if available
    if df_ops is not None and 'box_key' in df.columns:
        if 'box_key' in df_ops.columns and 'actual_deal_value' in df_ops.columns:
            df_ops['actual_deal_value'] = pd.to_numeric(df_ops['actual_deal_value'], errors='coerce')
            df = df.merge(df_ops[['box_key', 'actual_deal_value']], on='box_key', how='left')
    
    # 5) Clean numeric columns
    numeric_cols = ['number_of_guests', 'days_until_event', 'days_since_inquiry', 'bartenders_needed']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 6) Generate derived columns
    
    # Guest bins
    if 'number_of_guests' in df.columns:
        bins = [0, 50, 100, 200, 500, np.inf]
        labels = ['1-50', '51-100', '101-200', '201-500', '500+']
        df['guest_bin'] = pd.cut(df['number_of_guests'], bins=bins, labels=labels)
    
    # Days until event bins
    if 'days_until_event' in df.columns:
        bins = [0, 7, 30, 90, np.inf]
        labels = ['≤7d', '8–30d', '31–90d', '90+d']
        df['due_bin'] = pd.cut(df['days_until_event'], bins=bins, labels=labels)
    
    # Days since inquiry bins
    if 'days_since_inquiry' in df.columns:
        bins = [0, 1, 3, 7, 30, np.inf]
        labels = ['0d', '1-3d', '4-7d', '8-30d', '30d+']
        df['dsi_bin'] = pd.cut(df['days_since_inquiry'], bins=bins, labels=labels)
    
    # Staff-to-guest ratio
    if {'bartenders_needed', 'number_of_guests'}.issubset(df.columns):
        df['ratio'] = df['bartenders_needed'] / df['number_of_guests'].replace(0, np.nan)
        bins = [0, 0.01, 0.02, 0.05, np.inf]
        labels = ['<1%', '1-2%', '2-5%', '5%+']
        df['ratio_bin'] = pd.cut(df['ratio'], bins=bins, labels=labels)
    
    # Add weekday for inquiry
    if 'inquiry_date' in df.columns:
        df['weekday'] = df['inquiry_date'].dt.day_name()
    elif 'created' in df.columns:
        df['weekday'] = df['created'].dt.day_name()
    
    # Event month (if available)
    if 'event_date' in df.columns:
        df['event_month'] = df['event_date'].dt.month_name()
    
    return df

def display_kpi_summary(df):
    """Display key performance indicators in a row of metric cards"""
    # --- Intro copy for Conversion Analysis tab ---
    st.markdown("## Conversion Analysis<br>Get a top-level view of overall leads and how many are converting into won deals, all over time.", unsafe_allow_html=True)
    
    # --- KPI Summary cards ---
    total_leads = len(df)
    won = df['outcome'].sum()
    lost = total_leads - won
    conv_rate = won/total_leads if total_leads > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Leads", f"{total_leads:,}")
    col2.metric("Won Deals", f"{won:,}")
    col3.metric("Lost Deals", f"{lost:,}")
    col4.metric("Conversion Rate", f"{conv_rate:.1%}")
    
    # Sparkline under the conversion rate
    date_col = None
    for col in ['inquiry_date', 'created', 'event_date']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        # Create weekly conversion rate data for sparkline
        try:
            weekly = df.set_index(date_col).resample('W')['outcome'].agg(['size','sum'])
            weekly['rate'] = weekly['sum'] / weekly['size']
            weekly['rate'] = weekly['rate'].fillna(0)
            
            # Only show sparkline if we have data
            if not weekly.empty and weekly['size'].sum() > 0:
                with col4:
                    st.line_chart(weekly['rate'], height=100, use_container_width=True)
        except Exception as e:
            # Handle any errors gracefully
            with col4:
                st.info("Weekly trend data not available")

def setup_filters(df):
    """Setup sidebar filters for the dashboard"""
    
    # Get time range if date column exists
    date_col = None
    for col in ['inquiry_date', 'created', 'event_date']:
        if col in df.columns:
            date_col = col
            break
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    # Date filter
    selected_date_range = None
    with filter_col1:
        if date_col:
            try:
                min_date = df[date_col].min().date()
                max_date = df[date_col].max().date()
                
                # Default to last 90 days
                default_start = max_date - datetime.timedelta(days=90)
                default_start = max(default_start, min_date)
                
                selected_date_range = st.date_input(
                    "Date Range",
                    value=(default_start, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            except:
                st.warning("Date filtering unavailable - check date format in data")
    
    # Status filter
    selected_status = None
    with filter_col2:
        status_options = ['All', 'Won', 'Lost']
        selected_status = st.selectbox("Status", options=status_options)
    
    # Region filter
    selected_states = None
    with filter_col3:
        if 'state' in df.columns:
            states = df['state'].dropna().unique().tolist()
            selected_states = st.multiselect("State/Region", options=['All'] + states, default='All')
    
    return {
        'date_range': selected_date_range,
        'status': selected_status,
        'states': selected_states,
        'date_col': date_col
    }

def apply_filters(df, filters):
    """
    Apply selected filters to the dataframe in a centralized way
    
    Args:
        df (DataFrame): DataFrame to filter
        filters (dict): Dictionary containing filter settings
            - date_range: Tuple of (start_date, end_date)
            - status: 'All', 'Won', or 'Lost'
            - states: List of state values or ['All']
            - date_col: Column name to use for date filtering
            
    Returns:
        DataFrame: Filtered dataframe
    """
    filtered_df = df.copy()
    
    # First ensure all date columns are datetime type
    if isinstance(filtered_df, pd.DataFrame) and len(filtered_df) > 0:
        date_columns = ['inquiry_date', 'created', 'event_date']
        for col in date_columns:
            if col in filtered_df.columns and filtered_df[col].dtype != 'datetime64[ns]':
                try:
                    filtered_df[col] = pd.to_datetime(filtered_df[col], errors='coerce')
                except Exception as e:
                    st.warning(f"Error converting {col} to datetime: {e}")
    
    # 1. Date filter
    if (filters.get('date_range') and 
        isinstance(filters['date_range'], (list, tuple)) and 
        len(filters['date_range']) == 2 and 
        filters.get('date_col') and 
        filters['date_col'] in filtered_df.columns):
        
        try:
            start_date, end_date = filters['date_range']
            # Convert start and end dates to datetime for consistent comparison
            start_date_pd = pd.to_datetime(start_date)
            end_date_pd = pd.to_datetime(end_date)
            
            # Add time to end_date to include the full day
            end_date_pd = end_date_pd + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            # Apply filter with safe handling of NaN values
            if isinstance(filtered_df, pd.DataFrame) and len(filtered_df) > 0:
                mask = (
                    (filtered_df[filters['date_col']] >= start_date_pd) & 
                    (filtered_df[filters['date_col']] <= end_date_pd)
                )
                filtered_df = filtered_df[mask.fillna(False)]
                
        except Exception as e:
            st.warning(f"Error applying date filter: {e}")
    
    # 2. Status filter - ensure outcome column is properly formatted as numeric
    if isinstance(filtered_df, pd.DataFrame) and len(filtered_df) > 0:
        if 'outcome' in filtered_df.columns:
            try:
                filtered_df['outcome'] = pd.to_numeric(filtered_df['outcome'], errors='coerce').fillna(0).astype(int)
                
                if filters.get('status') == 'Won':
                    filtered_df = filtered_df[filtered_df['outcome'] == 1]
                elif filters.get('status') == 'Lost':
                    filtered_df = filtered_df[filtered_df['outcome'] == 0]
            except Exception as e:
                st.warning(f"Error applying status filter: {e}")
    
    # 3. State/Region filter - handle case insensitively
    if (isinstance(filtered_df, pd.DataFrame) and 
        len(filtered_df) > 0 and 
        'state' in filtered_df.columns and 
        filters.get('states') and 
        isinstance(filters['states'], list) and
        'All' not in filters['states']):
        
        try:
            # Convert state values to string for consistent comparison
            filtered_df['state'] = filtered_df['state'].astype(str)
            
            # Create mask with case-insensitive matching
            state_values = [str(s).lower() for s in filters['states']]
            mask = filtered_df['state'].str.lower().isin(state_values)
            filtered_df = filtered_df[mask.fillna(False)]
        except Exception as e:
            st.warning(f"Error applying state/region filter: {e}")
    
    return filtered_df

def plot_time_trends(df):
    """Create trend charts for conversion rate and lead volume over time"""
    
    # Determine appropriate date column
    date_col = None
    for col in ['inquiry_date', 'created', 'event_date']:
        if col in df.columns:
            date_col = col
            break
    
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        st.write("#### Conversion Rate Over Time")
        if date_col and not df.empty:
            try:
                # Ensure date column is datetime
                df_trend = df.copy()
                
                # Add week column
                df_trend['week'] = df_trend[date_col].dt.to_period('W').dt.start_time
                
                # Weekly conversion rate
                weekly_conv = df_trend.groupby('week').agg(
                    won=('outcome', 'sum'),
                    total=('outcome', 'count')
                ).assign(rate=lambda d: d['won']/d['total'])
                
                if not weekly_conv.empty:
                    # Plot weekly trend
                    fig, ax = plt.subplots(figsize=(10, 5))
                    weekly_conv['rate'].plot(kind='line', marker='o', ax=ax)
                    ax.set_ylabel('Conversion Rate')
                    ax.set_xlabel('Week')
                    ax.set_ylim(0, min(1, weekly_conv['rate'].max() * 1.2) if not weekly_conv['rate'].empty else 1)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Not enough time-based data for trend analysis")
            except Exception as e:
                st.error(f"Error creating time trend: {str(e)}")
        else:
            st.info("Date information not available for trend analysis")
    
    with trend_col2:
        st.write("#### Lead Volume Over Time")
        if date_col and not df.empty:
            try:
                # Ensure date column is datetime
                df_trend = df.copy()
                
                # Add week column
                df_trend['week'] = df_trend[date_col].dt.to_period('W').dt.start_time
                
                # Weekly volumes
                weekly_vol = df_trend.groupby('week').agg(
                    leads=('outcome', 'count'),
                    won=('outcome', 'sum')
                )
                
                if not weekly_vol.empty:
                    # Plot weekly volume
                    fig, ax = plt.subplots(figsize=(10, 5))
                    weekly_vol[['leads', 'won']].plot(kind='bar', ax=ax)
                    ax.set_ylabel('Count')
                    ax.set_xlabel('Week')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Week-over-week change
                    if len(weekly_vol) >= 2:
                        last_week = weekly_vol.iloc[-1]['leads']
                        prev_week = weekly_vol.iloc[-2]['leads']
                        wow_change = (last_week - prev_week) / prev_week if prev_week > 0 else 0
                        st.metric("Week-over-Week Change", f"{wow_change:.1%}")
                else:
                    st.info("Not enough time-based data for volume analysis")
            except Exception as e:
                st.error(f"Error creating volume trend: {str(e)}")
        else:
            st.info("Date information not available for trend analysis")

def plot_top_categories(df, col, title, min_count=10):
    """
    Plot conversion rates for top categories in a column
    
    Args:
        df (DataFrame): Dataframe with outcome data
        col (str): Column name to group by
        title (str): Title for the chart
        min_count (int): Minimum count to include a category
    """
    st.write(f"#### Conversion by {title}")
    
    if col in df.columns:
        try:
            # Fill NaN values
            s = df[col].fillna('Unknown')
            
            # Group and calculate conversion rates
            summary = (
                pd.DataFrame({'cat': s, 'outcome': df['outcome']})
                .groupby('cat')
                .agg(
                    total=('outcome', 'size'),
                    won=('outcome', 'sum')
                )
            )
            
            # Filter by minimum count
            summary = summary[summary['total'] >= min_count]
            
            if not summary.empty:
                # Calculate conversion rate
                summary['conv'] = summary['won'] / summary['total']
                
                # Calculate confidence intervals using Wilson score interval
                try:
                    from statsmodels.stats.proportion import proportion_confint
                    
                    # Apply Wilson confidence interval calculation for each row
                    summary['ci_low'], summary['ci_high'] = zip(*summary.apply(
                        lambda r: proportion_confint(r['won'], r['total'], method='wilson'),
                        axis=1
                    ))
                except ImportError:
                    # Fallback to simple standard error if statsmodels is not available
                    summary['ci_low'] = summary.apply(lambda r: max(0, r['conv'] - 1.96 * ((r['conv'] * (1 - r['conv'])) / r['total'])**0.5), axis=1)
                    summary['ci_high'] = summary.apply(lambda r: min(1, r['conv'] + 1.96 * ((r['conv'] * (1 - r['conv'])) / r['total'])**0.5), axis=1)
                
                # Calculate overall baseline for comparison
                baseline = df['outcome'].mean()
                
                # Sort and get top categories
                if isinstance(summary, pd.DataFrame) and 'conv' in summary.columns:
                    top = summary.sort_values('conv', ascending=False).head(5)
                    
                    # Prepare data for Plotly
                    plot_df = top.reset_index()
                    plot_df = plot_df.rename(columns={'cat': title, 'conv': 'Conversion Rate'})
                    
                    # Calculate error bar values (how far from the mean in each direction)
                    plot_df['error_minus'] = plot_df['Conversion Rate'] - plot_df['ci_low']
                    plot_df['error_plus'] = plot_df['ci_high'] - plot_df['Conversion Rate']
                    
                    # Apply sample size filtering
                    sample_size_min = 30  # Minimum recommended sample size for statistical analysis
                    
                    # Add annotation for low sample sizes
                    plot_df['sample_note'] = plot_df['total'].apply(
                        lambda x: f"n={x}" if x >= sample_size_min else f"n={x} (low sample)"
                    )
                    
                    # Create color-coded bar chart with Plotly
                    import plotly.express as px
                    fig = px.bar(
                        plot_df,
                        y=title,
                        x='Conversion Rate',
                        color='Conversion Rate',
                        color_continuous_scale='RdYlGn',  # Red to Green scale
                        text='sample_note',  # Include sample size annotation
                        error_x=dict(
                            type='data', 
                            symmetric=False,
                            array=plot_df['error_plus'],
                            arrayminus=plot_df['error_minus']
                        ),
                        orientation='h',
                        height=350,
                        labels={'Conversion Rate': 'Conversion Rate'}
                    )
                    
                    # Format percentages in hover text
                    fig.update_traces(
                        hovertemplate=f'<b>%{{y}}</b><br>Conversion Rate: %{{x:.1%}}<br>Sample Size: %{{text}}<extra></extra>'
                    )
                    
                    # Format x-axis as percentage
                    fig.update_layout(
                        title=f"Top {title} Conversion Rates",
                        xaxis_tickformat='.1%',
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    
                    # Add clickable data points for interactive drilldowns
                    fig.update_traces(
                        marker_line_width=1,
                        marker_line_color="white",
                        opacity=0.9,
                        hoverinfo="x+y",
                        textposition="auto",
                    )
                    
                    # Add annotation explaining interaction capability
                    fig.add_annotation(
                        text="Click on a bar to filter dashboard by that category",
                        xref="paper", yref="paper",
                        x=0, y=-0.1,
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        align="left"
                    )
                    
                    # Set up container for chart with event handling
                    chart_container = st.container()
                    
                    # Display the chart
                    with chart_container:
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # Add interactive filter selection with direct clicks
                    st.write("🔍 **Filter by category:**")
                    col_filters = st.columns(min(5, len(plot_df)))
                    for i, (idx, row) in enumerate(plot_df.iterrows()):
                        category_name = row[title]
                        with col_filters[i % 5]:
                            if st.button(f"{category_name}", key=f"filter_{title}_{i}"):
                                # Store in session state as filter
                                st.session_state.filter_values[title] = category_name
                                st.rerun()
                    
                    # Setup session state for filtering if not already present
                    if 'filter_values' not in st.session_state:
                        st.session_state.filter_values = {}
                    
                    # Create a button to clear filters if any are active
                    if title in st.session_state.filter_values:
                        if st.button(f"Clear {title} filter", key=f"clear_{title}"):
                            del st.session_state.filter_values[title]
                            st.rerun()
                    
                    # Add best/worst metrics below the chart
                    best_cat = plot_df.iloc[0]
                    worst_cat = plot_df.iloc[-1]
                    
                    # Get scalar values for formatting
                    best_name = best_cat[title]
                    best_val = float(best_cat['Conversion Rate'])
                    worst_name = worst_cat[title]
                    worst_val = float(worst_cat['Conversion Rate'])
                    
                    # Display metrics with delta comparison to baseline
                    col1, col2 = st.columns(2)
                    col1.metric(
                        "🏆 Best Performer", 
                        f"{best_name}: {best_val:.1%}",
                        delta=f"+{best_val - baseline:.1%} vs. average"
                    )
                    col2.metric(
                        "💀 Worst Performer", 
                        f"{worst_name}: {worst_val:.1%}",
                        delta=f"{worst_val - baseline:.1%} vs. average"
                    )
            else:
                st.info(f"Not enough {title} data to analyze (need at least {min_count} leads per category)")
        except Exception as e:
            st.error(f"Error analyzing {title}: {str(e)}")
    else:
        st.info(f"No {title} data available")

def plot_booking_types(df, min_count=10):
    """Plot conversion rates by booking type"""
    booking_col = None
    if 'booking_type' in df.columns:
        booking_col = 'booking_type'
    
    if booking_col:
        plot_top_categories(df, booking_col, "Booking Type", min_count)
    else:
        st.info("No Booking Type data available")

def plot_referral_marketing_sources(df):
    """Plot conversion rates by referral and marketing sources"""
    col1, col2 = st.columns(2)
    
    with col1:
        plot_top_categories(df, 'referral_source', "Referral Source")
    
    with col2:
        plot_top_categories(df, 'marketing_source', "Marketing Source")

def plot_timing_factors(df):
    """Plot conversion rates by various timing factors"""
    st.subheader("Timing Factors")
    col1, col2, col3 = st.columns(3)
    
    # Days Until Event
    with col1:
        st.write("#### Days Until Event")
        if 'due_bin' in df.columns:
            try:
                summary = df.groupby('due_bin')['outcome'].mean().fillna(0)
                
                if not summary.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    summary.plot(kind='bar', ax=ax)
                    ax.set_ylim(0, min(1, summary.max() * 1.2))
                    ax.set_ylabel("Conversion Rate")
                    
                    for i, v in enumerate(summary):
                        ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Not enough data for days until event analysis")
            except Exception as e:
                st.error(f"Error analyzing days until event: {str(e)}")
        else:
            st.info("No Days Until Event data available")
    
    # Days Since Inquiry
    with col2:
        st.write("#### Days Since Inquiry")
        if 'dsi_bin' in df.columns:
            try:
                summary = df.groupby('dsi_bin')['outcome'].mean().fillna(0)
                
                if not summary.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    summary.plot(kind='bar', ax=ax)
                    ax.set_ylim(0, min(1, summary.max() * 1.2))
                    ax.set_ylabel("Conversion Rate")
                    
                    for i, v in enumerate(summary):
                        ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Not enough data for days since inquiry analysis")
            except Exception as e:
                st.error(f"Error analyzing days since inquiry: {str(e)}")
        else:
            st.info("No Days Since Inquiry data available")
    
    # Submission Weekday
    with col3:
        st.write("#### Inquiry Weekday")
        if 'weekday' in df.columns:
            try:
                # Order by day of week
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                summary = df.groupby('weekday')['outcome'].mean().reindex(weekday_order).fillna(0)
                
                if not summary.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    summary.plot(kind='bar', ax=ax)
                    ax.set_ylim(0, min(1, summary.max() * 1.2))
                    ax.set_ylabel("Conversion Rate")
                    
                    for i, v in enumerate(summary):
                        ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Not enough data for weekday analysis")
            except Exception as e:
                st.error(f"Error analyzing weekday: {str(e)}")
        else:
            st.info("No submission weekday data available")

def plot_size_factors(df):
    """Plot conversion rates by size-related factors"""
    st.subheader("Size & Staffing Factors")
    col1, col2 = st.columns(2)
    
    # Number of Guests
    with col1:
        st.write("#### Number of Guests")
        if 'guest_bin' in df.columns:
            try:
                summary = df.groupby('guest_bin')['outcome'].mean().fillna(0)
                
                if not summary.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    summary.plot(kind='bar', ax=ax)
                    ax.set_ylim(0, min(1, summary.max() * 1.2))
                    ax.set_ylabel("Conversion Rate")
                    
                    for i, v in enumerate(summary):
                        ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Not enough data for guest size analysis")
            except Exception as e:
                st.error(f"Error analyzing guest size: {str(e)}")
        else:
            st.info("No Number of Guests data available")
    
    # Staff-to-Guest Ratio
    with col2:
        st.write("#### Staff-to-Guest Ratio")
        if 'ratio_bin' in df.columns:
            try:
                summary = df.groupby('ratio_bin')['outcome'].mean().fillna(0)
                
                if not summary.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    summary.plot(kind='bar', ax=ax)
                    ax.set_ylim(0, min(1, summary.max() * 1.2))
                    ax.set_ylabel("Conversion Rate")
                    
                    for i, v in enumerate(summary):
                        ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Not enough data for staff ratio analysis")
            except Exception as e:
                st.error(f"Error analyzing staff ratio: {str(e)}")
        else:
            st.info("Staffing data unavailable")

def plot_geographic_insights(df):
    """Plot conversion rates by geography"""
    st.subheader("Geographic Insights")
    
    if 'state' in df.columns:
        try:
            # Get states with at least 5 leads
            state_counts = df.groupby('state').size()
            valid_states = state_counts[state_counts >= 5].index
            
            if len(valid_states) > 0:
                # Filter to valid states
                df_geo = df[df['state'].isin(valid_states)]
                
                # Calculate conversion rates
                rates = df_geo.groupby('state')['outcome'].mean().sort_values(ascending=False)
                
                if not rates.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    rates.plot(kind='bar', ax=ax)
                    ax.set_ylim(0, min(1, rates.max() * 1.2))
                    ax.set_ylabel("Conversion Rate")
                    
                    for i, v in enumerate(rates):
                        ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Not enough data for geographic analysis")
            else:
                st.info("Not enough data per state for meaningful analysis (need at least 5 leads)")
        except Exception as e:
            st.error(f"Error analyzing geography: {str(e)}")
    else:
        st.info("No State/Region data available")

def show_data_quality(df):
    """Show data quality metrics and anomalies"""
    st.subheader("Data Quality Assessment")
    
    # Calculate missing percentages
    miss = df.isna().mean().mul(100).round(1).reset_index()
    miss.columns = ['Field', '% Missing']
    miss = miss.sort_values('% Missing', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("#### Missing Data")
        st.table(miss)
    
    with col2:
        st.write("#### Data Anomalies")
        
        # Check for negative days until event
        if 'days_until_event' in df.columns:
            anomalies = df[df['days_until_event'] < 0].shape[0]
            if anomalies > 0:
                st.warning(f"{anomalies} leads have negative Days Until Event (data error)")
            else:
                st.success("No negative Days Until Event values found")
        
        # Check for extreme values in number of guests
        if 'number_of_guests' in df.columns:
            extreme_guests = df[df['number_of_guests'] > 1000].shape[0]
            if extreme_guests > 0:
                st.warning(f"{extreme_guests} leads have more than 1,000 guests (potential data error)")
            else:
                st.success("No extreme guest count values found")

def run_conversion_analysis(df_original):
    """
    Run the full conversion analysis dashboard with the provided dataframe
    
    Args:
        df_original (DataFrame): Original dataframe to analyze
    """
    # Initialize session state for filter values if not present
    if 'filter_values' not in st.session_state:
        st.session_state.filter_values = {}
    
    # Display active filters if any
    active_filters = st.session_state.filter_values
    if active_filters:
        st.write("#### Active Chart Filters")
        filter_cols = st.columns(min(len(active_filters), 3))
        for i, (key, value) in enumerate(active_filters.items()):
            with filter_cols[i % 3]:
                st.info(f"**{key}:** {value}")
        
        # Add a button to clear all filters
        if st.button("Clear All Chart Filters"):
            st.session_state.filter_values = {}
            st.rerun()
    
    # Clean and normalize data
    df = normalize_data(df_original)
    
    # 1. Setup and apply UI filters
    filters = setup_filters(df)
    filtered_df = apply_filters(df, filters)
    
    # Apply any click-based filters from interactive charts
    for filter_col, filter_value in st.session_state.filter_values.items():
        if filter_col in filtered_df.columns:
            # Add additional filters based on chart interactions
            mask = filtered_df[filter_col].astype(str) == str(filter_value)
            filtered_df = filtered_df[mask]
            
    # If no data after filtering, show a warning
    if filtered_df.empty:
        st.warning("No data matches all active filters. Try removing some filters.")
        if st.button("Reset All Filters"):
            st.session_state.filter_values = {}
            st.rerun()
        return
    
    # 2. Display KPI Summary
    display_kpi_summary(filtered_df)
    
    # Show data quality metrics
    with st.expander("Data Quality Analysis"):
        show_data_quality(filtered_df)
    
    # 3. Time trends
    st.subheader("Conversion Trends")
    plot_time_trends(filtered_df)
    
    # 4. Category analysis
    st.subheader("Conversion by Category")
    
    # Booking types
    plot_booking_types(filtered_df)
    
    # Referral and marketing sources
    plot_referral_marketing_sources(filtered_df)
    
    # 5. Timing factors analysis
    plot_timing_factors(filtered_df)
    
    # 6. Size and staffing analysis
    plot_size_factors(filtered_df)
    
    # 7. Geographic insights
    plot_geographic_insights(filtered_df)
    
    return filtered_df