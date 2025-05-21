"""
conversion_tab.py - Conversion Analysis Tab Module

This module provides the implementation for the Conversion Analysis tab
in the Sales Conversion Analytics Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from conversion import analyze_phone_matches
from advanced_analytics import plot_conversion_by_category
# Import improved visualizations
from improved_visualizations import (
    plot_conversion_by_booking_type,
    plot_conversion_by_referral_source,
    plot_deal_value_analysis,
    plot_data_completeness
)

def render_conversion_tab(df):
    """
    Render the complete Conversion Analysis tab
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    """
    if df is None or df.empty:
        st.warning("No data available. Please load data first.")
        return
    
    st.markdown("## Conversion Analysis")
    st.markdown("*Showing insights based on all available data*")
    
    # Display key metrics
    display_kpi_summary(df)
    
    # Main conversion insights
    col1, col2 = st.columns(2)
    
    with col1:
        # Check if booking_type or event_type data is available
        has_booking_data = 'clean_booking_type' in df.columns
        has_event_data = 'event_type' in df.columns
        
        if has_booking_data or has_event_data:
            if has_booking_data:
                st.markdown("### Conversion by Booking Type")
            elif has_event_data:
                st.markdown("### Conversion by Event Type")
            
            if 'outcome' in df.columns:
                plot_booking_types(df)
            else:
                st.info("Outcome data not available for conversion analysis")
        else:
            st.markdown("### Conversion by Booking Type")
            st.info("Booking type/Event type data not available")
        
        st.markdown("### Conversion by Time Factors")
        plot_timing_factors(df)
    
    with col2:
        st.markdown("### Conversion by Referral Source")
        if 'referral_source' in df.columns and 'outcome' in df.columns:
            plot_referral_sources(df)
        else:
            st.info("Referral source data not available")
        
        st.markdown("### Conversion by Event Size")
        plot_size_factors(df)
    
    # Geographic insights
    st.markdown("### Geographic Insights")
    plot_geographic_insights(df)
    
    # Phone matching analysis
    st.markdown("### Contact Matching Analysis")
    phone_analysis = analyze_phone_matches(df)
    
    if phone_analysis:
        match_rates, match_counts = phone_analysis
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Conversion by Area Code Match")
            st.dataframe(match_rates)
        with col2:
            st.markdown("#### Area Code Distribution")
            st.dataframe(match_counts)
            
            # Show the best and worst performing area codes
            if 'area_code' in match_rates.columns and not match_rates.empty:
                match_rates = match_rates.sort_values('conversion_rate', ascending=False)
                
                best_area = match_rates.iloc[0]
                worst_area = match_rates.iloc[-1]
                
                st.markdown("**Best performing area code:**")
                st.markdown(f"Area code {best_area['area_code']} - {best_area['conversion_rate']:.1%} conversion rate")
                
                st.markdown("**Worst performing area code:**")
                st.markdown(f"Area code {worst_area['area_code']} - {worst_area['conversion_rate']:.1%} conversion rate")
    else:
        st.info("Phone matching analysis data not available")
    
    # Data quality metrics
    st.markdown("### Data Quality")
    show_data_quality(df)


def display_kpi_summary(df):
    """Display key performance indicators in a row of metric cards"""
    if df is None or df.empty:
        return
    
    # Calculate key metrics
    total_leads = len(df)
    won_leads = df['won'].sum() if 'won' in df.columns else 0
    conversion_rate = won_leads / total_leads if total_leads > 0 else 0
    
    # For average deal value, ensure we have the necessary columns
    avg_deal_value = 0
    if 'actual_deal_value' in df.columns:
        # Only consider rows with deal values for won deals
        if 'won' in df.columns:
            won_deals = df[df['won'] == True]
            deal_values = won_deals.loc[won_deals['actual_deal_value'].notna() & (won_deals['actual_deal_value'] > 0), 'actual_deal_value']
            avg_deal_value = deal_values.mean() if not deal_values.empty else 0
        else:
            # For all deals if won status isn't available
            deal_values = df.loc[df['actual_deal_value'].notna() & (df['actual_deal_value'] > 0), 'actual_deal_value']
            avg_deal_value = deal_values.mean() if not deal_values.empty else 0
        
        # Log diagnostics for debugging
        st.session_state['deal_value_count'] = len(deal_values) if not deal_values.empty else 0
    
    # Calculate average time to close
    avg_days_to_close = 0
    if 'days_since_inquiry' in df.columns:
        won_days = df.loc[df['won'] == True, 'days_since_inquiry'] if 'won' in df.columns else []
        avg_days_to_close = won_days.mean() if len(won_days) > 0 else 0
    
    # Create a row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Leads", f"{total_leads:,}")
    
    with col2:
        st.metric("Won Deals", f"{int(won_leads):,}")
    
    with col3:
        st.metric("Conversion Rate", f"{conversion_rate:.1%}")
    
    with col4:
        st.metric("Avg. Deal Value", f"${avg_deal_value:,.2f}" if avg_deal_value > 0 else "N/A")


def plot_booking_types(df, min_count=5):
    """Plot conversion rates by booking type or event type using an interactive Altair chart"""
    from improved_visualizations import plot_conversion_by_booking_type
    
    # Create a clean copy to work with
    clean_df = df.copy()
    
    # Clean null values
    if 'booking_type' in clean_df.columns:
        clean_df = clean_df.dropna(subset=['booking_type', 'outcome'])
        
        # Group sparse categories (fewer than 20 leads)
        counts = clean_df['booking_type'].value_counts()
        sparse_categories = counts[counts < 20].index.tolist()
        if sparse_categories:
            clean_df['booking_type'] = clean_df['booking_type'].apply(
                lambda x: 'Other Types' if x in sparse_categories else x
            )
        
        # Use our improved visualization with better error handling
        plot_conversion_by_booking_type(clean_df)
    else:
        st.warning("Booking type data not available")


def plot_referral_sources(df, min_count=5):
    """Plot conversion rates by referral source using an interactive Altair chart"""
    from improved_visualizations import plot_conversion_by_referral_source
    
    # Create a clean copy to work with
    clean_df = df.copy()
    
    # Clean null values
    if 'referral_source' in clean_df.columns:
        clean_df = clean_df.dropna(subset=['referral_source', 'outcome'])
        
        # Group sparse categories (fewer than 20 leads)
        counts = clean_df['referral_source'].value_counts()
        sparse_categories = counts[counts < 20].index.tolist()
        if sparse_categories:
            clean_df['referral_source'] = clean_df['referral_source'].apply(
                lambda x: 'Other Sources' if x in sparse_categories else x
            )
        
        # Use our improved visualization with better error handling
        plot_conversion_by_referral_source(clean_df)
    else:
        st.warning("Referral source data not available")


def plot_timing_factors(df):
    """Plot conversion rates by various timing factors"""
    if df is None or df.empty:
        st.info("No data available for timing analysis")
        return
    
    # Create a clean copy to work with
    clean_df = df.copy()
    
    # Check if we have the necessary columns
    # Support multiple column name variations
    has_days_until = (('days_until_event' in clean_df.columns or 'days_until' in clean_df.columns) 
                      and 'outcome' in clean_df.columns)
    has_days_since = (('days_since_inquiry' in clean_df.columns or 'days_since' in clean_df.columns) 
                       and 'outcome' in clean_df.columns)
    has_weekday = 'inquiry_date' in clean_df.columns and 'outcome' in clean_df.columns
    
    if not (has_days_until or has_days_since or has_weekday):
        st.info("Timing data not available. Missing days until event, days since inquiry, or inquiry date columns.")
        return
    
    # Standardize column names for consistency
    if 'days_until' in clean_df.columns and 'days_until_event' not in clean_df.columns:
        clean_df['days_until_event'] = clean_df['days_until']
    if 'days_since' in clean_df.columns and 'days_since_inquiry' not in clean_df.columns:
        clean_df['days_since_inquiry'] = clean_df['days_since']
    
    # Count valid entries for each time factor
    valid_counts = {}
    if has_days_until:
        days_until_col = 'days_until_event' if 'days_until_event' in clean_df.columns else 'days_until'
        valid_counts['days_until'] = clean_df[clean_df[days_until_col].notna() & 
                                            np.isfinite(clean_df[days_until_col])].shape[0]
    if has_days_since:
        days_since_col = 'days_since_inquiry' if 'days_since_inquiry' in clean_df.columns else 'days_since'
        valid_counts['days_since'] = clean_df[clean_df[days_since_col].notna() & 
                                            np.isfinite(clean_df[days_since_col])].shape[0]
    if has_weekday:
        valid_counts['weekday'] = clean_df[clean_df['inquiry_date'].notna()].shape[0]
        
    # Show overview of available data (only if we have data to show)
    if valid_counts:
        st.markdown("**Available Time Factor Data:**")
        for factor, count in valid_counts.items():
            st.markdown(f"- {factor}: {count} valid entries")
    
    # Require at least 5 valid entries for meaningful analysis
    min_required = 5
    
    # Days until event analysis
    if has_days_until:
        st.markdown("#### Conversion by Days Until Event")
        try:
            # Make a copy and filter out NaN/Inf values
            valid_df = df[df['days_until_event'].notna() & np.isfinite(df['days_until_event'])].copy()
            
            if len(valid_df) > 0:
                if 'days_until_bin' in df.columns:
                    # Use the pre-binned column
                    plot_conversion_by_category(valid_df, 'days_until_bin', 'Days Until Event', sort_by='natural')
                else:
                    # Create bins manually
                    bins = [0, 30, 90, 180, 365, float('inf')]
                    labels = ['< 30 days', '30-90 days', '90-180 days', '180-365 days', '365+ days']
                    valid_df['days_until_bin'] = pd.cut(valid_df['days_until_event'], bins=bins, labels=labels)
                    
                    # Additional check to ensure no NaN values in the binned column
                    valid_df = valid_df.dropna(subset=['days_until_bin'])
                    if len(valid_df) > 0:
                        plot_conversion_by_category(valid_df, 'days_until_bin', 'Days Until Event', sort_by='natural')
                    else:
                        st.info("Not enough valid data for days until event analysis")
            else:
                st.info("Not enough valid data for days until event analysis")
        except Exception as e:
            st.error(f"Error in days until event analysis: {str(e)}")
    
    # Days since inquiry analysis
    if has_days_since:
        st.markdown("#### Conversion by Days Since Inquiry")
        try:
            # Filter out any NaN or infinite values
            valid_df = df[df['days_since_inquiry'].notna() & np.isfinite(df['days_since_inquiry'])].copy()
            
            if len(valid_df) > 0:
                bins = [0, 7, 14, 30, 60, float('inf')]
                labels = ['< 7 days', '7-14 days', '14-30 days', '30-60 days', '60+ days']
                valid_df['days_since_bin'] = pd.cut(valid_df['days_since_inquiry'], bins=bins, labels=labels)
                
                # Additional check to ensure no NaN values in the binned column
                valid_df = valid_df.dropna(subset=['days_since_bin'])
                if len(valid_df) > 0:
                    plot_conversion_by_category(valid_df, 'days_since_bin', 'Days Since Inquiry', sort_by='natural')
                else:
                    st.info("Not enough valid data for days since inquiry analysis")
            else:
                st.info("Not enough valid data for days since inquiry analysis")
        except Exception as e:
            st.error(f"Error in days since inquiry analysis: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    # Weekday analysis
    if has_weekday:
        st.markdown("#### Conversion by Day of Week")
        try:
            # Filter out any NaN values in inquiry_date
            valid_df = df.dropna(subset=['inquiry_date']).copy()
            
            if len(valid_df) > 0:
                valid_df['weekday'] = valid_df['inquiry_date'].dt.day_name()
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Check if any weekday values were created
                if valid_df['weekday'].notna().any():
                    plot_conversion_by_category(valid_df, 'weekday', 'Day of Week', sort_by='custom', custom_order=weekday_order)
                else:
                    st.info("Could not determine weekdays from inquiry dates")
            else:
                st.info("Not enough valid inquiry dates for weekday analysis")
        except Exception as e:
            st.error(f"Error in weekday analysis: {str(e)}")
            import traceback
            st.text(traceback.format_exc())


def plot_size_factors(df):
    """Plot conversion rates by size-related factors"""
    if df is None or df.empty:
        return
    
    # Check if we have the necessary columns
    has_guests = 'number_of_guests' in df.columns and 'outcome' in df.columns
    has_bartenders = 'bartenders_needed' in df.columns and 'outcome' in df.columns
    
    if not (has_guests or has_bartenders):
        st.info("Size data not available")
        return
    
    # Guest count analysis
    if has_guests:
        st.markdown("#### Conversion by Guest Count")
        try:
            # Filter out any NaN or invalid guest counts
            valid_df = df[df['number_of_guests'].notna() & (df['number_of_guests'] > 0)].copy()
            
            if len(valid_df) >= 5:  # Require at least 5 valid guest counts
                if 'guests_bin' in df.columns:
                    # Use the pre-binned column if it exists
                    plot_conversion_by_category(valid_df, 'guests_bin', 'Number of Guests', sort_by='natural')
                else:
                    # Create bins manually
                    bins = [0, 50, 100, 150, 200, 300, float('inf')]
                    labels = ['< 50', '50-100', '100-150', '150-200', '200-300', '300+']
                    
                    # Ensure guests are numeric and create bins
                    valid_df['number_of_guests'] = pd.to_numeric(valid_df['number_of_guests'], errors='coerce')
                    valid_df = valid_df[valid_df['number_of_guests'].notna()].copy()
                    
                    if len(valid_df) >= 5:
                        valid_df['guests_bin'] = pd.cut(valid_df['number_of_guests'], bins=bins, labels=labels)
                        
                        # Check if binning worked properly
                        if valid_df['guests_bin'].notna().sum() >= 5:
                            plot_conversion_by_category(valid_df, 'guests_bin', 'Number of Guests', sort_by='natural')
                            
                            # Add data counts for transparency
                            guest_counts = valid_df.groupby('guests_bin').size().reset_index(name='count')
                            st.markdown("**Guest count distribution:**")
                            for _, row in guest_counts.iterrows():
                                st.markdown(f"- {row['guests_bin']}: {row['count']} leads")
                        else:
                            st.info("Data binning failed - check that guest counts are valid numbers")
                    else:
                        st.info(f"Not enough valid guest counts (found {len(valid_df)})")
            else:
                st.info(f"Not enough valid guest counts for analysis (found {len(valid_df)})")
        except Exception as e:
            st.error(f"Error in guest count analysis: {str(e)}")
            st.info("Try checking guest count data format or adjust data quality filters")
    
    # Bartender count analysis
    if has_bartenders:
        st.markdown("#### Conversion by Bartenders Needed")
        try:
            # Filter out any NaN or invalid bartender counts
            valid_df = df[df['bartenders_needed'].notna() & (df['bartenders_needed'] >= 0)].copy()
            
            if len(valid_df) >= 5:  # Require at least 5 valid counts
                # Convert to numeric to be sure
                valid_df['bartenders_needed'] = pd.to_numeric(valid_df['bartenders_needed'], errors='coerce')
                valid_df = valid_df[valid_df['bartenders_needed'].notna()].copy()
                
                if len(valid_df) >= 5:
                    # Create bins for bartenders needed
                    bins = [0, 1, 2, 3, 5, float('inf')]
                    labels = ['0', '1', '2', '3-5', '5+']
                    
                    # Create a new column with binned bartenders needed
                    valid_df['bartenders_bin'] = pd.cut(
                        valid_df['bartenders_needed'], 
                        bins=bins, 
                        labels=labels
                    )
                    
                    # Check if binning worked properly
                    if valid_df['bartenders_bin'].notna().sum() >= 5:
                        # Plot the data
                        plot_conversion_by_category(valid_df, 'bartenders_bin', 'Bartenders Needed', sort_by='natural')
                        
                        # Add data counts for transparency
                        bartender_counts = valid_df.groupby('bartenders_bin').size().reset_index(name='count')
                        st.markdown("**Bartender count distribution:**")
                        for _, row in bartender_counts.iterrows():
                            st.markdown(f"- {row['bartenders_bin']}: {row['count']} leads")
                    else:
                        st.info("Data binning failed - check that bartender counts are valid numbers")
                else:
                    st.info(f"Not enough valid bartender counts (found {len(valid_df)})")
            else:
                st.info(f"Not enough valid bartender count data for analysis (found {len(valid_df)})")
        except Exception as e:
            st.error(f"Error in bartender count analysis: {str(e)}")
            st.info("Try checking bartender count data format or adjust data quality filters")


def plot_geographic_insights(df):
    """Plot conversion rates by geography"""
    if df is None or df.empty:
        return
    
    # Check if we have the necessary columns
    has_state = 'state' in df.columns and 'outcome' in df.columns
    has_city = 'city' in df.columns and 'outcome' in df.columns
    
    if not (has_state or has_city):
        st.info("Geographic data not available")
        return
    
    col1, col2 = st.columns(2)
    
    # State analysis
    if has_state:
        with col1:
            st.markdown("#### Conversion by State")
            plot_conversion_by_category(df, 'state', 'State', top_n=10)
    
    # City analysis
    if has_city:
        with col2:
            st.markdown("#### Conversion by City")
            plot_conversion_by_category(df, 'city', 'City', top_n=10)


def show_data_quality(df):
    """Show data quality metrics and anomalies using interactive visualizations"""
    if df is None or df.empty:
        st.info("No data available for quality analysis")
        return
    
    st.markdown("### Data Health Check")
    st.markdown("This section helps diagnose why some charts might be missing or incomplete")
    
    # Map columns to the charts they affect
    chart_dependencies = {
        'inquiry_date': ['Days Since Inquiry', 'Day of Week'],
        'event_date': ['Days Until Event', 'Event Month'],
        'booking_type': ['Booking Type Conversion'],
        'event_type': ['Event Type Conversion'],
        'number_of_guests': ['Guest Count Analysis'],
        'days_until_event': ['Days Until Event Analysis'],
        'days_since_inquiry': ['Days Since Inquiry Analysis'],
        'city': ['Geographic Analysis'],
        'state': ['Geographic Analysis'],
        'referral_source': ['Referral Source Analysis'],
        'marketing_source': ['Marketing Source Analysis'],
        'bartenders_needed': ['Bartender Analysis'],
        'actual_deal_value': ['Deal Value Analysis']
    }
    
    # Use our improved Altair-based data completeness visualization
    completeness_df = plot_data_completeness(df, chart_dependencies)
    
    # The plot_data_completeness function handles the visualization and explanation of data quality issues
    
    # Show additional data quality insights section
    st.markdown("### Missing Data Impact on Charts")
    
    # Fields affecting key charts
    critical_fields = completeness_df[completeness_df['Status'].isin(['Critical', 'Poor'])]
    if not critical_fields.empty:
        st.warning("⚠️ The following charts may be missing or incomplete due to data quality issues:")
        
        # Group by affected charts to show what's impacted
        affected_charts = {}
        for _, row in critical_fields.iterrows():
            for chart in row['Affects Charts'].split(', '):
                if chart not in affected_charts:
                    affected_charts[chart] = []
                affected_charts[chart].append((row['Column'], row['Completeness']))
        
        # Display affected charts and their missing fields
        for chart, fields in affected_charts.items():
            st.markdown(f"**{chart}**: Missing data in {', '.join([f[0] for f in fields])}")
        
        # Add suggestions
        st.markdown("### 🛠️ Suggestions to Fix")
        st.markdown("""
        1. **Adjust filters**: Try setting the 'Minimum Data Completeness' slider to 0 to include all leads
        2. **Check data import**: Ensure CSV files have all required columns properly formatted
        3. **Add data**: For testing, you can upload more complete sample data
        """)
    else:
        st.success("✅ Data quality is good. All charts should display properly.")