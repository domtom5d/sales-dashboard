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
        st.markdown("### Conversion by Booking Type")
        if 'clean_booking_type' in df.columns and 'outcome' in df.columns:
            plot_booking_types(df)
        else:
            st.info("Booking type data not available")
        
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
        # Only consider rows with deal values
        deal_values = df.loc[df['actual_deal_value'].notna(), 'actual_deal_value']
        avg_deal_value = deal_values.mean() if not deal_values.empty else 0
    
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
    """Plot conversion rates by booking type"""
    if 'clean_booking_type' not in df.columns or 'outcome' in df.columns:
        st.info("Booking type data not available")
        return
    
    # Group by booking type and calculate conversion rates
    booking_type_data = df.groupby('clean_booking_type').agg(
        won=('won', 'sum'),
        total=('outcome', 'count'),
    ).reset_index()
    
    # Calculate conversion rate
    booking_type_data['conversion_rate'] = booking_type_data['won'] / booking_type_data['total']
    
    # Filter out booking types with fewer than min_count leads
    booking_type_data = booking_type_data[booking_type_data['total'] >= min_count]
    
    # Sort by conversion rate and get top booking types
    top_booking_types = booking_type_data.sort_values('conversion_rate', ascending=False)
    
    if not top_booking_types.empty:
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the data
        sns.barplot(
            x='conversion_rate', 
            y='clean_booking_type', 
            data=top_booking_types,
            ax=ax
        )
        
        # Customize the plot
        ax.set_xlabel('Conversion Rate')
        ax.set_ylabel('Booking Type')
        ax.set_title('Conversion Rate by Booking Type')
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # Add the percentage labels to the end of each bar
        for i, row in enumerate(top_booking_types.itertuples()):
            ax.text(
                row.conversion_rate + 0.01, 
                i, 
                f'{row.conversion_rate:.1%} ({int(row.won)}/{int(row.total)})',
                va='center'
            )
        
        # Show the plot
        st.pyplot(fig)
        
        # Show the best and worst booking types with insights
        best_type = top_booking_types.iloc[0]
        worst_type = top_booking_types.iloc[-1]
        
        st.markdown("**Insights:**")
        st.markdown(f"- **{best_type['clean_booking_type']}** has the highest conversion rate at {best_type['conversion_rate']:.1%}")
        st.markdown(f"- **{worst_type['clean_booking_type']}** has the lowest conversion rate at {worst_type['conversion_rate']:.1%}")
    else:
        st.info("Not enough booking type data available")


def plot_referral_sources(df, min_count=5):
    """Plot conversion rates by referral source"""
    if 'referral_source' not in df.columns or 'outcome' not in df.columns:
        st.info("Referral source data not available")
        return
    
    # Group by referral source and calculate conversion rates
    referral_data = df.groupby('referral_source').agg(
        won=('won', 'sum'),
        total=('outcome', 'count'),
    ).reset_index()
    
    # Calculate conversion rate
    referral_data['conversion_rate'] = referral_data['won'] / referral_data['total']
    
    # Filter out referral sources with fewer than min_count leads
    referral_data = referral_data[referral_data['total'] >= min_count]
    
    # Sort by conversion rate and get top referral sources
    top_referrals = referral_data.sort_values('conversion_rate', ascending=False).head(10)
    
    if not top_referrals.empty:
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the data
        sns.barplot(
            x='conversion_rate', 
            y='referral_source', 
            data=top_referrals,
            ax=ax
        )
        
        # Customize the plot
        ax.set_xlabel('Conversion Rate')
        ax.set_ylabel('Referral Source')
        ax.set_title('Conversion Rate by Referral Source')
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # Add the percentage labels to the end of each bar
        for i, row in enumerate(top_referrals.itertuples()):
            ax.text(
                row.conversion_rate + 0.01, 
                i, 
                f'{row.conversion_rate:.1%} ({int(row.won)}/{int(row.total)})',
                va='center'
            )
        
        # Show the plot
        st.pyplot(fig)
        
        # Show the best and worst referral sources with insights
        best_referral = top_referrals.iloc[0]
        worst_referral = top_referrals.iloc[-1]
        
        st.markdown("**Insights:**")
        st.markdown(f"- **{best_referral['referral_source']}** has the highest conversion rate at {best_referral['conversion_rate']:.1%}")
        st.markdown(f"- **{worst_referral['referral_source']}** has the lowest conversion rate at {worst_referral['conversion_rate']:.1%}")
    else:
        st.info("Not enough referral source data available")


def plot_timing_factors(df):
    """Plot conversion rates by various timing factors"""
    if df is None or df.empty:
        return
    
    # Check if we have the necessary columns
    has_days_until = 'days_until_event' in df.columns and 'outcome' in df.columns
    has_days_since = 'days_since_inquiry' in df.columns and 'outcome' in df.columns
    has_weekday = 'inquiry_date' in df.columns and 'outcome' in df.columns
    
    if not (has_days_until or has_days_since or has_weekday):
        st.info("Timing data not available")
        return
    
    # Days until event analysis
    if has_days_until:
        st.markdown("#### Conversion by Days Until Event")
        if 'days_until_bin' in df.columns:
            # Use the pre-binned column
            plot_conversion_by_category(df, 'days_until_bin', 'Days Until Event', sort_by='natural')
        else:
            # Create bins manually
            bins = [0, 30, 90, 180, 365, float('inf')]
            labels = ['< 30 days', '30-90 days', '90-180 days', '180-365 days', '365+ days']
            df['days_until_bin'] = pd.cut(df['days_until_event'], bins=bins, labels=labels)
            plot_conversion_by_category(df, 'days_until_bin', 'Days Until Event', sort_by='natural')
    
    # Days since inquiry analysis
    if has_days_since:
        st.markdown("#### Conversion by Days Since Inquiry")
        bins = [0, 7, 14, 30, 60, float('inf')]
        labels = ['< 7 days', '7-14 days', '14-30 days', '30-60 days', '60+ days']
        df['days_since_bin'] = pd.cut(df['days_since_inquiry'], bins=bins, labels=labels)
        plot_conversion_by_category(df, 'days_since_bin', 'Days Since Inquiry', sort_by='natural')
    
    # Weekday analysis
    if has_weekday:
        st.markdown("#### Conversion by Day of Week")
        df['weekday'] = df['inquiry_date'].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        plot_conversion_by_category(df, 'weekday', 'Day of Week', sort_by='custom', custom_order=weekday_order)


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
        if 'guests_bin' in df.columns:
            # Use the pre-binned column
            plot_conversion_by_category(df, 'guests_bin', 'Number of Guests', sort_by='natural')
        else:
            # Create bins manually
            bins = [0, 50, 100, 150, 200, 300, float('inf')]
            labels = ['< 50', '50-100', '100-150', '150-200', '200-300', '300+']
            df['guests_bin'] = pd.cut(df['number_of_guests'], bins=bins, labels=labels)
            plot_conversion_by_category(df, 'guests_bin', 'Number of Guests', sort_by='natural')
    
    # Bartender count analysis
    if has_bartenders:
        st.markdown("#### Conversion by Bartenders Needed")
        bins = [0, 1, 2, 3, 5, float('inf')]
        labels = ['0', '1', '2', '3-5', '5+']
        df['bartenders_bin'] = pd.cut(df['bartenders_needed'], bins=bins, labels=labels)
        plot_conversion_by_category(df, 'bartenders_bin', 'Bartenders Needed', sort_by='natural')


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
    """Show data quality metrics and anomalies"""
    if df is None or df.empty:
        return
    
    st.markdown("#### Data Completeness")
    
    # Calculate completeness for key columns
    key_columns = [
        'inquiry_date', 'event_date', 'booking_type', 'number_of_guests',
        'days_until_event', 'city', 'state', 'referral_source'
    ]
    
    completeness = {}
    for col in key_columns:
        if col in df.columns:
            non_null = df[col].notna().sum()
            completeness[col] = non_null / len(df)
    
    # Create a completeness dataframe
    completeness_df = pd.DataFrame({
        'Column': list(completeness.keys()),
        'Completeness': list(completeness.values())
    })
    
    # Sort by completeness
    completeness_df = completeness_df.sort_values('Completeness', ascending=False)
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    sns.barplot(
        x='Completeness', 
        y='Column', 
        data=completeness_df,
        ax=ax
    )
    
    # Customize the plot
    ax.set_xlabel('Data Completeness')
    ax.set_ylabel('Data Field')
    ax.set_title('Data Completeness by Field')
    
    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # Add the percentage labels to the end of each bar
    for i, row in enumerate(completeness_df.itertuples()):
        ax.text(
            row.Completeness + 0.01, 
            i, 
            f'{row.Completeness:.1%}',
            va='center'
        )
    
    # Show the plot
    st.pyplot(fig)
    
    # Show data insights
    st.markdown("**Data Quality Insights:**")
    
    # Find columns with low completeness (< 80%)
    low_completeness = completeness_df[completeness_df['Completeness'] < 0.8]
    if not low_completeness.empty:
        st.markdown("Fields with low data completeness (potential data quality issues):")
        for _, row in low_completeness.iterrows():
            st.markdown(f"- **{row['Column']}**: {row['Completeness']:.1%} complete")
    else:
        st.markdown("- All fields have good data completeness (>= 80%)")