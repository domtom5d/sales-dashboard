"""
deal_value_insights.py - Deal Value Analysis Features

This module provides functions to analyze deal values, revenue trends,
and price-per-guest metrics for the conversion analytics dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def analyze_deal_values_by_category(df, category_col='booking_type'):
    """
    Analyze average deal values by a category column (e.g., booking type)
    
    Args:
        df (DataFrame): Processed dataframe with outcome and deal value data
        category_col (str): Column name to group by (default: 'booking_type')
    
    Returns:
        DataFrame: Summary of average deal values by category
    """
    # Ensure deal value column exists and is numeric
    if 'actual_deal_value' not in df.columns:
        return None
    
    # Convert to numeric and handle errors
    df['actual_deal_value'] = pd.to_numeric(df['actual_deal_value'], errors='coerce')
    
    # Check if we have the category column
    if category_col not in df.columns:
        return None
    
    # Group by category and calculate average deal value
    category_values = df.groupby(category_col)['actual_deal_value'].agg(
        average=('mean'),
        total=('sum'),
        count=('count')
    ).reset_index()
    
    # Sort by average value
    category_values = category_values.sort_values('average', ascending=False)
    
    return category_values

def analyze_revenue_over_time(df, date_col='event_date', freq='M'):
    """
    Analyze total revenue over time periods
    
    Args:
        df (DataFrame): Processed dataframe with outcome and deal value data
        date_col (str): Column name with date information (default: 'event_date')
        freq (str): Frequency for grouping ('M'=month, 'W'=week, etc.)
    
    Returns:
        DataFrame: Summary of revenue over time periods
    """
    # Ensure deal value column exists and is numeric
    if 'actual_deal_value' not in df.columns:
        return None
    
    # Convert to numeric and handle errors
    df['actual_deal_value'] = pd.to_numeric(df['actual_deal_value'], errors='coerce')
    
    # Check if we have the date column
    if date_col not in df.columns:
        return None
    
    # Create datetime column
    date_col_dt = f"{date_col}_dt"
    df[date_col_dt] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Create period column for grouping
    period_col = f"{date_col}_period"
    if freq == 'M':
        df[period_col] = df[date_col_dt].dt.to_period('M').dt.to_timestamp()
    elif freq == 'W':
        df[period_col] = df[date_col_dt].dt.to_period('W').dt.to_timestamp()
    elif freq == 'Q':
        df[period_col] = df[date_col_dt].dt.to_period('Q').dt.to_timestamp()
    else:
        df[period_col] = df[date_col_dt].dt.to_period(freq).dt.to_timestamp()
    
    # Group by period and calculate revenue
    revenue_by_period = df.dropna(subset=[period_col, 'actual_deal_value']).groupby(period_col).agg(
        total_revenue=('actual_deal_value', 'sum'),
        deal_count=('actual_deal_value', 'count'),
        avg_deal_value=('actual_deal_value', 'mean')
    ).reset_index()
    
    # Sort by period
    revenue_by_period = revenue_by_period.sort_values(period_col)
    
    return revenue_by_period

def analyze_price_per_guest(df):
    """
    Analyze price per guest metrics
    
    Args:
        df (DataFrame): Processed dataframe with outcome, deal value, and guest data
    
    Returns:
        dict: Dictionary containing price per guest analysis
        {
            'overall': DataFrame with overall stats,
            'by_time': DataFrame with time-based trends (if available),
            'by_category': DataFrame with category-based differences (if available)
        }
    """
    # Ensure we have the required columns
    if not all(col in df.columns for col in ['actual_deal_value', 'number_of_guests']):
        return None
    
    # Convert to numeric and handle errors
    df['actual_deal_value'] = pd.to_numeric(df['actual_deal_value'], errors='coerce')
    df['number_of_guests'] = pd.to_numeric(df['number_of_guests'], errors='coerce')
    
    # Calculate price per guest
    df['price_per_guest'] = df['actual_deal_value'] / df['number_of_guests'].replace(0, np.nan)
    
    # Overall stats
    overall_stats = df['price_per_guest'].describe().to_frame()
    
    results = {
        'overall': overall_stats
    }
    
    # Time-based trends if event_date is available
    if 'event_date' in df.columns:
        df['event_date_dt'] = pd.to_datetime(df['event_date'], errors='coerce')
        df['month'] = df['event_date_dt'].dt.to_period('M').dt.to_timestamp()
        
        ppg_by_month = df.dropna(subset=['month', 'price_per_guest']).groupby('month').agg(
            avg_price_per_guest=('price_per_guest', 'mean'),
            median_price_per_guest=('price_per_guest', 'median'),
            count=('price_per_guest', 'count')
        ).reset_index()
        
        results['by_time'] = ppg_by_month.sort_values('month')
    
    # Category-based analysis if booking_type is available
    if 'booking_type' in df.columns:
        ppg_by_category = df.groupby('booking_type').agg(
            avg_price_per_guest=('price_per_guest', 'mean'),
            median_price_per_guest=('price_per_guest', 'median'),
            count=('price_per_guest', 'count')
        ).reset_index()
        
        results['by_category'] = ppg_by_category.sort_values('avg_price_per_guest', ascending=False)
    
    return results

def display_deal_value_insights(df):
    """Display deal value insights in the dashboard"""
    
    st.subheader("Deal Value Insights")
    
    # Deal value columns
    value_col1, value_col2 = st.columns(2)
    
    with value_col1:
        # 1. Average Deal Value by Booking Type
        st.write("#### Average Deal Value by Booking Type")
        
        category_values = analyze_deal_values_by_category(df, 'booking_type')
        
        if category_values is not None and not category_values.empty:
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(category_values['booking_type'], 
                           category_values['average'], 
                           color='#2980b9')
            
            # Add value labels
            for i, (_, row) in enumerate(category_values.iterrows()):
                ax.text(row['average'] + (ax.get_xlim()[1] * 0.01), 
                        i, 
                        f"${row['average']:,.0f}", 
                        va='center')
            
            ax.set_xlabel('Average Deal Value ($)')
            ax.set_title('Average Deal Value by Booking Type')
            st.pyplot(fig)
            
            # Show top value booking type
            if len(category_values) > 1:
                top_type = category_values.iloc[0]
                bottom_type = category_values.iloc[-1]
                st.info(f"💡 **{top_type['booking_type']}** bookings have an average value of **${top_type['average']:,.0f}**, which is {(top_type['average']/bottom_type['average'] - 1):.1%} higher than **{bottom_type['booking_type']}** bookings.")
        else:
            st.info("Insufficient deal value or booking type data")
    
    with value_col2:
        # 2. Total Revenue Over Time
        st.write("#### Total Revenue by Month")
        
        revenue_data = analyze_revenue_over_time(df, 'event_date', 'M')
        
        if revenue_data is not None and not revenue_data.empty:
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(revenue_data['event_date_period'], 
                           revenue_data['total_revenue'], 
                           color='#27ae60')
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
            
            ax.set_ylabel('Total Revenue ($)')
            ax.set_title('Monthly Revenue')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Calculate total revenue and averages
            total_revenue = revenue_data['total_revenue'].sum()
            avg_monthly = revenue_data['total_revenue'].mean()
            st.metric("Total Revenue", f"${total_revenue:,.0f}", 
                     delta=f"${avg_monthly:,.0f} avg/month")
            
            # Show revenue trend
            if len(revenue_data) > 1:
                last_period = revenue_data.iloc[-1]
                prev_period = revenue_data.iloc[-2]
                pct_change = (last_period['total_revenue'] / prev_period['total_revenue']) - 1
                if abs(pct_change) > 0.05:  # Only show if change is significant
                    direction = "up" if pct_change > 0 else "down"
                    st.info(f"💡 Revenue is trending **{direction}** with a {abs(pct_change):.1%} change in the most recent period.")
        else:
            st.info("Insufficient time-based revenue data")
    
    # 3. Price Per Guest Analysis
    st.write("#### Price Per Guest Analysis")
    
    ppg_analysis = analyze_price_per_guest(df)
    
    if ppg_analysis is not None:
        ppg_col1, ppg_col2 = st.columns(2)
        
        with ppg_col1:
            # Overall stats
            if 'overall' in ppg_analysis:
                overall = ppg_analysis['overall']
                
                st.metric("Average Price Per Guest", 
                          f"${overall.loc['mean', 'price_per_guest']:.2f}", 
                          delta=f"±${overall.loc['std', 'price_per_guest']:.2f}")
        
        with ppg_col2:
            # By category
            if 'by_category' in ppg_analysis:
                category_ppg = ppg_analysis['by_category']
                
                top_category = category_ppg.iloc[0]
                if len(category_ppg) > 1:
                    st.metric(f"Highest: {top_category['booking_type']}", 
                              f"${top_category['avg_price_per_guest']:.2f}")
                    
        # Time trend
        if 'by_time' in ppg_analysis:
            time_ppg = ppg_analysis['by_time']
            
            if not time_ppg.empty:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(time_ppg['month'], time_ppg['avg_price_per_guest'], 
                        marker='o', linewidth=2, color='#8e44ad')
                
                # Format y-axis as currency
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.2f}'))
                
                ax.set_ylabel('Average Price Per Guest ($)')
                ax.set_title('Price Per Guest Trend')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.info("Insufficient price per guest data")