"""
advanced_analytics.py - Additional analytics features for the dashboard

This module contains functions for more advanced analytics such as:
- Referral source analysis
- Marketing source analysis
- Booking type performance
- Price per guest analysis
- Event month seasonality
- Day of week patterns
- Staff-to-guest ratio analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_referral_sources(df, min_count=10):
    """
    Analyze conversion rates by referral source
    
    Args:
        df (DataFrame): Processed dataframe with outcome
        min_count (int): Minimum number of leads for a referral source to be included
    
    Returns:
        DataFrame: Summary of referral sources with conversion rates
    """
    if 'referral_source' not in df.columns:
        return None
    
    ref_summary = (
        df.groupby('referral_source')['outcome']
          .agg(Total='size', Won='sum')
          .assign(Conversion=lambda x: x['Won']/x['Total'])
          .sort_values('Conversion', ascending=False)
          .reset_index()
    )
    
    # Filter to sources with at least min_count leads
    ref_summary = ref_summary[ref_summary['Total'] >= min_count]
    
    # Add percentage formatting for display
    ref_summary['Conversion %'] = (ref_summary['Conversion']*100).round(1).astype(str) + '%'
    
    return ref_summary


def analyze_marketing_sources(df, min_count=10):
    """
    Analyze conversion rates by marketing source
    
    Args:
        df (DataFrame): Processed dataframe with outcome
        min_count (int): Minimum number of leads for a marketing source to be included
    
    Returns:
        DataFrame: Summary of marketing sources with conversion rates
    """
    if 'marketing_source' not in df.columns:
        return None
    
    mkt_summary = (
        df.groupby('marketing_source')['outcome']
          .agg(Total='size', Won='sum')
          .assign(Conversion=lambda x: x['Won']/x['Total'])
          .sort_values('Conversion', ascending=False)
          .reset_index()
    )
    
    # Filter to sources with at least min_count leads
    mkt_summary = mkt_summary[mkt_summary['Total'] >= min_count]
    
    # Add percentage formatting for display
    mkt_summary['Conversion %'] = (mkt_summary['Conversion']*100).round(1).astype(str) + '%'
    
    return mkt_summary


def analyze_booking_types(df, min_count=10):
    """
    Analyze conversion rates by booking type with name cleanup
    
    Args:
        df (DataFrame): Processed dataframe with outcome
        min_count (int): Minimum number of leads for a booking type to be included
    
    Returns:
        DataFrame: Summary of booking types with conversion rates
    """
    if 'booking_type' not in df.columns:
        return None
    
    # Create a copy with cleaned booking types
    booking = df.copy()
    booking['booking_type_clean'] = (
        booking['booking_type']
          .fillna('Unknown')
          .astype(str)
          .str.lower()
          .str.replace(r'\d{4}$','', regex=True)
          .str.title()
    )
    
    # Consolidate low-volume booking types
    counts = booking['booking_type_clean'].value_counts()
    low_vol = counts[counts < min_count].index
    booking['booking_type_clean'] = booking['booking_type_clean'].replace(low_vol, 'Other')
    
    # Create summary
    bt_summary = (
        booking.groupby('booking_type_clean')['outcome']
               .agg(Total='size', Won='sum')
               .assign(Conversion=lambda x: x['Won']/x['Total'])
               .sort_values('Conversion', ascending=False)
               .reset_index()
    )
    
    # Add percentage formatting for display
    bt_summary['Conversion %'] = (bt_summary['Conversion']*100).round(1).astype(str) + '%'
    
    return bt_summary


def analyze_price_per_guest(df):
    """
    Analyze conversion rates by price per guest buckets
    
    Args:
        df (DataFrame): Processed dataframe with outcome and price info
    
    Returns:
        DataFrame: Summary of price per guest buckets with conversion rates
    """
    # Skip if we don't have the necessary columns
    if 'actual_deal_value' not in df.columns or 'number_of_guests' not in df.columns:
        return None
    
    # Create a copy with price per guest calculation
    ppg_df = df.copy()
    
    # Clean and convert data
    ppg_df['actual_deal_value'] = pd.to_numeric(ppg_df['actual_deal_value'], errors='coerce')
    ppg_df['number_of_guests'] = pd.to_numeric(ppg_df['number_of_guests'], errors='coerce')
    
    # Drop rows with missing or invalid values
    ppg_df = ppg_df.dropna(subset=['actual_deal_value', 'number_of_guests'])
    ppg_df = ppg_df[ppg_df['number_of_guests'] > 0]  # Avoid division by zero
    
    # Calculate price per guest
    ppg_df['price_per_guest'] = ppg_df['actual_deal_value'] / ppg_df['number_of_guests']
    
    # If we don't have enough data after cleaning
    if len(ppg_df) < 5:
        return None
    
    # Define buckets
    ppg_bins = [0, 50, 100, 200, np.inf]
    ppg_labels = ['<$50', '$50-99', '$100-199', '$200+']
    ppg_df['ppg_bin'] = pd.cut(ppg_df['price_per_guest'], bins=ppg_bins, labels=ppg_labels)
    
    # Create summary
    ppg_summary = (
        ppg_df.groupby('ppg_bin')['outcome']
          .agg(Total='size', Won='sum')
          .assign(Conversion=lambda x: x['Won']/x['Total'])
          .reset_index()
    )
    
    # Add percentage formatting for display
    ppg_summary['Conversion %'] = (ppg_summary['Conversion']*100).round(1).astype(str) + '%'
    
    return ppg_summary


def analyze_days_since_inquiry(df):
    """
    Analyze conversion rates by days since inquiry buckets
    
    Args:
        df (DataFrame): Processed dataframe with outcome and days_since_inquiry
    
    Returns:
        DataFrame: Summary of days since inquiry buckets with conversion rates
    """
    if 'days_since_inquiry' not in df.columns:
        return None
    
    # Create a copy with binned days since inquiry
    dsi_df = df.copy()
    
    # Drop nulls and ensure numeric
    dsi_df = dsi_df.dropna(subset=['days_since_inquiry'])
    dsi_df['days_since_inquiry'] = pd.to_numeric(dsi_df['days_since_inquiry'], errors='coerce')
    dsi_df = dsi_df.dropna(subset=['days_since_inquiry'])
    
    # If we don't have enough data
    if len(dsi_df) < 5:
        return None
    
    # Define buckets
    dsi_bins = [0, 1, 3, 7, 30, np.inf]
    dsi_labels = ['Same day', '1-3 days', '4-7 days', '8-30 days', '30+ days']
    
    # Create binned column
    dsi_df['dsi_bin'] = pd.cut(dsi_df['days_since_inquiry'], bins=dsi_bins, labels=dsi_labels)
    
    # Create summary
    dsi_summary = (
        dsi_df.groupby('dsi_bin')['outcome']
          .agg(Total='size', Won='sum')
          .assign(Conversion=lambda x: x['Won']/x['Total'])
          .reset_index()
    )
    
    # Add percentage formatting for display
    dsi_summary['Conversion %'] = (dsi_summary['Conversion']*100).round(1).astype(str) + '%'
    
    return dsi_summary


def analyze_event_month(df):
    """
    Analyze conversion rates by event month
    
    Args:
        df (DataFrame): Processed dataframe with outcome and event_date
    
    Returns:
        DataFrame: Summary of event months with conversion rates
    """
    if 'event_date' not in df.columns:
        return None
    
    # Create a copy with event month extracted
    month_df = df.copy()
    month_df['event_date'] = pd.to_datetime(month_df['event_date'], errors='coerce')
    month_df = month_df.dropna(subset=['event_date'])
    
    # If we don't have enough data after cleaning
    if len(month_df) < 5:
        return None
    
    # Extract month name
    month_df['event_month'] = month_df['event_date'].dt.month_name()
    
    # Create summary
    month_summary = (
        month_df.groupby('event_month')['outcome']
          .agg(Total='size', Won='sum')
          .assign(Conversion=lambda x: x['Won']/x['Total'])
          .reset_index()
    )
    
    # Add percentage formatting for display
    month_summary['Conversion %'] = (month_summary['Conversion']*100).round(1).astype(str) + '%'
    
    # Reorder months chronologically
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    month_summary['month_order'] = month_summary['event_month'].map(
        {month: i for i, month in enumerate(month_order)}
    )
    month_summary = month_summary.sort_values('month_order').drop(columns=['month_order'])
    
    return month_summary


def analyze_inquiry_weekday(df):
    """
    Analyze conversion rates by inquiry weekday
    
    Args:
        df (DataFrame): Processed dataframe with outcome and inquiry_date
    
    Returns:
        DataFrame: Summary of inquiry weekdays with conversion rates
    """
    if 'inquiry_date' not in df.columns:
        return None
    
    # Create a copy with inquiry weekday extracted
    wkday_df = df.copy()
    wkday_df['inquiry_date'] = pd.to_datetime(wkday_df['inquiry_date'], errors='coerce')
    wkday_df = wkday_df.dropna(subset=['inquiry_date'])
    
    # If we don't have enough data after cleaning
    if len(wkday_df) < 5:
        return None
    
    # Extract weekday name
    wkday_df['inquiry_weekday'] = wkday_df['inquiry_date'].dt.day_name()
    
    # Create summary
    wkday_summary = (
        wkday_df.groupby('inquiry_weekday')['outcome']
          .agg(Total='size', Won='sum')
          .assign(Conversion=lambda x: x['Won']/x['Total'])
          .reset_index()
    )
    
    # Add percentage formatting for display
    wkday_summary['Conversion %'] = (wkday_summary['Conversion']*100).round(1).astype(str) + '%'
    
    # Reorder weekdays chronologically
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    wkday_summary['weekday_order'] = wkday_summary['inquiry_weekday'].map(
        {day: i for i, day in enumerate(weekday_order)}
    )
    wkday_summary = wkday_summary.sort_values('weekday_order').drop(columns=['weekday_order'])
    
    return wkday_summary


def analyze_staff_ratio(df):
    """
    Analyze conversion rates by staff-to-guest ratio
    
    Args:
        df (DataFrame): Processed dataframe with outcome, bartenders_needed, and number_of_guests
    
    Returns:
        DataFrame: Summary of staff ratio buckets with conversion rates
    """
    if 'bartenders_needed' not in df.columns or 'number_of_guests' not in df.columns:
        return None
    
    # Create a copy with staff ratio calculated
    staff_df = df.copy()
    
    # Clean and convert data
    staff_df['number_of_guests'] = pd.to_numeric(staff_df['number_of_guests'], errors='coerce')
    staff_df['bartenders_needed'] = pd.to_numeric(staff_df['bartenders_needed'], errors='coerce')
    
    # Drop rows with missing or invalid values
    staff_df = staff_df.dropna(subset=['bartenders_needed', 'number_of_guests'])
    staff_df = staff_df[staff_df['number_of_guests'] > 0]  # Avoid division by zero
    
    # Calculate staff ratio
    staff_df['staff_ratio'] = staff_df['bartenders_needed'] / staff_df['number_of_guests']
    
    # If we don't have enough data after cleaning
    if len(staff_df) < 5:
        return None
    
    # Define buckets
    bins = [0, 0.01, 0.02, 0.05, np.inf]
    labels = ['<1%', '1-2%', '2-5%', '5%+']
    staff_df['staff_ratio_bin'] = pd.cut(staff_df['staff_ratio'], bins=bins, labels=labels)
    
    # Create summary
    sr_summary = (
        staff_df.groupby('staff_ratio_bin')['outcome']
          .agg(Total='size', Won='sum')
          .assign(Conversion=lambda x: x['Won']/x['Total'])
          .reset_index()
    )
    
    # Add percentage formatting for display
    sr_summary['Conversion %'] = (sr_summary['Conversion']*100).round(1).astype(str) + '%'
    
    return sr_summary


def plot_conversion_by_category(df, category_col, title, ax=None, sort_by='conversion', top_n=None, custom_order=None):
    """
    Plot conversion rates by a category column
    
    Args:
        df (DataFrame): Processed dataframe with outcome
        category_col (str): Column to group by
        title (str): Title for the plot
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        sort_by (str): Column to sort by ('conversion', 'volume', 'natural', or 'custom')
        top_n (int, optional): Limit to top N categories
        custom_order (list, optional): Custom ordering of categories (used when sort_by='custom')
    
    Returns:
        matplotlib.axes.Axes: The axes with the plot
    """
    if category_col not in df.columns:
        # Create a blank axis with a message if we don't have the data
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No data for {category_col}", 
                horizontalalignment='center', fontsize=14)
        ax.axis('off')
        return ax
    
    # Create summary
    summary = (
        df.groupby(category_col)['outcome']
          .agg(volume='size', won='sum')
          .assign(conversion=lambda x: x['won']/x['volume'])
          .reset_index()
    )
    
    # Sort by specified column and get top N if requested
    if sort_by == 'conversion':
        summary = summary.sort_values('conversion', ascending=False)
    elif sort_by == 'volume':
        summary = summary.sort_values('volume', ascending=False)
    elif sort_by == 'custom' and custom_order is not None:
        # Create a categorical column with the custom order
        category_order = pd.Categorical(
            summary[category_col], 
            categories=custom_order, 
            ordered=True
        )
        summary[category_col] = category_order
        summary = summary.sort_values(category_col)
    # Default to volume if sort_by is not recognized
    else:
        summary = summary.sort_values('volume', ascending=False)
        
    if top_n is not None:
        summary = summary.head(top_n)
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars with special coloring for referral sources
    if category_col == 'referral_source':
        # Create a color list with top 2 sources highlighted
        colors = ['#1E88E5' if i >= 2 else '#FFC107' for i in range(len(summary))]
        bars = ax.barh(summary[category_col], summary['conversion'], color=colors)
    else:
        # Default color for other categories
        bars = ax.barh(summary[category_col], summary['conversion'], color='dodgerblue')
    
    # Add data labels - format depends on the category type
    for i, bar in enumerate(bars):
        width = bar.get_width()
        volume = summary.iloc[i]['volume']
        category = summary.iloc[i][category_col]
        
        # Special formatting for booking types - show conversion% and count more prominently
        if category_col == 'booking_type' or category_col == 'booking_type_clean':
            ax.text(0.01, bar.get_y() + bar.get_height()/2, 
                    f"{width:.1%}", va='center', color='white', fontweight='bold', fontsize=10)
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"n={volume}", va='center')
        else:
            # Default formatting for other categories
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"{width:.1%} (n={volume})", va='center')
    
    # Customize plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Conversion Rate', fontsize=12)
    ax.set_xlim(0, max(summary['conversion']) * 1.2)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add a note for referral sources to explain the highlighting
    if category_col == 'referral_source':
        # Add a note about highlighted top sources
        ax.text(0.98, 0.02, "Top 2 sources highlighted", 
                transform=ax.transAxes, fontsize=9, ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    return ax


def run_all_analytics(df):
    """
    Run all advanced analytics functions on the dataset
    
    Args:
        df (DataFrame): Processed dataframe with outcome
    
    Returns:
        dict: Dictionary containing all analysis results
    """
    results = {
        'referral_sources': analyze_referral_sources(df),
        'marketing_sources': analyze_marketing_sources(df),
        'booking_types': analyze_booking_types(df),
        'price_per_guest': analyze_price_per_guest(df),
        'days_since_inquiry': analyze_days_since_inquiry(df),
        'event_month': analyze_event_month(df),
        'inquiry_weekday': analyze_inquiry_weekday(df),
        'staff_ratio': analyze_staff_ratio(df)
    }
    
    return results