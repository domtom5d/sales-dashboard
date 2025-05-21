"""
improved_visualizations.py - Enhanced visualization components with better error handling and diversity checks

This module provides improved visualizations that handle data quality issues gracefully,
including empty categories, low diversity, and proper error messages.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

def plot_conversion_by_booking_type(df):
    """
    Plot conversion rates by booking type with comprehensive error handling and data cleaning
    
    Args:
        df (DataFrame): Processed dataframe with outcome data
    """
    st.subheader("Conversion by Booking Type")

    # Step 1: Check for booking_type or event_type columns
    booking_col = None
    if 'booking_type' in df.columns:
        booking_col = 'booking_type'
    elif 'event_type' in df.columns:
        booking_col = 'event_type'
    elif 'clean_booking_type' in df.columns:
        booking_col = 'clean_booking_type'
    
    if booking_col is None:
        st.warning("No booking type or event type column found in the data.")
        return

    # Create a clean copy of the DataFrame
    clean_df = df.copy()
    
    # Ensure we only use rows with valid outcome
    clean_df = clean_df.dropna(subset=['outcome'])
    
    # Fill missing booking types with "Unknown"
    clean_df[booking_col] = clean_df[booking_col].fillna("Unknown")
    clean_df[booking_col] = clean_df[booking_col].replace("", "Unknown")
    
    # Check if we have enough variation to display
    unique_booking_types = clean_df[booking_col].nunique()
    if unique_booking_types <= 1:
        st.info(f"Not enough {booking_col} variation to display chart.")
        return

    # Group low-frequency booking types into "Other"
    type_counts = clean_df[booking_col].value_counts()
    min_count_for_own_category = 5  # Increased threshold for cleaner chart
    low_freq_types = type_counts[type_counts < min_count_for_own_category].index.tolist()
    
    # Apply grouping to create cleaner visualization
    if low_freq_types:
        clean_df[booking_col] = clean_df[booking_col].apply(
            lambda x: "Other Types" if x in low_freq_types else x
        )

    # Compute conversion rates by booking type
    try:
        # Check if we have boolean outcome or string outcome
        is_bool_outcome = clean_df['outcome'].dtype == bool
        
        if is_bool_outcome:
            # Process with boolean outcome (True/False)
            summary = (
                clean_df.groupby(booking_col)['outcome']
                .agg(['sum', 'count'])
                .reset_index()
            )
            summary.columns = [booking_col, 'Won', 'Total']
            summary['Lost'] = summary['Total'] - summary['Won']
        else:
            # Process with string outcome (Won/Lost)
            # Handle case-insensitive by converting to lowercase first
            clean_df['outcome_str'] = clean_df['outcome'].astype(str).str.lower()
            
            # Count by outcome value
            summary = clean_df.groupby(booking_col)['outcome_str'].value_counts().unstack(fill_value=0)
            
            # Reset index to make it a regular DataFrame
            summary = summary.reset_index()
            
            # Ensure we have 'won' and 'lost' columns
            for outcome_value in ['won', 'lost', 'true', 'false', '1', '0']:
                if outcome_value in summary.columns:
                    if outcome_value in ['won', 'true', '1']:
                        summary['Won'] = summary[outcome_value]
                    elif outcome_value in ['lost', 'false', '0']:
                        summary['Lost'] = summary[outcome_value]
            
            # If 'Won' or 'Lost' columns still don't exist, create them
            if 'Won' not in summary.columns:
                summary['Won'] = 0
            if 'Lost' not in summary.columns:
                summary['Lost'] = 0
            
            # Calculate total
            summary['Total'] = summary['Won'] + summary['Lost']
        
        # Calculate conversion rate
        summary['Conversion Rate'] = summary['Won'] / summary['Total']
        
        # Filter booking types with too few leads
        min_count_for_display = 5  # Minimum leads per category to display
        summary = summary[summary['Total'] >= min_count_for_display]
        
        # Check if we have enough data after filtering
        if summary.empty or summary['Conversion Rate'].nunique() <= 1:
            st.info(f"Not enough data per {booking_col} to display meaningful comparisons.")
            return

        # Create Altair chart
        chart_data = pd.DataFrame({
            'Booking Type': summary[booking_col],
            'Conversion Rate': summary['Conversion Rate'],
            'Total Leads': summary['Total'],
            'Won Deals': summary['Won'],
            'Lost Deals': summary['Lost']
        })
        
        # Sort by conversion rate for better visualization
        chart_data = chart_data.sort_values('Conversion Rate', ascending=False)
        
        # Create color scale based on conversion rate
        color_scale = alt.Scale(
            domain=[0, chart_data['Conversion Rate'].max()],
            range=['#a1c9f4', '#ff9f9b']  # Light blue to light red
        )
        
        # Create the chart with enhanced aesthetics
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Booking Type:N', title=f'{booking_col.replace("_", " ").title()}', sort='-y'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            color=alt.Color('Conversion Rate:Q', scale=color_scale),
            tooltip=[
                'Booking Type',
                alt.Tooltip('Conversion Rate:Q', format='.1%', title='Conversion Rate'),
                alt.Tooltip('Total Leads:Q', title='Total Leads'),
                alt.Tooltip('Won Deals:Q', title='Won Deals'),
                alt.Tooltip('Lost Deals:Q', title='Lost Deals')
            ]
        ).properties(
            width=600,
            height=400
        )
        
        # Display the chart
        st.altair_chart(chart, use_container_width=True)

        # Show insights about best and worst performers
        if summary['Conversion Rate'].nunique() > 1:
            # Filter out "Unknown" for insights unless it's >20% of data
            insights_data = summary.copy()
            unknown_row = insights_data[insights_data[booking_col] == "Unknown"]
            if not unknown_row.empty and unknown_row.iloc[0]['Total'] < (insights_data['Total'].sum() * 0.2):
                insights_data = insights_data[insights_data[booking_col] != "Unknown"]
            
            # Also filter out "Other Types" for cleaner insights
            insights_data = insights_data[insights_data[booking_col] != "Other Types"]
            
            if not insights_data.empty:
                # Find best and worst performers
                best = insights_data.sort_values('Conversion Rate', ascending=False).iloc[0]
                worst = insights_data.sort_values('Conversion Rate', ascending=True).iloc[0]
                
                # Calculate average conversion rate
                avg_rate = summary['Won'].sum() / summary['Total'].sum()
                
                # Calculate performance relative to average
                best_multiple = best['Conversion Rate'] / avg_rate if avg_rate > 0 else 0
                worst_multiple = worst['Conversion Rate'] / avg_rate if avg_rate > 0 else 0
                
                # Display insights with better formatting
                st.markdown("### Key Insights")
                
                # Best performer insight
                st.markdown(
                    f"**Top Performer:** {best[booking_col]} converts at **{best['Conversion Rate']:.1%}** "
                    f"({best_multiple:.1f}x better than average)"
                )
                
                # Worst performer insight (only if different from best)
                if best[booking_col] != worst[booking_col]:
                    st.markdown(
                        f"**Lowest Performer:** {worst[booking_col]} converts at **{worst['Conversion Rate']:.1%}** "
                        f"({worst_multiple:.1f}x the average)"
                    )
                
                # Add recommendation if there's a significant difference
                if best_multiple > 1.5:
                    st.markdown(
                        f"**Recommendation:** Consider focusing more resources on {best[booking_col]} events "
                        f"as they convert {best_multiple:.1f}x better than average."
                    )
        
    except Exception as e:
        st.error(f"Error generating booking type chart: {str(e)}")
        import traceback
        st.text(traceback.format_exc())


def plot_conversion_by_referral_source(df):
    """
    Plot conversion rates by referral source with enhanced error handling
    
    Args:
        df (DataFrame): Processed dataframe with outcome data
    """
    st.subheader("Conversion by Referral Source")

    if 'referral_source' not in df.columns:
        st.warning("Referral source column not found.")
        return

    df = df.copy()
    df = df.dropna(subset=['outcome'])  # Ensure we only use rows with valid outcome
    df['referral_source'] = df['referral_source'].fillna("Unknown").replace("", "Unknown")
    
    if df['referral_source'].nunique() <= 1:
        st.info("Not enough referral source variation to display chart.")
        return
    
    # Group low-frequency referral sources into "Other"
    source_counts = df['referral_source'].value_counts()
    min_count_for_own_category = 5  # Higher threshold for referral sources as they tend to be more numerous
    low_freq_sources = source_counts[source_counts < min_count_for_own_category].index.tolist()
    
    if low_freq_sources:
        df['referral_source'] = df['referral_source'].apply(
            lambda x: "Other Sources" if x in low_freq_sources else x
        )

    try:
        summary = df.groupby('referral_source', observed=True)['outcome'].value_counts().unstack(fill_value=0).reset_index()

        # Handle cases where won/lost might not exist
        if 'Won' not in summary.columns:
            summary['Won'] = 0
        if 'Lost' not in summary.columns:
            summary['Lost'] = 0
            
        summary['Total'] = summary['Won'] + summary['Lost']
        summary['Conversion Rate'] = summary['Won'] / summary['Total']

        # Filter out sources with not enough data
        min_count = 3
        summary = summary[summary['Total'] >= min_count]
        
        if summary.empty or summary['Conversion Rate'].nunique() <= 1:
            st.info("Not enough data per referral source to display meaningful comparisons.")
            return

        # Sort by conversion rate for better visualization
        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('referral_source:N', title='Referral Source', sort='-y'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            tooltip=['referral_source', alt.Tooltip('Conversion Rate:Q', format='.1%'), 'Total']
        ).properties(width=600)

        st.altair_chart(chart, use_container_width=True)

        # Show top insights - exclude Unknown from highlights unless it's >20% of data
        if summary['Conversion Rate'].nunique() > 1:
            # Filter out "Unknown" for best/worst calculations unless it's >20% of data
            summary_for_insights = summary.copy()
            unknown_row = summary_for_insights[summary_for_insights['referral_source'] == "Unknown"]
            if not unknown_row.empty and unknown_row.iloc[0]['Total'] < (summary_for_insights['Total'].sum() * 0.2):
                summary_for_insights = summary_for_insights[summary_for_insights['referral_source'] != "Unknown"]
            
            if not summary_for_insights.empty:
                best = summary_for_insights.sort_values('Conversion Rate', ascending=False).iloc[0]
                worst = summary_for_insights.sort_values('Conversion Rate', ascending=True).iloc[0]
                avg_rate = summary['Won'].sum() / summary['Total'].sum()
                
                # Calculate multiples vs average
                best_multiple = best['Conversion Rate'] / avg_rate if avg_rate > 0 else 0
                
                if best['referral_source'] != "Unknown" and best['referral_source'] != "Other Sources":
                    st.markdown(
                        f"Referral source **{best['referral_source']}** converts at **{best['Conversion Rate']:.1%}**, "
                        f"**{best_multiple:.1f}x** better than the average (**{avg_rate:.1%}**)."
                    )
                
                if (best['referral_source'] != worst['referral_source'] 
                    and worst['referral_source'] != "Unknown" 
                    and worst['referral_source'] != "Other Sources"):
                    st.markdown(
                        f"Referral source **{worst['referral_source']}** has the lowest conversion at only **{worst['Conversion Rate']:.1%}**."
                    )
    except Exception as e:
        st.error(f"Error generating referral source chart: {str(e)}")


def plot_conversion_by_event_type(df):
    """
    Plot conversion rates by event type with enhanced error handling
    
    Args:
        df (DataFrame): Processed dataframe with outcome data
    """
    st.subheader("Conversion by Event Type")

    if 'event_type' not in df.columns:
        st.warning("Event type column not found.")
        return

    df = df.copy()
    df = df.dropna(subset=['outcome'])  # Ensure we only use rows with valid outcome
    df['event_type'] = df['event_type'].fillna("Unknown").replace("", "Unknown")
    
    if df['event_type'].nunique() <= 1:
        st.info("Not enough event type variation to display chart.")
        return
    
    # Group low-frequency event types into "Other"
    type_counts = df['event_type'].value_counts()
    min_count_for_own_category = 4
    low_freq_types = type_counts[type_counts < min_count_for_own_category].index.tolist()
    
    if low_freq_types:
        df['event_type'] = df['event_type'].apply(
            lambda x: "Other Event Types" if x in low_freq_types else x
        )

    try:
        summary = df.groupby('event_type', observed=True)['outcome'].value_counts().unstack(fill_value=0).reset_index()
        
        if 'Won' not in summary.columns:
            summary['Won'] = 0
        if 'Lost' not in summary.columns:
            summary['Lost'] = 0

        summary['Total'] = summary['Won'] + summary['Lost']
        summary['Conversion Rate'] = summary['Won'] / summary['Total']

        # Filter out event types with too few leads
        min_count = 3
        summary = summary[summary['Total'] >= min_count]
        
        if summary.empty or summary['Conversion Rate'].nunique() <= 1:
            st.info("Not enough data per event type to display meaningful comparisons.")
            return

        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('event_type:N', title='Event Type', sort='-y'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            tooltip=['event_type', alt.Tooltip('Conversion Rate:Q', format='.1%'), 'Total']
        ).properties(width=600)

        st.altair_chart(chart, use_container_width=True)

        # Show top insights - exclude Unknown from highlights unless it's >20% of data
        if summary['Conversion Rate'].nunique() > 1:
            # Filter out "Unknown" for best/worst calculations unless it's >20% of data
            summary_for_insights = summary.copy()
            unknown_row = summary_for_insights[summary_for_insights['event_type'] == "Unknown"]
            if not unknown_row.empty and unknown_row.iloc[0]['Total'] < (summary_for_insights['Total'].sum() * 0.2):
                summary_for_insights = summary_for_insights[summary_for_insights['event_type'] != "Unknown"]
            
            # Also filter out the "Other" category for insights
            summary_for_insights = summary_for_insights[summary_for_insights['event_type'] != "Other Event Types"]
            
            if not summary_for_insights.empty:
                best = summary_for_insights.sort_values('Conversion Rate', ascending=False).iloc[0]
                worst = summary_for_insights.sort_values('Conversion Rate', ascending=True).iloc[0]
                avg_rate = summary['Won'].sum() / summary['Total'].sum()
                
                # Calculate multiples vs average
                best_multiple = best['Conversion Rate'] / avg_rate if avg_rate > 0 else 0
                
                if best['event_type'] != "Unknown":
                    st.markdown(
                        f"Event type **{best['event_type']}** converts at **{best['Conversion Rate']:.1%}**, "
                        f"**{best_multiple:.1f}x** better than the average (**{avg_rate:.1%}**)."
                    )
                
                if best['event_type'] != worst['event_type'] and worst['event_type'] != "Unknown":
                    st.markdown(
                        f"Event type **{worst['event_type']}** has the lowest conversion at only **{worst['Conversion Rate']:.1%}**."
                    )
    except Exception as e:
        st.error(f"Error generating event type chart: {str(e)}")


def plot_conversion_by_days_until_event(df):
    """
    Plot conversion rates by days until event
    
    Args:
        df (DataFrame): Processed dataframe with outcome data
    """
    st.subheader("Conversion by Days Until Event")

    # Check for column name variations: days_until or days_until_event
    days_column = None
    if 'days_until_event' in df.columns:
        days_column = 'days_until_event'
    elif 'days_until' in df.columns:
        days_column = 'days_until'
    
    if days_column is None:
        st.warning("Column for 'days until event' not found. Please ensure your data contains 'days_until_event' or 'days_until'.")
        return

    valid_df = df.dropna(subset=[days_column, 'outcome'])
    if valid_df.empty:
        st.info("Not enough valid data to display chart.")
        return

    try:
        valid_df['days_until_bin'] = pd.cut(valid_df[days_column], 
                                          bins=[-1, 0, 7, 30, 90, 180, 365, 10000],
                                          labels=['Past/0', '0-7d', '8-30d', '31-90d', '91-180d', '181-365d', '1yr+'])

        summary = valid_df.groupby('days_until_bin', observed=True)['outcome'].value_counts().unstack(fill_value=0).reset_index()
        
        if 'Won' not in summary.columns:
            summary['Won'] = 0
        if 'Lost' not in summary.columns:
            summary['Lost'] = 0
            
        summary['Total'] = summary['Won'] + summary['Lost']
        summary['Conversion Rate'] = summary['Won'] / summary['Total']

        # Filter out bins with too few data points
        min_count = 2
        summary = summary[summary['Total'] >= min_count]
        
        if summary.empty:
            st.info("Not enough data in each time period to display days until event analysis.")
            return

        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('days_until_bin:N', title='Days Until Event'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            tooltip=['days_until_bin', alt.Tooltip('Conversion Rate:Q', format='.1%'), 'Total']
        ).properties(width=600)

        st.altair_chart(chart, use_container_width=True)
        
        if summary['Conversion Rate'].nunique() > 1:
            best_bin = summary.sort_values('Conversion Rate', ascending=False).iloc[0]
            avg_rate = summary['Won'].sum() / summary['Total'].sum()
            best_multiple = best_bin['Conversion Rate'] / avg_rate if avg_rate > 0 else 0
            
            st.markdown(
                f"Events **{best_bin['days_until_bin']}** away have the highest conversion rate at **{best_bin['Conversion Rate']:.1%}** "
                f"(**{best_multiple:.1f}x** better than average)."
            )
    except Exception as e:
        st.error(f"Error generating days until event chart: {str(e)}")


def plot_conversion_by_days_since_inquiry(df):
    """
    Plot conversion rates by days since inquiry
    
    Args:
        df (DataFrame): Processed dataframe with outcome data
    """
    st.subheader("Conversion by Days Since Inquiry")

    # Check for column name variations: days_since or days_since_inquiry
    days_column = None
    if 'days_since_inquiry' in df.columns:
        days_column = 'days_since_inquiry'
    elif 'days_since' in df.columns:
        days_column = 'days_since'
    
    if days_column is None:
        st.warning("Column for 'days since inquiry' not found. Please ensure your data contains 'days_since_inquiry' or 'days_since'.")
        return

    valid_df = df.dropna(subset=[days_column, 'outcome'])
    if valid_df.empty:
        st.info("Not enough valid data to display chart.")
        return

    try:
        valid_df['days_since_bin'] = pd.cut(valid_df[days_column], 
                                          bins=[-1, 1, 7, 30, 90, 180, 365, 10000],
                                          labels=['<1d', '1-7d', '8-30d', '31-90d', '91-180d', '181-365d', '1yr+'])

        summary = valid_df.groupby('days_since_bin', observed=True)['outcome'].value_counts().unstack(fill_value=0).reset_index()
        
        if 'Won' not in summary.columns:
            summary['Won'] = 0
        if 'Lost' not in summary.columns:
            summary['Lost'] = 0
            
        summary['Total'] = summary['Won'] + summary['Lost']
        summary['Conversion Rate'] = summary['Won'] / summary['Total']

        # Filter out bins with too few data points
        min_count = 2
        summary = summary[summary['Total'] >= min_count]
        
        if summary.empty:
            st.info("Not enough data in each time period to display days since inquiry analysis.")
            return

        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('days_since_bin:N', title='Days Since Inquiry'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            tooltip=['days_since_bin', alt.Tooltip('Conversion Rate:Q', format='.1%'), 'Total']
        ).properties(width=600)

        st.altair_chart(chart, use_container_width=True)
        
        if summary['Conversion Rate'].nunique() > 1:
            best_bin = summary.sort_values('Conversion Rate', ascending=False).iloc[0]
            avg_rate = summary['Won'].sum() / summary['Total'].sum()
            best_multiple = best_bin['Conversion Rate'] / avg_rate if avg_rate > 0 else 0
            
            st.markdown(
                f"Inquiries from **{best_bin['days_since_bin']}** ago have the highest conversion rate at **{best_bin['Conversion Rate']:.1%}** "
                f"(**{best_multiple:.1f}x** better than average)."
            )
    except Exception as e:
        st.error(f"Error generating days since inquiry chart: {str(e)}")


def plot_conversion_by_guest_count(df):
    """
    Plot conversion rates by guest count
    
    Args:
        df (DataFrame): Processed dataframe with outcome data
    """
    st.subheader("Conversion by Guest Count")

    # Check for column name variations
    guest_column = None
    if 'number_of_guests' in df.columns:
        guest_column = 'number_of_guests'
    elif 'guest_count' in df.columns:
        guest_column = 'guest_count'
    
    if guest_column is None:
        st.warning("Column for guest count not found. Please ensure your data contains 'number_of_guests' or 'guest_count'.")
        return

    valid_df = df.dropna(subset=[guest_column, 'outcome'])
    if valid_df.empty:
        st.info("Not enough valid data to display chart.")
        return

    try:
        valid_df['guest_bin'] = pd.cut(valid_df[guest_column], 
                                     bins=[0, 50, 100, 200, 500, 1000, 99999],
                                     labels=['0-50', '51-100', '101-200', '201-500', '501-1000', '1000+'])

        summary = valid_df.groupby('guest_bin', observed=True)['outcome'].value_counts().unstack(fill_value=0).reset_index()
        
        if 'Won' not in summary.columns:
            summary['Won'] = 0
        if 'Lost' not in summary.columns:
            summary['Lost'] = 0
            
        summary['Total'] = summary['Won'] + summary['Lost']
        summary['Conversion Rate'] = summary['Won'] / summary['Total']

        # Filter bins with too few data points
        min_count = 2
        summary = summary[summary['Total'] >= min_count]
        
        if summary.empty:
            st.info("Not enough data in each guest count range to display analysis.")
            return

        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('guest_bin:N', title='Guest Count'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            tooltip=['guest_bin', alt.Tooltip('Conversion Rate:Q', format='.1%'), 'Total']
        ).properties(width=600)

        st.altair_chart(chart, use_container_width=True)
        
        if summary['Conversion Rate'].nunique() > 1:
            best_bin = summary.sort_values('Conversion Rate', ascending=False).iloc[0]
            avg_rate = summary['Won'].sum() / summary['Total'].sum()
            best_multiple = best_bin['Conversion Rate'] / avg_rate if avg_rate > 0 else 0
            
            st.markdown(
                f"Events with **{best_bin['guest_bin']}** guests have the highest conversion rate at **{best_bin['Conversion Rate']:.1%}** "
                f"(**{best_multiple:.1f}x** better than average)."
            )
    except Exception as e:
        st.error(f"Error generating guest count chart: {str(e)}")


def plot_conversion_by_bartenders(df):
    """
    Plot conversion rates by bartenders needed
    
    Args:
        df (DataFrame): Processed dataframe with outcome data
    """
    st.subheader("Conversion by Bartenders Needed")

    # Check for column name variations
    bartender_column = None
    if 'bartenders_needed' in df.columns:
        bartender_column = 'bartenders_needed'
    elif 'bartenders' in df.columns:
        bartender_column = 'bartenders'
    
    if bartender_column is None:
        st.warning("Column for bartenders not found. Please ensure your data contains 'bartenders_needed' or 'bartenders'.")
        return

    valid_df = df.dropna(subset=[bartender_column, 'outcome'])
    if valid_df.empty:
        st.info("Not enough valid data to display chart.")
        return

    try:
        valid_df['bartender_bin'] = pd.cut(valid_df[bartender_column],
                                        bins=[-1, 0, 1, 2, 3, 5, 999],
                                        labels=['0', '1', '2', '3', '4-5', '6+'])

        summary = valid_df.groupby('bartender_bin', observed=True)['outcome'].value_counts().unstack(fill_value=0).reset_index()
        
        if 'Won' not in summary.columns:
            summary['Won'] = 0
        if 'Lost' not in summary.columns:
            summary['Lost'] = 0
            
        summary['Total'] = summary['Won'] + summary['Lost']
        summary['Conversion Rate'] = summary['Won'] / summary['Total']

        # Filter bins with too few data points
        min_count = 2
        summary = summary[summary['Total'] >= min_count]
        
        if summary.empty:
            st.info("Not enough data in each bartender count range to display analysis.")
            return

        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('bartender_bin:N', title='Bartenders Needed'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            tooltip=['bartender_bin', alt.Tooltip('Conversion Rate:Q', format='.1%'), 'Total']
        ).properties(width=600)

        st.altair_chart(chart, use_container_width=True)
        
        if summary['Conversion Rate'].nunique() > 1:
            best_bin = summary.sort_values('Conversion Rate', ascending=False).iloc[0]
            avg_rate = summary['Won'].sum() / summary['Total'].sum()
            best_multiple = best_bin['Conversion Rate'] / avg_rate if avg_rate > 0 else 0
            
            st.markdown(
                f"Events requiring **{best_bin['bartender_bin']}** bartenders have the highest conversion rate at **{best_bin['Conversion Rate']:.1%}** "
                f"(**{best_multiple:.1f}x** better than average)."
            )
    except Exception as e:
        st.error(f"Error generating bartenders chart: {str(e)}")


def plot_conversion_by_state(df):
    """
    Plot conversion rates by state
    
    Args:
        df (DataFrame): Processed dataframe with outcome data
    """
    st.subheader("Conversion by State")

    if 'state' not in df.columns:
        st.warning("State column not found.")
        return

    df = df.copy()
    df['state'] = df['state'].fillna("Unknown").replace("", "Unknown")
    if df['state'].nunique() <= 1:
        st.info("Not enough state variation to display chart.")
        return

    try:
        summary = df.groupby('state')['outcome'].value_counts().unstack(fill_value=0).reset_index()
        
        if 'Won' not in summary.columns:
            summary['Won'] = 0
        if 'Lost' not in summary.columns:
            summary['Lost'] = 0
            
        summary['Total'] = summary['Won'] + summary['Lost']
        summary['Conversion Rate'] = summary['Won'] / summary['Total']

        # Filter states with too few data points
        min_count = 2
        summary = summary[summary['Total'] >= min_count]
        
        if summary.empty:
            st.info("Not enough data per state to display analysis.")
            return

        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('state:N', sort='-y'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            tooltip=['state', alt.Tooltip('Conversion Rate:Q', format='.1%'), 'Total']
        ).properties(width=600)

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating state chart: {str(e)}")


def plot_conversion_by_city(df):
    """
    Plot conversion rates by city
    
    Args:
        df (DataFrame): Processed dataframe with outcome data
    """
    st.subheader("Conversion by City")

    if 'city' not in df.columns:
        st.warning("City column not found.")
        return

    df = df.copy()
    df['city'] = df['city'].fillna("Unknown").replace("", "Unknown")
    
    # Get top cities to avoid overcrowding the chart
    top_cities = df['city'].value_counts().nlargest(15).index.tolist()
    city_df = df[df['city'].isin(top_cities)]
    
    if city_df.empty or city_df['city'].nunique() <= 1:
        st.info("Not enough city variation to display chart.")
        return

    try:
        summary = city_df.groupby('city')['outcome'].value_counts().unstack(fill_value=0).reset_index()
        
        if 'Won' not in summary.columns:
            summary['Won'] = 0
        if 'Lost' not in summary.columns:
            summary['Lost'] = 0
            
        summary['Total'] = summary['Won'] + summary['Lost']
        summary['Conversion Rate'] = summary['Won'] / summary['Total']

        # Filter cities with too few data points
        min_count = 2
        summary = summary[summary['Total'] >= min_count]
        
        if summary.empty:
            st.info("Not enough data per city to display analysis.")
            return

        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('city:N', sort='-y'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            tooltip=['city', alt.Tooltip('Conversion Rate:Q', format='.1%'), 'Total']
        ).properties(width=600)

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating city chart: {str(e)}")


def plot_deal_value_analysis(df):
    """
    Plot deal value analysis with proper handling of missing data
    
    Args:
        df (DataFrame): Processed dataframe with outcome and deal value data
    """
    st.subheader("Deal Value Analysis")

    # Check for required columns
    if 'actual_deal_value' not in df.columns:
        st.warning("Deal value data not available")
        return
    
    # Only use valid positive deal values
    df = df.copy()
    valid_deals = df[df['actual_deal_value'].notna() & (df['actual_deal_value'] > 0)]
    
    if len(valid_deals) < 5:
        st.info(f"Not enough valid deal value data (found {len(valid_deals)} deals with values)")
        return

    try:
        # Create reasonable bins based on distribution
        min_val = valid_deals['actual_deal_value'].min()
        max_val = valid_deals['actual_deal_value'].max()
        
        # Create bins with reasonable ranges
        bins = [0, 1000, 2500, 5000, 10000, 25000, float('inf')]
        bin_labels = ['$0-$1k', '$1k-$2.5k', '$2.5k-$5k', '$5k-$10k', '$10k-$25k', '$25k+']
        
        valid_deals['deal_value_bin'] = pd.cut(
            valid_deals['actual_deal_value'],
            bins=bins,
            labels=bin_labels,
            include_lowest=True
        )
        
        # Compute conversion rate by deal value bin
        summary = valid_deals.groupby('deal_value_bin')['outcome'].value_counts().unstack(fill_value=0).reset_index()
        
        if 'Won' not in summary.columns:
            summary['Won'] = 0
        if 'Lost' not in summary.columns:
            summary['Lost'] = 0
            
        summary['Total'] = summary['Won'] + summary['Lost']
        summary['Conversion Rate'] = summary['Won'] / summary['Total']
        
        # Filter out bins with too few deals
        min_count = 2
        summary = summary[summary['Total'] >= min_count]
        
        if summary.empty:
            st.info("Not enough data in each price range to display deal value analysis.")
            return

        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('deal_value_bin:N', title='Deal Value Range'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            tooltip=['deal_value_bin', alt.Tooltip('Conversion Rate:Q', format='.1%'), 'Total']
        ).properties(width=600)

        st.altair_chart(chart, use_container_width=True)
        
        if summary['Conversion Rate'].nunique() > 1:
            best_bin = summary.sort_values('Conversion Rate', ascending=False).iloc[0]
            worst_bin = summary.sort_values('Conversion Rate', ascending=True).iloc[0]
            st.markdown(
                f"Price range **{best_bin['deal_value_bin']}** has the highest conversion rate at **{best_bin['Conversion Rate']:.1%}**, " 
                f"while **{worst_bin['deal_value_bin']}** has the lowest at **{worst_bin['Conversion Rate']:.1%}**."
            )
        
    except Exception as e:
        st.error(f"Error generating deal value analysis: {str(e)}")


def plot_data_completeness(df, chart_dependencies=None):
    """
    Plot data completeness with better explanation and styling
    
    Args:
        df (DataFrame): Processed dataframe
        chart_dependencies (dict, optional): Dictionary mapping column names to affected charts
    """
    st.subheader("Data Completeness Analysis")
    
    if chart_dependencies is None:
        chart_dependencies = {
            'booking_type': ['Booking Type Conversion'],
            'referral_source': ['Referral Source Conversion'],
            'event_type': ['Event Type Analysis'],
            'number_of_guests': ['Guest Count Analysis'],
            'actual_deal_value': ['Deal Value Analysis'],
            'inquiry_date': ['Timing Analysis'],
            'bartenders_needed': ['Staff Ratio Analysis'],
            'event_date': ['Event Month Analysis'],
            'street': ['Geographic Analysis']
        }
    
    # Calculate completeness for each column
    completeness = {}
    for col in chart_dependencies.keys():
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # For numeric columns, check for NaN, infinity, and valid values
                valid_values = df[col].notna() & np.isfinite(df[col])
                non_null = valid_values.sum()
            else:
                # For non-numeric columns, check for NaN and empty strings
                non_null = (~df[col].isna() & (df[col] != "")).sum()
            completeness[col] = (non_null / len(df)) * 100  # Convert to percentage
        else:
            completeness[col] = 0
    
    # Create a completeness dataframe
    completeness_df = pd.DataFrame({
        'Column': list(completeness.keys()),
        'Completeness': list(completeness.values())
    })
    
    # Add chart dependencies to the dataframe
    completeness_df['Affects Charts'] = completeness_df['Column'].map(
        lambda col: ', '.join(chart_dependencies.get(col, []))
    )
    
    # Add status column
    completeness_df['Status'] = 'Good'
    completeness_df.loc[completeness_df['Completeness'] < 75, 'Status'] = 'Fair'
    completeness_df.loc[completeness_df['Completeness'] < 50, 'Status'] = 'Poor'
    completeness_df.loc[completeness_df['Completeness'] < 25, 'Status'] = 'Critical'
    
    # Sort by completeness
    completeness_df = completeness_df.sort_values('Completeness', ascending=False)
    
    # Add explanation section
    st.subheader("Understanding Chart Display Issues")
    
    # Check for data category issues
    critical_columns = completeness_df[completeness_df['Status'] == 'Critical']['Column'].tolist()
    poor_columns = completeness_df[completeness_df['Status'] == 'Poor']['Column'].tolist()
    
    # Explain what's wrong with any charts that might not be showing properly
    with st.expander("Why are some charts not displaying correctly?", expanded=True):
        st.markdown("""
        ### Common Dashboard Issues
        
        Charts in this dashboard may not display correctly for several reasons:
        
        1. **Missing Data**: If there are too many blank values in critical fields.
        2. **Low Category Diversity**: When all records show the same category value (like "Uncategorized").
        3. **Insufficient Data Volume**: Some visualizations require a minimum number of records per category.
        """)
        
        # Add specific information about the current dataset
        if critical_columns:
            st.markdown("### Critical Data Issues Found")
            st.markdown("The following fields have **severe data issues** (under 25% completeness):")
            for col in critical_columns:
                affected_charts = completeness_df.loc[completeness_df['Column'] == col, 'Affects Charts'].iloc[0]
                st.markdown(f"- **{col}**: {affected_charts}")
        
        if poor_columns:
            st.markdown("### Poor Data Quality Found")
            st.markdown("The following fields have **poor data quality** (under 50% completeness):")
            for col in poor_columns:
                affected_charts = completeness_df.loc[completeness_df['Column'] == col, 'Affects Charts'].iloc[0]
                st.markdown(f"- **{col}**: {affected_charts}")
        
        # Check for category diversity issues by examining key category columns
        category_cols = ['booking_type', 'event_type', 'referral_source']
        diversity_issues = []
        
        for col in category_cols:
            if col in df.columns:
                # Count truly unique values (ignoring null/empty)
                valid_values = df[col].dropna().replace('', np.nan).dropna()
                unique_count = valid_values.nunique()
                
                if unique_count <= 1 and len(valid_values) > 0:
                    diversity_issues.append((col, valid_values.iloc[0] if len(valid_values) > 0 else "Unknown"))
        
        if diversity_issues:
            st.markdown("### Category Diversity Issues")
            st.markdown("The following fields lack diversity (all records show same value):")
            for col, value in diversity_issues:
                st.markdown(f"- **{col}**: All records show '{value}'")
            
            st.info("TIP: Try adjusting your data preprocessing to ensure more diverse category values.")
    
    # Create completeness chart with Altair for better interactivity
    chart_data = completeness_df.copy()
    
    # Create a color scale
    color_scale = alt.Scale(
        domain=['Critical', 'Poor', 'Fair', 'Good'],
        range=['#F44336', '#FF9800', '#FFC107', '#4CAF50']
    )
    
    chart = alt.Chart(chart_data).mark_bar().encode(
        y=alt.Y('Column:N', title='Data Field', sort='-x'),
        x=alt.X('Completeness:Q', title='Completeness (%)', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('Status:N', scale=color_scale, legend=alt.Legend(title="Status")),
        tooltip=['Column', 'Completeness', 'Status', 'Affects Charts']
    ).properties(
        width=700,
        height=400,
        title='Data Completeness by Field'
    )
    
    # Add text labels
    text = chart.mark_text(
        align='left',
        baseline='middle',
        dx=3,
        color='black'
    ).encode(
        text=alt.Text('Completeness:Q', format='.1f')
    )
    
    # Combine chart and text
    final_chart = (chart + text)
    
    st.altair_chart(final_chart, use_container_width=True)
    
    # Return completeness data for potential use elsewhere
    return completeness_df