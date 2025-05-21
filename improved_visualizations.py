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
    Plot conversion rates by booking type with enhanced error handling
    
    Args:
        df (DataFrame): Processed dataframe with outcome data
    """
    st.subheader("Conversion by Booking Type")

    # Check for required columns
    if 'booking_type' not in df.columns or 'outcome' not in df.columns:
        st.warning("Missing required columns: booking_type or outcome")
        return
    
    # Sanitize missing values
    df = df.copy()
    df['booking_type'] = df['booking_type'].fillna("Unknown")
    df['booking_type'] = df['booking_type'].replace("", "Unknown")

    # Only proceed if there's more than one type
    unique_booking_types = df['booking_type'].nunique()
    if unique_booking_types <= 1:
        st.info(f"Not enough booking type variation to display chart. All records show: {df['booking_type'].iloc[0]}")
        return

    # Compute conversion rate by booking type
    try:
        summary = (
            df.groupby('booking_type')['outcome']
            .value_counts(normalize=False)
            .unstack(fill_value=0)
            .reset_index()
        )

        if True not in summary.columns:
            summary[True] = 0
        if False not in summary.columns:
            summary[False] = 0

        summary['Total'] = summary[True] + summary[False]
        summary['Conversion Rate'] = summary[True] / summary['Total']
        
        # Filter out booking types with too few leads
        min_count = 3  # Minimum leads per category
        summary = summary[summary['Total'] >= min_count]
        
        if summary.empty or summary['Conversion Rate'].nunique() <= 1:
            st.info("Not enough data per booking type to display meaningful comparisons.")
            return

        # Plot using Altair
        summary_renamed = summary.rename(columns={True: 'Won', False: 'Lost'})
        chart_data = pd.DataFrame({
            'booking_type': summary_renamed['booking_type'],
            'Conversion Rate': summary_renamed['Conversion Rate']
        })
        
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('booking_type:N', title='Booking Type', sort='-y'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            tooltip=['booking_type', alt.Tooltip('Conversion Rate:Q', format='.1%')]
        ).properties(
            width=600,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)

        # Show top insights
        best = summary.sort_values('Conversion Rate', ascending=False).iloc[0]
        worst = summary.sort_values('Conversion Rate', ascending=True).iloc[0]
        delta = best['Conversion Rate'] - worst['Conversion Rate']
        st.markdown(
            f"**{best['booking_type']}** has the highest conversion rate at **{best['Conversion Rate']:.1%}**, "
            f"compared to **{worst['booking_type']}** at **{worst['Conversion Rate']:.1%}** "
            f"(**{delta:.1%}** difference)."
        )
        
    except Exception as e:
        st.error(f"Error generating booking type chart: {str(e)}")


def plot_conversion_by_referral_source(df):
    """
    Plot conversion rates by referral source with enhanced error handling
    
    Args:
        df (DataFrame): Processed dataframe with outcome data
    """
    st.subheader("Conversion by Referral Source")

    # Check for required columns
    if 'referral_source' not in df.columns or 'outcome' not in df.columns:
        st.warning("Missing required columns: referral_source or outcome")
        return
    
    # Sanitize missing values
    df = df.copy()
    df['referral_source'] = df['referral_source'].fillna("Unknown")
    df['referral_source'] = df['referral_source'].replace("", "Unknown")

    # Only proceed if there's more than one source
    unique_referral_sources = df['referral_source'].nunique()
    if unique_referral_sources <= 1:
        st.info(f"Not enough referral source variation to display chart. All records show: {df['referral_source'].iloc[0]}")
        return

    # Compute conversion rate by referral source
    try:
        summary = (
            df.groupby('referral_source')['outcome']
            .value_counts(normalize=False)
            .unstack(fill_value=0)
            .reset_index()
        )

        if True not in summary.columns:
            summary[True] = 0
        if False not in summary.columns:
            summary[False] = 0

        summary['Total'] = summary[True] + summary[False]
        summary['Conversion Rate'] = summary[True] / summary['Total']
        
        # Filter out referral sources with too few leads
        min_count = 3  # Minimum leads per category
        summary = summary[summary['Total'] >= min_count]
        
        if summary.empty or summary['Conversion Rate'].nunique() <= 1:
            st.info("Not enough data per referral source to display meaningful comparisons.")
            return

        # Plot using Altair - sort by conversion rate
        summary_renamed = summary.rename(columns={True: 'Won', False: 'Lost'})
        chart_data = pd.DataFrame({
            'referral_source': summary_renamed['referral_source'],
            'Conversion Rate': summary_renamed['Conversion Rate']
        })
        
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('referral_source:N', title='Referral Source', sort='-y'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            tooltip=['referral_source', alt.Tooltip('Conversion Rate:Q', format='.1%')]
        ).properties(
            width=600,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)

        # Show top insights
        best = summary.sort_values('Conversion Rate', ascending=False).iloc[0]
        worst = summary.sort_values('Conversion Rate', ascending=True).iloc[0]
        delta = best['Conversion Rate'] - worst['Conversion Rate']
        st.markdown(
            f"**{best['referral_source']}** has the highest conversion rate at **{best['Conversion Rate']:.1%}**, "
            f"compared to **{worst['referral_source']}** at **{worst['Conversion Rate']:.1%}** "
            f"(**{delta:.1%}** difference)."
        )
        
    except Exception as e:
        st.error(f"Error generating referral source chart: {str(e)}")


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
        # Create deal value bins
        deals_with_value = valid_deals.copy()
        
        # Create reasonable bins based on distribution
        min_val = deals_with_value['actual_deal_value'].min()
        max_val = deals_with_value['actual_deal_value'].max()
        
        # Create 5 bins with reasonable ranges
        bin_count = 5
        bins = np.linspace(min_val, max_val, bin_count + 1)
        bin_labels = [f"${bins[i]:.0f} - ${bins[i+1]:.0f}" for i in range(bin_count)]
        
        deals_with_value['deal_value_bin'] = pd.cut(
            deals_with_value['actual_deal_value'],
            bins=bins,
            labels=bin_labels,
            include_lowest=True
        )
        
        # Compute conversion rate by deal value bin
        summary = (
            deals_with_value.groupby('deal_value_bin')['outcome']
            .value_counts(normalize=False)
            .unstack(fill_value=0)
            .reset_index()
        )
        
        if True not in summary.columns:
            summary[True] = 0
        if False not in summary.columns:
            summary[False] = 0

        summary['Total'] = summary[True] + summary[False]
        summary['Conversion Rate'] = summary[True] / summary['Total']
        
        # Filter out bins with too few deals
        min_count = 2  # Lower threshold for deal value since we might have fewer data points
        summary = summary[summary['Total'] >= min_count]
        
        if summary.empty:
            st.info("Not enough data in each price range to display deal value analysis.")
            return

        # Plot using Altair
        summary_renamed = summary.rename(columns={True: 'Won', False: 'Lost'})
        chart_data = pd.DataFrame({
            'Deal Value Range': summary_renamed['deal_value_bin'],
            'Conversion Rate': summary_renamed['Conversion Rate'],
            'Total Deals': summary_renamed['Total']
        })
        
        # Use Altair for better interactive visualizations
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Deal Value Range:N', title='Deal Value Range'),
            y=alt.Y('Conversion Rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            tooltip=['Deal Value Range', 
                     alt.Tooltip('Conversion Rate:Q', format='.1%'),
                     'Total Deals']
        ).properties(
            width=600,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)
        
        # Add insights
        if summary['Conversion Rate'].nunique() > 1:
            best_bin = summary.sort_values('Conversion Rate', ascending=False).iloc[0]
            worst_bin = summary.sort_values('Conversion Rate', ascending=True).iloc[0]
            st.markdown(
                f"Price range **{best_bin['deal_value_bin']}** has the highest conversion rate at **{best_bin['Conversion Rate']:.1%}**"
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