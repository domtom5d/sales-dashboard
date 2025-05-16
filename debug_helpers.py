"""
debug_helpers.py - Diagnostic utilities for dashboard debugging

This module provides utility functions for diagnosing and debugging
issues in the Sales Conversion Analytics Dashboard.
"""

import pandas as pd
import streamlit as st

def create_health_check_table(df):
    """
    Create a health check table to verify if required columns exist
    for each dashboard tab.
    
    Args:
        df (DataFrame): The processed dataframe
        
    Returns:
        DataFrame: Health check table
    """
    # Define required columns for each tab
    tab_requirements = {
        "Conversion Analysis": ['outcome', 'created'] if 'created' in df.columns else ['outcome'],
        "Feature Correlation": ['outcome'] + [c for c in df.select_dtypes('number').columns if c != 'outcome'][:5],
        "Lead Scoring": ['outcome'] + [c for c in df.select_dtypes('number').columns if c != 'outcome'][:5],
        "Raw Data": [] # Raw data tab has no specific requirements
    }
    
    # Check health of each tab
    health_data = []
    for tab_name, required_cols in tab_requirements.items():
        if not required_cols:  # Skip if no requirements
            health_data.append({"Tab": tab_name, "Status": "✅ OK"})
            continue
            
        missing = [c for c in required_cols if c not in df.columns]
        status = "❌ Missing: " + ", ".join(missing) if missing else "✅ OK"
        health_data.append({"Tab": tab_name, "Status": status})
    
    return pd.DataFrame(health_data)

def get_data_summary(df):
    """
    Generate a summary of the data for debugging
    
    Args:
        df (DataFrame): The processed dataframe
        
    Returns:
        str: Markdown-formatted summary
    """
    if df is None or df.empty:
        return "No data available"
    
    try:
        # Basic metrics
        total_records = len(df)
        
        # Outcome metrics if available
        if 'outcome' in df.columns:
            win_count = int(df['outcome'].sum())
            loss_count = total_records - win_count
            win_rate = win_count / total_records * 100 if total_records > 0 else 0
            outcome_section = f"""
            - Won deals: {win_count} ({win_rate:.1f}%)
            - Lost deals: {loss_count}
            """
        else:
            outcome_section = "- Outcome column not available"
            
        # Column metrics
        num_cols = len(df.select_dtypes(['int64', 'float64']).columns)
        cat_cols = len(df.select_dtypes(['object', 'category']).columns)
        date_cols = len(df.select_dtypes(['datetime64']).columns)
        
        # Missing data
        missing_pct = df.isna().mean().mean() * 100
        
        return f"""
        ### Data Summary
        
        **Basic Information:**
        - Total records: {total_records}
        {outcome_section}
        
        **Column Types:**
        - Numeric columns: {num_cols}
        - Categorical columns: {cat_cols}
        - Date columns: {date_cols}
        
        **Data Quality:**
        - Missing data: {missing_pct:.1f}% overall
        """
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def safe_display_df_preview(df, num_rows=5):
    """
    Safely display a preview of a dataframe with error handling
    
    Args:
        df (DataFrame): DataFrame to preview
        num_rows (int): Number of rows to display
    """
    try:
        if df is None:
            st.warning("DataFrame is None")
            return
            
        if df.empty:
            st.warning("DataFrame is empty")
            return
            
        # Display column names
        st.write("**Columns:**", df.columns.tolist())
        
        # Display head
        st.write("**Data Preview:**")
        st.dataframe(df.head(num_rows))
        
        # Display dtypes
        st.write("**Column Types:**")
        dtypes_df = pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.astype(str)})
        st.dataframe(dtypes_df)
        
    except Exception as e:
        st.error(f"Error displaying DataFrame preview: {str(e)}")