"""
raw_data_tab.py - Raw Data Tab Module

This module provides the implementation for the Raw Data tab
in the Sales Conversion Analytics Dashboard.
"""

import streamlit as st
import pandas as pd
import base64

def render_raw_data_tab(df):
    """
    Render the Raw Data tab with dataframe display and download options
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    """
    st.subheader("Raw Data")
    st.markdown("Explore the underlying dataset used for all analyses.")
    
    # Add data shape debugging
    st.write(f"DEBUG: DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
    
    # Display the dataframe
    st.dataframe(df)
    
    # Download options
    st.markdown("### Download Data")
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sales_data.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # Show data info
    with st.expander("Data Information"):
        # Show column descriptions
        st.markdown("#### Column Descriptions")
        col_desc = pd.DataFrame({
            'Column': df.columns.tolist(),
            'Type': df.dtypes.astype(str).tolist(),
            'Non-Null Count': df.count().tolist(),
            'Null Count': df.isna().sum().tolist(),
            'Non-Null %': (df.count() / len(df) * 100).round(2).astype(str) + '%'
        })
        st.dataframe(col_desc)
        
        # Display summary statistics for numeric columns
        st.markdown("#### Numeric Columns Summary")
        numeric_cols = df.select_dtypes(include=['int64', 'float64'])
        if not numeric_cols.empty:
            st.dataframe(numeric_cols.describe())
        else:
            st.info("No numeric columns found in the dataset.")
        
        # Display categorical columns value counts
        st.markdown("#### Categorical Columns")
        categorical_cols = df.select_dtypes(include=['object', 'category'])
        
        if not categorical_cols.empty:
            cat_col = st.selectbox("Select a categorical column to view value counts:", 
                                   options=categorical_cols.columns.tolist())
            
            if cat_col:
                # Get value counts
                value_counts = df[cat_col].value_counts().reset_index()
                value_counts.columns = [cat_col, 'Count']
                
                # Calculate percentage
                value_counts['Percentage'] = (value_counts['Count'] / len(df) * 100).round(2)
                value_counts['Percentage'] = value_counts['Percentage'].astype(str) + '%'
                
                # Display
                st.dataframe(value_counts)
        else:
            st.info("No categorical columns found in the dataset.")

    # Data quality assessment
    with st.expander("Data Quality Assessment"):
        st.markdown("#### Missing Values")
        
        # Count missing values by column
        missing = df.isna().sum().reset_index()
        missing.columns = ['Column', 'Missing Count']
        missing['Missing Percentage'] = (missing['Missing Count'] / len(df) * 100).round(2)
        missing['Missing Percentage'] = missing['Missing Percentage'].astype(str) + '%'
        missing = missing.sort_values(by='Missing Count', ascending=False)
        
        # Only show columns with missing values
        missing = missing[missing['Missing Count'] > 0]
        
        if not missing.empty:
            st.dataframe(missing)
        else:
            st.success("No missing values found in the dataset!")
        
        # Duplicate rows check
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            st.warning(f"Found {duplicate_count} duplicate rows in the dataset ({(duplicate_count / len(df) * 100).round(2)}% of data).")
        else:
            st.success("No duplicate rows found in the dataset!")
        
        # Outliers detection for numeric columns
        st.markdown("#### Potential Outliers")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if numeric_cols:
            outlier_col = st.selectbox("Select a numeric column to check for outliers:", 
                                      options=numeric_cols)
            
            if outlier_col and outlier_col in df.columns:
                # Calculate quartiles and IQR
                q1 = df[outlier_col].quantile(0.25)
                q3 = df[outlier_col].quantile(0.75)
                iqr = q3 - q1
                
                # Define outlier bounds
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Count outliers
                outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    st.warning(f"Found {outlier_count} potential outliers in column '{outlier_col}' ({(outlier_count / len(df) * 100).round(2)}% of data).")
                    st.markdown(f"**Lower bound:** {lower_bound:.2f}, **Upper bound:** {upper_bound:.2f}")
                    
                    # Show histogram with outlier bounds
                    fig, ax = plt.subplots()
                    df[outlier_col].hist(bins=30, ax=ax)
                    ax.axvline(lower_bound, color='r', linestyle='--')
                    ax.axvline(upper_bound, color='r', linestyle='--')
                    ax.set_title(f"Distribution of {outlier_col} with Outlier Bounds")
                    st.pyplot(fig)
                    
                    # Show the outliers
                    st.markdown("#### Outlier Records")
                    st.dataframe(outliers)
                else:
                    st.success(f"No outliers detected in column '{outlier_col}'.")
        else:
            st.info("No numeric columns available for outlier detection.")