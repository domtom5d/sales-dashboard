"""
raw_data_tab.py - Raw Data Tab Module

This module provides the implementation for the Raw Data tab
in the Sales Conversion Analytics Dashboard.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def render_raw_data_tab(df):
    """
    Render the Raw Data tab with dataframe display and download options
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    """
    st.subheader("Raw Data")
    st.markdown("Explore the complete, unfiltered dataset used for all analyses.")
    
    # Get the complete dataset from session state if available
    # This ensures we're showing the full unfiltered dataset
    if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
        display_df = st.session_state.processed_df
    else:
        display_df = df  # Fall back to filtered df if full dataset not available
    
    # Check if we have any data to display
    if display_df is None or display_df.empty:
        st.warning("No data loaded. Please upload CSV files or connect to the database.")
        return
    
    # Show data dimensions and overview
    row_count = len(display_df)
    col_count = len(display_df.columns)
    st.info(f"Dataset contains {row_count:,} rows and {col_count} columns.")
    
    # Display the dataframe with scrollable interface and size limits
    st.dataframe(display_df, height=400)
    
    # Download options
    st.markdown("### Download Data")
    
    # Create download button that uses st.download_button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download Full Dataset (CSV)",
        data=csv,
        file_name="sales_conversion_data.csv",
        mime="text/csv"
    )
    
    # Show data info
    with st.expander("Data Information"):
        # Show column descriptions
        st.markdown("#### Column Descriptions")
        col_desc = pd.DataFrame({
            'Column': display_df.columns.tolist(),
            'Type': display_df.dtypes.astype(str).tolist(),
            'Non-Null Count': display_df.count().tolist(),
            'Null Count': display_df.isna().sum().tolist(),
            'Non-Null %': (display_df.count() / len(display_df) * 100).round(2).astype(str) + '%'
        })
        st.dataframe(col_desc)
        
        # Display summary statistics for numeric columns
        st.markdown("#### Numeric Columns Summary")
        numeric_cols = display_df.select_dtypes(include=['int64', 'float64'])
        if not numeric_cols.empty:
            st.dataframe(numeric_cols.describe())
        else:
            st.info("No numeric columns found in the dataset.")
        
        # Display categorical columns value counts
        st.markdown("#### Categorical Columns")
        categorical_cols = display_df.select_dtypes(include=['object', 'category'])
        
        if not categorical_cols.empty:
            cat_col = st.selectbox("Select a categorical column to view value counts:", 
                                   options=categorical_cols.columns.tolist())
            
            if cat_col:
                # Get value counts
                value_counts = display_df[cat_col].value_counts().reset_index()
                value_counts.columns = [cat_col, 'Count']
                
                # Calculate percentage
                value_counts['Percentage'] = (value_counts['Count'] / len(display_df) * 100).round(2)
                value_counts['Percentage'] = value_counts['Percentage'].astype(str) + '%'
                
                # Display
                st.dataframe(value_counts)
        else:
            st.info("No categorical columns found in the dataset.")

    # Data quality assessment
    with st.expander("Data Quality Assessment"):
        st.markdown("#### Missing Values")
        
        # Count missing values by column
        missing = display_df.isna().sum().reset_index()
        missing.columns = ['Column', 'Missing Count']
        missing['Missing Percentage'] = (missing['Missing Count'] / len(display_df) * 100).round(2)
        missing['Missing Percentage'] = missing['Missing Percentage'].astype(str) + '%'
        missing = missing.sort_values(by='Missing Count', ascending=False)
        
        # Only show columns with missing values
        missing = missing[missing['Missing Count'] > 0]
        
        if not missing.empty:
            st.dataframe(missing)
        else:
            st.success("No missing values found in the dataset!")
        
        # Duplicate rows check
        duplicate_count = display_df.duplicated().sum()
        if duplicate_count > 0:
            st.warning(f"Found {duplicate_count} duplicate rows in the dataset ({(duplicate_count / len(display_df) * 100).round(2)}% of data).")
        else:
            st.success("No duplicate rows found in the dataset!")
        
        # Outliers detection for numeric columns
        st.markdown("#### Potential Outliers")
        numeric_cols = display_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if numeric_cols:
            outlier_col = st.selectbox("Select a numeric column to check for outliers:", 
                                      options=numeric_cols)
            
            if outlier_col and outlier_col in display_df.columns:
                # Skip outlier detection if column has too many NaN values
                if display_df[outlier_col].isna().sum() > len(display_df) * 0.5:
                    st.warning(f"Column '{outlier_col}' has too many missing values for reliable outlier detection.")
                else:
                    try:
                        # Calculate quartiles and IQR
                        q1 = display_df[outlier_col].quantile(0.25)
                        q3 = display_df[outlier_col].quantile(0.75)
                        iqr = q3 - q1
                        
                        # Define outlier bounds
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        # Count outliers
                        outliers = display_df[(display_df[outlier_col] < lower_bound) | (display_df[outlier_col] > upper_bound)]
                        outlier_count = len(outliers)
                        
                        if outlier_count > 0:
                            st.warning(f"Found {outlier_count} potential outliers in column '{outlier_col}' ({(outlier_count / len(display_df) * 100).round(2)}% of data).")
                            st.markdown(f"**Lower bound:** {lower_bound:.2f}, **Upper bound:** {upper_bound:.2f}")
                            
                            # Show histogram with outlier bounds
                            fig, ax = plt.subplots()
                            display_df[outlier_col].hist(bins=30, ax=ax)
                            ax.axvline(lower_bound, color='r', linestyle='--')
                            ax.axvline(upper_bound, color='r', linestyle='--')
                            ax.set_title(f"Distribution of {outlier_col} with Outlier Bounds")
                            st.pyplot(fig)
                            
                            # Show the outliers (limited to 100 rows for performance)
                            st.markdown("#### Sample Outlier Records (Up to 100)")
                            st.dataframe(outliers.head(100))
                        else:
                            st.success(f"No outliers detected in column '{outlier_col}'.")
                    except Exception as e:
                        st.error(f"Error analyzing outliers: {str(e)}")
        else:
            st.info("No numeric columns available for outlier detection.")