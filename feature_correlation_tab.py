"""
feature_correlation_tab.py - Feature Correlation Analysis Tab Module

This module provides the implementation for the Feature Correlation Analysis tab
in the Sales Conversion Analytics Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import calculate_correlations

def render_feature_correlation_tab(df):
    """
    Render the complete Feature Correlation Analysis tab
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    """
    st.subheader("Feature Correlation Analysis")
    st.markdown("This analysis helps you understand which factors most strongly correlate with won and lost deals.")
    
    # Add data shape debugging
    st.write(f"DEBUG: DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
    
    # Select only numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) > 0 and 'outcome' in df.columns:
        # Calculate and display correlation with outcome
        outcome_corr, full_corr = calculate_correlations(df)
        
        if not outcome_corr.empty:
            st.markdown("### Correlation with Deal Outcome")
            st.markdown("Positive values indicate factors that correlate with winning deals, negative values with losing deals.")
        
        # Create a horizontal bar chart of correlations if data is available
        if not outcome_corr.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort by absolute correlation value (handle different column names)
            correlation_col = 'Correlation with Outcome' if 'Correlation with Outcome' in outcome_corr.columns else 'correlation'
            feature_col = 'index' if 'index' in outcome_corr.columns else 'feature'
            
            # Make sure outcome_corr is sorted properly
            outcome_corr = outcome_corr.sort_values(by=correlation_col, key=abs, ascending=False)
            
            # Create color map (blue for positive, red for negative)
            colors = ['#1E88E5' if c >= 0 else '#f44336' for c in outcome_corr[correlation_col]]
            
            # Plot data
            ax.barh(outcome_corr[feature_col], outcome_corr[correlation_col], color=colors)
            
            # Customize plot
            ax.set_xlabel('Correlation with Win/Loss')
            ax.set_ylabel('Feature')
            ax.set_title('Feature Correlation with Deal Outcome')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add labels with the actual correlation values
            for i, v in enumerate(outcome_corr[correlation_col]):
                ax.text(v + (0.01 if v >= 0 else -0.01), 
                        i, 
                        f"{v:.2f}", 
                        va='center', 
                        ha='left' if v >= 0 else 'right',
                        fontweight='bold')
            
            # Only show plot if we have data
            st.pyplot(fig)
        else:
            st.warning("Not enough data to calculate correlations. Make sure your dataset has numeric features and outcome labels.")
        
        # Display feature explanations if we have correlations data
        if not outcome_corr.empty:
            st.markdown("### Feature Explanations")
            
            # Get the correlation column name
            correlation_col = 'Correlation with Outcome' if 'Correlation with Outcome' in outcome_corr.columns else 'correlation'
            feature_col = 'index' if 'index' in outcome_corr.columns else 'feature'
            
            for i, row in outcome_corr.iterrows():
                # Skip features with very low correlation
                corr_value = row[correlation_col]
                if abs(corr_value) < 0.05:
                    continue
                
                feature = row[feature_col]
                
                if corr_value > 0:
                    st.markdown(f"**{feature}**: Positive correlation ({corr_value:.2f}) - Higher values tend to be associated with **won deals**")
                else:
                    st.markdown(f"**{feature}**: Negative correlation ({corr_value:.2f}) - Higher values tend to be associated with **lost deals**")
        
        # Add correlation matrix heatmap if we have enough features
        if not full_corr.empty and len(full_corr) > 2:
            st.markdown("### Correlation Matrix")
            st.markdown("This heatmap shows how different features correlate with each other.")
            
            # Create a correlation heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(full_corr, dtype=bool))
            
            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            # Draw the heatmap with the mask and correct aspect ratio
            sns.heatmap(full_corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True, fmt=".2f")
            
            # Set plot title and labels
            ax.set_title('Feature Correlation Matrix')
            
            # Display the plot
            st.pyplot(fig)
    else:
        st.warning("Not enough numeric data available for correlation analysis. Make sure your dataset includes numeric features and outcome labels.")

    # Add guidance on how to interpret correlations
    with st.expander("How to interpret correlations"):
        st.markdown("""
        ### Understanding Correlation Values
        
        Correlation values range from -1 to 1:
        
        - **Values close to 1**: Strong positive correlation (as one value increases, the other tends to increase)
        - **Values close to -1**: Strong negative correlation (as one value increases, the other tends to decrease)
        - **Values close to 0**: Little to no correlation (no clear pattern)
        
        ### What This Means For Sales
        
        - **Positive correlations with outcome**: These factors are associated with winning deals. Consider prioritizing leads with high values in these areas.
        - **Negative correlations with outcome**: These factors are associated with losing deals. Leads with high values in these areas may require special attention or different approaches.
        
        Remember that correlation does not imply causation. These relationships show patterns in your historical data but don't necessarily indicate that one factor causes another.
        """)