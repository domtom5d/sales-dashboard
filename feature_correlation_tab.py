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

def calculate_correlations(df, outcome_col='outcome', numeric_cols=None):
    """
    Calculate correlation between features and outcome
    
    Args:
        df (DataFrame): DataFrame with numeric features and outcome
        outcome_col (str): Name of the outcome column (default: 'outcome')
        numeric_cols (list, optional): List of numeric columns to use for correlation.
                                      If None, will automatically select numeric columns.
    
    Returns:
        tuple: (
            DataFrame: Sorted correlations with outcome,
            DataFrame: Full correlation matrix
        )
    """
    # If no numeric columns specified, automatically select numeric ones
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Make sure outcome_col is in numeric_cols
    if outcome_col not in numeric_cols and outcome_col in df.columns:
        numeric_cols.append(outcome_col)
    
    # Validate that we have enough data
    if len(numeric_cols) <= 1:
        return None, None
    
    # Create a copy of the DataFrame with only numeric columns
    numeric_df = df[numeric_cols].copy()
    
    # Convert all columns to numeric, replacing non-numeric values with NaN
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    
    # Calculate correlation matrix using Pearson correlation
    corr_matrix = numeric_df.corr(method='pearson')
    
    # Extract correlations with outcome
    if outcome_col in corr_matrix.columns:
        # Get correlations with outcome and drop the outcome's self-correlation
        outcome_corr = corr_matrix[outcome_col].drop(outcome_col).reset_index()
        # Rename columns
        outcome_corr.columns = ['feature', 'correlation']
        # Sort by absolute correlation value
        outcome_corr = outcome_corr.sort_values('correlation', key=abs, ascending=False)
        
        return outcome_corr, corr_matrix
    else:
        return None, corr_matrix

def render_feature_correlation_tab(df):
    """
    Render the complete Feature Correlation Analysis tab
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    """
    st.subheader("Feature Correlation Analysis")
    st.markdown("This analysis helps you understand which factors most strongly correlate with won and lost deals.")
    
    # Handle both uppercase and lowercase outcome column
    outcome_col = None
    if 'Outcome' in df.columns:
        outcome_col = 'Outcome'
    elif 'outcome' in df.columns:
        outcome_col = 'outcome'
    
    if outcome_col is None:
        st.warning("No outcome column found in the data. This tab requires either 'outcome' or 'Outcome' column.")
        return
    
    # Select only numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if outcome_col not in numeric_cols:
        st.warning(f"The {outcome_col} column is not numeric. This tab requires a numeric outcome column (0 for lost, 1 for won).")
        return
    
    if len(numeric_cols) <= 1:
        st.warning("Not enough numeric columns available for correlation analysis.")
        return
    
    try:
        # Create a copy with just numeric columns
        numeric_df = df[numeric_cols].copy()
        
        # Make sure all columns are properly converted to numeric
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        
        # Drop columns with too many NaN values or constant values
        for col in numeric_df.columns:
            if numeric_df[col].nunique() <= 1 or numeric_df[col].isna().mean() > 0.5:
                if col != outcome_col:  # Keep the outcome column regardless
                    numeric_df = numeric_df.drop(columns=[col])
        
        # Drop columns that are just IDs or keys
        for col in numeric_df.columns:
            if 'key' in col.lower() or 'id' in col.lower():
                if col != outcome_col:  # Keep the outcome column regardless
                    numeric_df = numeric_df.drop(columns=[col])
        
        # Drop rows with NaN in outcome column
        numeric_df = numeric_df.dropna(subset=[outcome_col])
        
        # Use calculate_correlations function to get correlation data
        outcome_corr, corr_matrix = calculate_correlations(numeric_df, outcome_col=outcome_col)
        
        if outcome_corr is not None:
            # Rename columns to match existing code
            outcome_corr.columns = ['Feature', 'Correlation']
            
            # Remove near-zero correlations
            outcome_corr = outcome_corr[outcome_corr['Correlation'].abs() > 0.01]
            
            if not outcome_corr.empty:
                st.markdown("### Correlation with Deal Outcome")
                st.markdown("Positive values indicate factors that correlate with winning deals, negative values with losing deals.")
                
                # Create a horizontal bar chart of correlations
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create color map (blue for positive, red for negative)
                colors = ['#1E88E5' if c >= 0 else '#f44336' for c in outcome_corr['Correlation']]
                
                # Plot data
                ax.barh(outcome_corr['Feature'], outcome_corr['Correlation'], color=colors)
                
                # Customize plot
                ax.set_xlabel('Correlation with Win/Loss')
                ax.set_ylabel('Feature')
                ax.set_title('Feature Correlation with Deal Outcome')
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                # Add labels with the actual correlation values
                for i, v in enumerate(outcome_corr['Correlation']):
                    ax.text(v + (0.01 if v >= 0 else -0.01), 
                            i, 
                            f"{v:.2f}", 
                            va='center', 
                            ha='left' if v >= 0 else 'right',
                            fontweight='bold')
                
                # Only show plot if we have data
                st.pyplot(fig)
                
                # Display feature explanations
                st.markdown("### Feature Explanations")
                
                # Get top positive and negative correlations
                positive_corrs = outcome_corr[outcome_corr['Correlation'] > 0].head(3)
                negative_corrs = outcome_corr[outcome_corr['Correlation'] < 0].head(3)
                
                # Show positive correlations
                if not positive_corrs.empty:
                    st.markdown("#### Top factors associated with WON deals:")
                    for i, row in positive_corrs.iterrows():
                        st.markdown(f"- **{row['Feature']}**: Correlation {row['Correlation']:.2f} - Higher values are associated with won deals")
                
                # Show negative correlations
                if not negative_corrs.empty:
                    st.markdown("#### Top factors associated with LOST deals:")
                    for i, row in negative_corrs.iterrows():
                        st.markdown(f"- **{row['Feature']}**: Correlation {row['Correlation']:.2f} - Higher values are associated with lost deals")
                
                # Create correlation matrix heatmap if we have enough features
                if len(corr_matrix) > 2:
                    st.markdown("### Correlation Matrix")
                    st.markdown("This heatmap shows how different features correlate with each other.")
                    
                    # Limit to top correlated features for readability
                    top_features = list(outcome_corr['Feature'].head(8))
                    if outcome_col not in top_features:
                        top_features.append(outcome_col)
                    
                    # Create a subset correlation matrix
                    subset_corr = corr_matrix.loc[top_features, top_features]
                    
                    # Create a correlation heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Generate a custom diverging colormap
                    cmap = sns.diverging_palette(230, 20, as_cmap=True)
                    
                    # Draw the heatmap
                    sns.heatmap(subset_corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, 
                                annot=True, fmt=".2f", ax=ax)
                    
                    # Rotate x-axis labels for readability
                    plt.xticks(rotation=45, ha='right')
                    
                    # Set plot title
                    ax.set_title('Feature Correlation Matrix')
                    
                    # Display the plot
                    st.pyplot(fig)
            else:
                st.warning("No significant correlations found between features and the outcome.")
        else:
            st.warning(f"The {outcome_col} column was not included in the correlation matrix.")
    
    except Exception as e:
        st.error(f"Error performing correlation analysis: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

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