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
    
    # Check if there's any data
    if df.empty:
        st.warning("No data available for correlation analysis. Please ensure your dataset is loaded correctly.")
        return
        
    # Select only numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Display empty state with helpful message if no numeric features are found
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found in the dataset. Correlation analysis requires numeric data.")
        
        # Show helpful suggestions
        st.markdown("""
        ### Suggestions to fix this issue:
        1. Ensure your data includes numeric columns (e.g., price, quantity, days between events)
        2. Convert categorical variables to numeric if appropriate (e.g., yes/no to 1/0)
        3. Check that data types are correctly identified during import
        """)
        return
    
    if outcome_col not in numeric_cols:
        st.warning(f"The {outcome_col} column is not numeric. This tab requires a numeric outcome column (0 for lost, 1 for won).")
        
        # Show sample of outcome column to help diagnose
        if outcome_col in df.columns:
            st.markdown(f"**Sample values in '{outcome_col}' column:**")
            st.write(df[outcome_col].value_counts().head(10))
            st.markdown("The outcome column should be numeric with values 0 (lost) and 1 (won).")
        return
    
    if len(numeric_cols) <= 1:
        st.warning("Not enough numeric columns available for correlation analysis. At least two numeric columns are needed.")
        
        # Show available columns for reference
        st.markdown("**Available columns in dataset:**")
        for col_type, columns in df.dtypes.groupby(df.dtypes).items():
            st.markdown(f"- {col_type}: {', '.join(columns.index.tolist())}")
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
                
                # Create a horizontal bar chart of correlations using seaborn
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Limit to top 15 correlations for readability
                top_corrs = outcome_corr.head(15)
                
                # Create color palette - blue for positive, red for negative
                colors = ['#1E88E5' if c >= 0 else '#f44336' for c in top_corrs['Correlation']]
                
                # Plot with seaborn barplot
                sns.barplot(
                    x='Correlation', 
                    y='Feature',
                    data=top_corrs,
                    palette=colors,
                    ax=ax
                )
                
                # Customize plot
                ax.set_xlabel('Correlation with Win/Loss')
                ax.set_ylabel('Feature')
                ax.set_title('Top Feature Correlations with Deal Outcome')
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                # Add labels with the actual correlation values
                for i, v in enumerate(top_corrs['Correlation']):
                    ax.text(v + (0.01 if v >= 0 else -0.01), 
                            i, 
                            f"{v:.2f}", 
                            va='center', 
                            ha='left' if v >= 0 else 'right',
                            fontweight='bold')
                
                # Display number of features analyzed
                st.caption(f"Showing top {len(top_corrs)} of {len(outcome_corr)} features with significant correlation.")
                
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
                    
                    # Add option to choose number of features to display
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        max_features = min(15, len(outcome_corr))
                        num_features = st.slider("Number of features", 3, max_features, min(8, max_features))
                        
                    with col2:
                        matrix_display = st.radio(
                            "Display format",
                            options=["Heatmap", "Table"],
                            horizontal=True
                        )
                    
                    # Limit to top correlated features for readability
                    top_features = list(outcome_corr['Feature'].head(num_features))
                    if outcome_col not in top_features:
                        top_features.append(outcome_col)
                    
                    # Create a subset correlation matrix
                    subset_corr = corr_matrix.loc[top_features, top_features]
                    
                    # Display as chosen format
                    if matrix_display == "Heatmap":
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
                        # Format table with highlighting
                        def color_corr(val):
                            """Color function for correlation values in dataframe styling"""
                            color = '#B7E0F2' if val > 0 else '#F2B8C6' if val < 0 else 'white'
                            intensity = min(abs(val) * 2, 1)  # Scale for color intensity
                            return f'background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {intensity})'
                        
                        # Display as styled table with formatting
                        st.dataframe(
                            subset_corr.style.format("{:.2f}")
                                          .applymap(color_corr)
                                          .set_caption("Correlation Matrix"),
                            use_container_width=True
                        )
                    
                    # Add explanation for how to interpret the matrix
                    st.caption("The correlation matrix shows relationships between all pairs of features. Values close to 1 or -1 indicate strong correlations.")
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