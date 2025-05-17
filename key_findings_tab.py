"""
key_findings_tab.py - Key Findings Tab Module

This module provides the implementation for the Key Findings tab
in the Sales Conversion Analytics Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from findings import generate_findings

def render_key_findings_tab(df):
    """
    Render the Key Findings tab with dynamically generated insights
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    """
    st.subheader("Key Findings")
    st.markdown("This tab highlights the most important insights from your data and model analysis.")
    
    # Check if we have model data in session state
    if ('model_metrics' in st.session_state and 
        'weights_df' in st.session_state and 
        'thresholds' in st.session_state):
        
        try:
            # Get data for findings
            y_scores = None
            thresholds = None
            
            # Extract model predictions from session state
            if 'model_metrics' in st.session_state and st.session_state.model_metrics is not None:
                metrics = st.session_state.model_metrics
                if 'y_pred_proba' in metrics:
                    y_scores = metrics['y_pred_proba']
                elif 'y_pred_proba' in st.session_state:
                    y_scores = st.session_state.y_pred_proba
                elif 'y_probs' in st.session_state:
                    y_scores = st.session_state.y_probs
            
            # Get thresholds from session state
            if 'thresholds' in st.session_state:
                thresholds = st.session_state.thresholds
            
            # Ensure we have the necessary data for findings
            if y_scores is None or thresholds is None:
                st.warning("Missing model scores or thresholds. Please regenerate the model in the Lead Scoring tab.")
                return
            
            # Get feature weights from session state
            weights_df = None
            if 'weights_df' in st.session_state and st.session_state.weights_df is not None:
                weights_df = st.session_state.weights_df
            
            # Generate dynamic findings
            findings = generate_findings(df, y_scores, thresholds, weights_df=weights_df)
            
            # Display findings with bullet points
            st.markdown("### Key Insights")
            
            if findings and len(findings) > 0:
                for finding in findings:
                    st.markdown(f"• {finding}")
                
                # Add explanation about feature importance
                if 'weights_df' in st.session_state and st.session_state.weights_df is not None:
                    weights_df = st.session_state.weights_df
                    if not weights_df.empty:
                        st.markdown("### Feature Importance Analysis")
                        
                        # Get top positive and negative features
                        weights_df = weights_df.sort_values(by='weight', key=abs, ascending=False)
                        positive_features = weights_df[weights_df['weight'] > 0].head(3)
                        negative_features = weights_df[weights_df['weight'] < 0].head(3)
                        
                        if not positive_features.empty:
                            st.markdown("**Factors that increase conversion probability:**")
                            for i, row in positive_features.iterrows():
                                st.markdown(f"• Higher **{row['feature']}** values strongly correlate with won deals (weight: +{row['weight']:.2f})")
                        
                        if not negative_features.empty:
                            st.markdown("**Factors that decrease conversion probability:**")
                            for i, row in negative_features.iterrows():
                                st.markdown(f"• Higher **{row['feature']}** values correlate with lost deals (weight: {row['weight']:.2f})")
                
                st.info("These findings are dynamically generated from your current data and will update as your data changes.")
            else:
                st.warning("Could not generate findings from the current dataset. Try filtering your data differently or regenerating the model.")
        
        except Exception as e:
            st.error(f"Error generating findings: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            st.info("Try clicking 'Generate Lead Scoring Model' on the Lead Scoring tab first if you haven't already.")
    
    else:
        st.info("Please run the Lead Scoring model to see key findings based on your data.")
        
        # Show example findings to demonstrate how the tab will look
        with st.expander("See example insights"):
            st.markdown("""
            The Key Findings tab will display insights like these, but specific to your data:
            
            • **Urgency:** Leads closing within 7 days convert at 45%, vs. those >30 days at 10%.
            • **Geography:** Region A leads close at 38%, while Region B at 18%.
            • **Seasonality:** July month has 32% conversion rate, lowest is January at 14%.
            • **Event Type:** Corporate events convert at 28%, Social events at 20%.
            • **Phone‐Match:** Local numbers convert at 16% vs. non‐local at 10%.
            • **Time to Conversion:** Average: 12.5 days, Median: 8.0 days.
            • **Event Type Conversion Speed:** Corporate events convert fastest (8.5 days), while Weddings take longest (18.2 days).
            """)
    
    # Add additional explanation of how to use these findings
    with st.expander("How to use these findings"):
        st.markdown("""
        ### Using Key Findings in Your Sales Strategy
        
        The insights on this page are calculated by analyzing your historical sales data and the predictive model. Here's how to use this information:
        
        1. **Prioritize high-probability segments**: Focus your sales efforts on leads that match the characteristics of your best-converting segments.
        
        2. **Adjust your approach based on timing**: If certain time windows show higher conversion rates, consider adjusting your follow-up cadence accordingly.
        
        3. **Segment your outreach**: Different types of leads (corporate vs. social, local vs. non-local) may require different messaging and sales approaches.
        
        4. **Monitor seasonal trends**: Plan your sales and marketing efforts around seasonal patterns identified in the data.
        
        5. **Address low-performing segments**: For segments with particularly low conversion rates, consider testing new approaches or qualification criteria.
        
        These findings will update automatically when you regenerate the model or filter your data differently.
        """)