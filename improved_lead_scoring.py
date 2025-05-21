"""
improved_lead_scoring.py - Enhanced lead scoring functionality

This module provides improved lead scoring with configurable weights,
category-specific scoring, and visual lead classification.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURABLE WEIGHTS (based on model feature importance) ---
WEIGHTS = {
    'number_of_guests': 3.5,
    'days_until_event': -2.0,
    'days_since_inquiry': -2.5,
    'bartenders_needed': 1.5,
    'referral_tier': 4.0,
    'is_corporate': 2.0,
    'phone_area_match': 1.0,
}

# --- SCORING FUNCTION ---
def compute_lead_score(row):
    """
    Compute a lead score based on configurable weights
    
    Args:
        row (Series): DataFrame row containing lead attributes
        
    Returns:
        float: Calculated lead score
    """
    score = 0
    
    # Add guest count contribution (logarithmic scaling)
    if 'number_of_guests' in row and pd.notna(row['number_of_guests']):
        score += WEIGHTS['number_of_guests'] * np.log1p(row['number_of_guests'])
    
    # Add days until event contribution (inverse relationship)
    if 'days_until_event' in row and pd.notna(row['days_until_event']):
        score += WEIGHTS['days_until_event'] * (1 / (1 + row['days_until_event']))
    
    # Add days since inquiry contribution (linear decay)
    if 'days_since_inquiry' in row and pd.notna(row['days_since_inquiry']):
        score += WEIGHTS['days_since_inquiry'] * (row['days_since_inquiry'] / 30)
    
    # Add bartenders needed contribution
    if 'bartenders_needed' in row and pd.notna(row['bartenders_needed']):
        score += WEIGHTS['bartenders_needed'] * row['bartenders_needed']
    
    # Add referral tier contribution
    if 'referral_tier' in row and pd.notna(row['referral_tier']):
        score += WEIGHTS['referral_tier'] * row['referral_tier']
    
    # Add corporate event contribution
    if 'is_corporate' in row and pd.notna(row['is_corporate']):
        score += WEIGHTS['is_corporate'] * int(row['is_corporate'])
    
    # Add phone area match contribution
    if 'phone_area_match' in row and pd.notna(row['phone_area_match']):
        score += WEIGHTS['phone_area_match'] * int(row['phone_area_match'])
    
    return score

# --- LEAD CATEGORY CLASSIFICATION ---
def classify_score(score):
    """
    Classify a lead score into a category with emoji
    
    Args:
        score (float): Lead score value
        
    Returns:
        str: Category with emoji indicator
    """
    if score >= 12:
        return '🔥 Hot'
    elif score >= 8:
        return '🌤️ Warm'
    elif score >= 4:
        return '❄️ Cool'
    else:
        return '🧊 Cold'

# --- MAIN PROCESSING FUNCTION ---
def process_lead_scoring(df):
    """
    Process dataframe to calculate lead scores and categories
    
    Args:
        df (DataFrame): Input dataframe with lead data
        
    Returns:
        DataFrame: Processed dataframe with lead scores and categories
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Define required columns and use those available
    required_columns = ['number_of_guests', 'days_until_event', 'days_since_inquiry',
                        'bartenders_needed', 'referral_tier', 'is_corporate', 'phone_area_match']
    available_columns = [col for col in required_columns if col in df.columns]
    
    # Check if we have enough data to score
    if len(available_columns) < 3:
        st.warning("Not enough data columns to calculate scores reliably.")
        return result_df
    
    # Drop rows with missing values in available columns
    result_df = result_df.dropna(subset=available_columns)
    
    # Calculate lead scores
    result_df['lead_score'] = result_df.apply(compute_lead_score, axis=1)
    
    # Add score categories
    result_df['score_category'] = result_df['lead_score'].apply(classify_score)
    
    return result_df

# --- VISUALIZATION FUNCTIONS ---
def plot_lead_score_distribution(df):
    """
    Plot the distribution of lead scores with category highlighting
    
    Args:
        df (DataFrame): Dataframe with lead_score and score_category columns
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if 'lead_score' not in df.columns:
        return None
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(df['lead_score'], bins=20, kde=True, ax=ax)
    
    # Add category thresholds
    thresholds = [4, 8, 12]
    colors = ['blue', 'cyan', 'orange', 'red']
    labels = ['Cold', 'Cool', 'Warm', 'Hot']
    
    # Shade regions for different categories
    xmin, xmax = ax.get_xlim()
    boundaries = [xmin] + thresholds + [xmax]
    
    for i in range(len(boundaries) - 1):
        ax.axvspan(
            boundaries[i], 
            boundaries[i+1], 
            alpha=0.2, 
            color=colors[i],
            label=labels[i]
        )
    
    # Customize plot
    ax.set_title('Lead Score Distribution by Category', fontsize=14)
    ax.set_xlabel('Lead Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(title='Lead Categories')
    
    return fig

# --- STREAMLIT UI FUNCTION ---
def render_lead_scoring_ui(df):
    """
    Render the lead scoring UI in Streamlit
    
    Args:
        df (DataFrame): Input dataframe with lead data
    """
    st.header("🤖 Lead Scoring Calculator")

    # Process lead scoring
    scored_df = process_lead_scoring(df)
    
    # Show top scoring leads
    st.subheader("Top 10 High Scoring Leads")
    
    if 'lead_score' in scored_df.columns:
        # Select display columns
        display_cols = ['lead_score', 'score_category']
        
        # Add name and email if available
        for col in ['name', 'email', 'inquiry_date', 'event_date', 'booking_type']:
            if col in scored_df.columns:
                display_cols.insert(0, col)
        
        # Show top leads
        st.dataframe(
            scored_df.sort_values(by='lead_score', ascending=False)
            .head(10)[display_cols]
        )
        
        # Show score distribution
        st.subheader("Lead Score Distribution")
        fig = plot_lead_score_distribution(scored_df)
        if fig:
            st.pyplot(fig)
        
        # Show category counts
        if 'score_category' in scored_df.columns:
            category_counts = scored_df['score_category'].value_counts()
            st.bar_chart(category_counts)
    
    # Manual lead scoring form
    st.subheader("Manual Lead Scoring Test")
    with st.form("manual_test"):
        # Form inputs
        col1, col2 = st.columns(2)
        
        with col1:
            guests = st.number_input("Guests", 0, 1000, 50)
            days_until = st.number_input("Days Until Event", 0, 365, 30)
            days_since = st.number_input("Days Since Inquiry", 0, 365, 1)
            bartenders = st.slider("Bartenders Needed", 0, 10, 2)
        
        with col2:
            tier = st.slider("Referral Tier (1–3)", 1, 3, 2)
            corp = st.checkbox("Corporate Event")
            phone_match = st.checkbox("Phone Area Code Matches State")
        
        submitted = st.form_submit_button("Calculate Score")

        if submitted:
            # Create test row with form data
            test_row = pd.Series({
                'number_of_guests': guests,
                'days_until_event': days_until,
                'days_since_inquiry': days_since,
                'bartenders_needed': bartenders,
                'referral_tier': tier,
                'is_corporate': corp,
                'phone_area_match': phone_match,
            })
            
            # Calculate score
            score = compute_lead_score(test_row)
            category = classify_score(score)
            
            # Display result with styling
            if '🔥' in category:
                st.success(f"Lead Score: {score:.2f} → {category}")
            elif '🌤️' in category:
                st.warning(f"Lead Score: {score:.2f} → {category}")
            elif '❄️' in category:
                st.info(f"Lead Score: {score:.2f} → {category}")
            else:
                st.error(f"Lead Score: {score:.2f} → {category}")
            
            # Explain score components
            st.subheader("Score Breakdown")
            components = []
            
            if 'number_of_guests' in test_row:
                val = WEIGHTS['number_of_guests'] * np.log1p(test_row['number_of_guests'])
                components.append(f"Guest Count: {val:.2f}")
                
            if 'days_until_event' in test_row:
                val = WEIGHTS['days_until_event'] * (1 / (1 + test_row['days_until_event']))
                components.append(f"Days Until Event: {val:.2f}")
                
            if 'days_since_inquiry' in test_row:
                val = WEIGHTS['days_since_inquiry'] * (test_row['days_since_inquiry'] / 30)
                components.append(f"Days Since Inquiry: {val:.2f}")
                
            if 'bartenders_needed' in test_row:
                val = WEIGHTS['bartenders_needed'] * test_row['bartenders_needed']
                components.append(f"Bartenders Needed: {val:.2f}")
                
            if 'referral_tier' in test_row:
                val = WEIGHTS['referral_tier'] * test_row['referral_tier']
                components.append(f"Referral Tier: {val:.2f}")
                
            if 'is_corporate' in test_row:
                val = WEIGHTS['is_corporate'] * int(test_row['is_corporate'])
                components.append(f"Corporate Event: {val:.2f}")
                
            if 'phone_area_match' in test_row:
                val = WEIGHTS['phone_area_match'] * int(test_row['phone_area_match'])
                components.append(f"Phone Area Match: {val:.2f}")
            
            # Display components
            for comp in components:
                st.text(comp)