"""
enhanced_lead_scoring_tab.py - Advanced Lead Scoring Tab with improved visualizations

This module provides an enhanced implementation of the Lead Scoring tab
that incorporates category-specific decay, regional influence, and
color-coded lead categories.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enhanced_lead_scoring import (
    train_enhanced_lead_scoring_model,
    score_lead_enhanced,
    plot_enhanced_lead_score_visualization
)

def render_enhanced_lead_scoring_tab(df):
    """
    Render the enhanced Lead Scoring tab
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    """
    st.subheader("Advanced Lead Scoring Model")
    st.markdown("Use this model to evaluate new leads and prioritize the most promising opportunities.")
    
    # Add explanation of enhanced features
    with st.expander("About the Enhanced Lead Scoring", expanded=False):
        st.markdown("""
        ### Advanced Lead Scoring Features
        
        This lead scoring model improves upon the basic scoring with:
        
        1. **Category-aware decay functions** - Each event type has its own half-life for lead freshness
        2. **Regional conversion boost** - Leads from high-performing regions get a score boost
        3. **Smart referral source tiering** - Referral sources are grouped into tiers based on past performance
        4. **Staff-to-guest ratio optimization** - Finds the optimal staffing ratio for your business
        5. **Color-coded lead categories** - Clear "Hot", "Warm", "Cool", and "Cold" categories with thresholds
        """)
        
        st.markdown("""
        ### How to Use This Tool
        
        1. Click **Generate Enhanced Model** to analyze your historical data
        2. View the Feature Importance chart to understand what drives conversions
        3. Use the Lead Scoring Calculator to score and categorize new leads
        4. Note how the score decays over time based on the lead's event type
        """)
    
    if 'outcome' not in df.columns:
        st.warning("Outcome column not found in the data. Lead scoring requires a column indicating won/lost status.")
        return
    
    # Create a button to generate or regenerate the model
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Generate Enhanced Lead Scoring Model")
        st.markdown("This analyzes your historical data to find what factors predict successful deals.")
    
    with col2:
        generate_button = st.button(
            "Generate Enhanced Model", 
            help="Click to train a machine learning model on your data to predict which leads are most likely to convert",
            key="generate_enhanced_model"
        )
    
    # Generate model if button is clicked or if not already in session state
    if generate_button or 'enhanced_model' not in st.session_state:
        try:
            with st.spinner("Generating enhanced lead scoring model..."):
                # Generate model
                model, scaler, feature_weights, thresholds, metrics = train_enhanced_lead_scoring_model(df)
                
                # Store in session state
                st.session_state.enhanced_model = model
                st.session_state.enhanced_scaler = scaler
                st.session_state.enhanced_weights = feature_weights
                st.session_state.enhanced_thresholds = thresholds
                st.session_state.enhanced_metrics = metrics
                
                # Display category half-lives if available
                if metrics and 'category_half_lives' in metrics and metrics['category_half_lives'] is not None:
                    category_half_lives = metrics['category_half_lives']
                    category_field = metrics['category_field']
                    
                    with st.expander("Category-Specific Sales Cycle Lengths", expanded=True):
                        st.markdown(f"**Using '{category_field}' for category-specific decay rates**")
                        st.markdown("Each event type has its own typical sales cycle length:")
                        
                        # Create dataframe for display
                        half_life_df = pd.DataFrame({
                            'Category': category_half_lives.index,
                            'Half-Life (Days)': category_half_lives.values.round(1)
                        })
                        
                        st.dataframe(half_life_df, use_container_width=True)
                        
                        # Show explanation
                        st.markdown("""
                        **Why this matters**: Leads decay at different rates depending on the event type. 
                        For example, wedding leads might stay viable for 45 days, while corporate events 
                        may convert much faster (14 days). The model now uses these category-specific
                        half-lives for more accurate lead scoring.
                        """)
                
                st.success("Enhanced lead scoring model generated successfully!")
        except Exception as e:
            st.error(f"Error generating enhanced lead scoring model: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            return
    else:
        # Use existing model from session state
        if 'enhanced_model' in st.session_state and st.session_state.enhanced_model is not None:
            model = st.session_state.enhanced_model
            scaler = st.session_state.enhanced_scaler
            feature_weights = st.session_state.enhanced_weights
            thresholds = st.session_state.enhanced_thresholds
            metrics = st.session_state.enhanced_metrics
        else:
            st.warning("No enhanced lead scoring model found. Please click 'Generate Enhanced Model' to create one.")
            return
    
    # Make sure we have valid data to display
    if feature_weights is None or feature_weights.empty:
        st.warning("Unable to generate lead scoring model. Please ensure you have sufficient data with clear outcomes.")
        return
    
    # Display model metrics with better visualization
    st.markdown("### Model Performance")
    
    # Create quality assessment based on ROC AUC
    quality_text = ""
    quality_color = ""
    if metrics and 'roc_auc' in metrics:
        roc_auc = metrics['roc_auc']
        if roc_auc >= 0.8:
            quality_text = "Excellent"
            quality_color = "green"
        elif roc_auc >= 0.7:
            quality_text = "Good"
            quality_color = "lightgreen"
        elif roc_auc >= 0.6:
            quality_text = "Fair"
            quality_color = "orange"
        else:
            quality_text = "Poor"
            quality_color = "red"
    
    # Show metrics in columns with explanations
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if metrics and 'roc_auc' in metrics:
            auc_value = metrics['roc_auc']
            st.metric("Model Quality", f"{quality_text}", delta=None)
            st.caption(f"ROC AUC: {auc_value:.3f}")
    
    with col2:
        if metrics and 'accuracy' in metrics:
            accuracy = metrics['accuracy']
            st.metric("Accuracy", f"{accuracy:.1%}")
            st.caption("Percentage of correct predictions")
    
    with col3:
        if metrics and 'f1' in metrics:
            f1 = metrics['f1']
            st.metric("F1 Score", f"{f1:.3f}")
            st.caption("Balance of precision and recall")
    
    with col4:
        if metrics and 'category_half_lives' in metrics and metrics['category_half_lives'] is not None:
            avg_half_life = metrics['category_half_lives'].mean()
            st.metric("Avg. Half-Life", f"{avg_half_life:.1f} days")
            st.caption("How fast leads decay on average")
    
    # Display feature weights in a more informative way
    st.markdown("### Key Conversion Drivers")
    st.markdown("These factors have the strongest influence on predicting won deals.")
    
    # Check if feature_weights is a DataFrame with the expected structure
    if isinstance(feature_weights, pd.DataFrame) and 'weight' in feature_weights.columns and 'feature' in feature_weights.columns:
        # Sort by absolute weight
        feature_weights = feature_weights.sort_values(by='weight', key=abs, ascending=False)
        
        # Create horizontal bar chart with enhanced styling
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Limit to top 15 features for readability
        display_weights = feature_weights.head(15)
        
        if not display_weights.empty:
            # Create color map (blue for positive, red for negative)
            colors = ['#1E88E5' if w >= 0 else '#f44336' for w in display_weights['weight']]
            
            # Plot data
            y_pos = np.arange(len(display_weights))
            ax.barh(y_pos, display_weights['weight'], color=colors)
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(display_weights['feature'])
            ax.set_xlabel('Weight (positive = increases win probability)')
            ax.set_title('Top Factors in Lead Scoring Model', fontsize=14, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Check if there are weights to plot
            if not display_weights['weight'].empty and max(abs(display_weights['weight'])) > 0:
                # Add labels with the actual weight values
                for i, v in enumerate(display_weights['weight']):
                    label_color = 'black'
                    direction = '+' if v >= 0 else ''  # Add plus sign for positive values for clarity
                    ax.text(v + (0.01 if v >= 0 else -0.01) * max(abs(display_weights['weight'])), 
                            i, 
                            f"{direction}{v:.2f}", 
                            va='center', 
                            ha='left' if v >= 0 else 'right',
                            fontweight='bold',
                            color=label_color)
            
            st.pyplot(fig)
        else:
            st.warning("No feature importance data available to display.")
    elif isinstance(feature_weights, pd.Series):
        # Convert series to dataframe for display
        df_weights = pd.DataFrame({
            'feature': feature_weights.index,
            'importance': feature_weights.values
        }).sort_values(by='importance', ascending=False)
        
        st.subheader("Top Features Driving Conversions")
        st.dataframe(df_weights.head(10), use_container_width=True)
    else:
        st.warning("Feature importance data not in expected format. Please regenerate the model.")
    
    # Score distribution plot
    if metrics is not None:
        st.markdown("### Lead Score Distribution")
        st.markdown("This shows how your won and lost leads score in the model, and where the thresholds are set.")
        
        try:
            # Check if we have the required data for visualization
            if 'won_scores' not in metrics or 'lost_scores' not in metrics:
                st.warning("Missing won/lost scores data required for visualization.")
                # Debug info
                st.write("Available metrics keys:", list(metrics.keys()))
            elif len(metrics['won_scores']) == 0 and len(metrics['lost_scores']) == 0:
                st.warning("Not enough data to plot score distribution. Need more won and lost leads.")
            else:
                # Use our enhanced visualization
                fig = plot_enhanced_lead_score_visualization(metrics, "Lead Score Distribution by Outcome")
                st.pyplot(fig)
                
                # Add explanation of categories
                st.markdown("""
                ### Lead Categories Explanation
                
                - **Hot Leads** (Red zone): High probability to close. These leads should be your top priority.
                - **Warm Leads** (Orange zone): Good potential but need attention. Follow up soon to convert.
                - **Cool Leads** (Cyan zone): Limited potential but still possible. Worth a follow-up if time permits.
                - **Cold Leads** (Blue zone): Very unlikely to convert. Consider these low priority.
                """)
        except Exception as e:
            st.error(f"Error plotting score distributions: {str(e)}")
            # Show stack trace for debugging
            import traceback
            st.text(traceback.format_exc())
    
    # Enhanced lead scoring calculator
    st.markdown("### Lead Scoring Calculator")
    st.markdown("Enter values for a new lead to calculate its score and conversion probability.")
    
    # Create form for lead scoring
    with st.form("enhanced_lead_scoring_form"):
        # Get top features based on importance
        top_features = feature_weights.head(10)['feature'].tolist()
        
        # Create columns for the form
        form_col1, form_col2 = st.columns(2)
        
        # Store input values
        lead_data = {}
        
        # Days since inquiry (critical for decay calculation)
        days_since_field = None
        for field in ['days_since_inquiry', 'days_since', 'DaysSinceInquiry']:
            if field in top_features:
                days_since_field = field
                break
        
        if days_since_field is None:
            # Add it anyway as it's needed for decay
            days_since_field = 'days_since_inquiry'
            
        # Add this field first as it's critical
        lead_data[days_since_field] = form_col1.number_input(
            "Days Since Inquiry", 
            min_value=0, 
            value=7, 
            help="Number of days since the lead first inquired"
        )
        
        # Event category (for category-specific decay)
        category_field = metrics.get('category_field') if metrics else None
        
        if category_field:
            # Get unique categories from the data
            categories = df[category_field].dropna().unique().tolist()
            categories = [c for c in categories if c and str(c).strip()]
            
            if categories:
                selected_category = form_col2.selectbox(
                    f"Event Category ({category_field})",
                    options=categories,
                    help="Select the event category for this lead"
                )
                lead_data[category_field] = selected_category
        
        # Create form fields for remaining top features
        added_fields = [days_since_field, category_field] if category_field else [days_since_field]
        
        for i, feature in enumerate(top_features):
            if feature in added_fields:
                continue
                
            # Alternate between columns
            col = form_col1 if i % 2 == 0 else form_col2
            
            # Set input field based on feature name
            if 'days' in feature.lower() or 'guest' in feature.lower() or 'count' in feature.lower() or 'bartender' in feature.lower():
                # Integer input
                lead_data[feature] = col.number_input(
                    f"{feature.replace('_', ' ').title()}", 
                    min_value=0, 
                    step=1
                )
            elif 'price' in feature.lower() or 'value' in feature.lower() or 'revenue' in feature.lower() or 'cost' in feature.lower():
                # Monetary input
                lead_data[feature] = col.number_input(
                    f"{feature.replace('_', ' ').title()}", 
                    min_value=0.0, 
                    step=100.0,
                    format="%.2f"
                )
            elif any(term in feature.lower() for term in ['ratio', 'percentage', 'rate', 'index']):
                # Ratio input
                lead_data[feature] = col.number_input(
                    f"{feature.replace('_', ' ').title()}",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f"
                )
            elif feature in df.columns and df[feature].dtype == bool:
                # Boolean input
                lead_data[feature] = col.checkbox(f"{feature.replace('_', ' ').title()}")
            else:
                # Default numeric input
                lead_data[feature] = col.number_input(f"{feature.replace('_', ' ').title()}")
            
            added_fields.append(feature)
        
        # Add remaining key fields that might not be in top features
        for field in ['state', 'State', 'city', 'City', 'referral_source', 'ReferralSource']:
            if field not in added_fields and field in df.columns:
                # Get unique values
                unique_values = df[field].dropna().unique().tolist()
                unique_values = [v for v in unique_values if v and str(v).strip()]
                
                if unique_values:
                    col = form_col1 if field.lower() in ['state', 'region'] else form_col2
                    selected_value = col.selectbox(
                        f"{field.replace('_', ' ').title()}",
                        options=sorted(unique_values),
                        index=0
                    )
                    lead_data[field] = selected_value
                    added_fields.append(field)
        
        # Submit button
        submit_button = st.form_submit_button("Calculate Lead Score")
    
    # Calculate score if form is submitted
    if submit_button:
        # Ensure we have a model
        if 'enhanced_model' in st.session_state and st.session_state.enhanced_model is not None:
            model = st.session_state.enhanced_model
            scaler = st.session_state.enhanced_scaler
            metrics = st.session_state.enhanced_metrics
            
            # Calculate lead score
            score, category, score_breakdown = score_lead_enhanced(
                lead_data, model, scaler, feature_weights, metrics
            )
            
            # Display results
            st.markdown("### Lead Score Results")
            
            # Create color based on category
            category_colors = {
                "Hot": "#e74c3c",    # Red
                "Warm": "#f39c12",   # Orange
                "Cool": "#3498db",   # Blue
                "Cold": "#2c3e50"    # Dark blue
            }
            
            color = category_colors.get(category, "#7f8c8d")  # Default gray
            
            # Create a more visually appealing score display with gauge and category highlight
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create a gauge-like visualization using matplotlib
                fig, ax = plt.subplots(figsize=(8, 3))
                
                # Draw the gauge background
                ax.barh([0], [1], height=0.3, color="#ecf0f1", alpha=0.5)
                
                # Draw the score indicator
                ax.barh([0], [score], height=0.3, color=color)
                
                # Add a marker at the score position
                ax.scatter(score, 0, s=300, color=color, zorder=3, marker='o', edgecolors='white', linewidth=2)
                
                # Add thresholds
                hot_threshold = metrics.get('thresholds', {}).get('hot', 0.7)
                warm_threshold = metrics.get('thresholds', {}).get('warm', 0.5)
                cool_threshold = metrics.get('thresholds', {}).get('cool', 0.3)
                
                ax.axvline(cool_threshold, color='#3498db', linestyle='--', alpha=0.7)
                ax.axvline(warm_threshold, color='#f39c12', linestyle='--', alpha=0.7)
                ax.axvline(hot_threshold, color='#e74c3c', linestyle='--', alpha=0.7)
                
                # Add text labels for thresholds
                ax.text(cool_threshold/2, -0.15, "Cold", ha='center', fontsize=9, color='#3498db')
                ax.text((warm_threshold + cool_threshold)/2, -0.15, "Cool", ha='center', fontsize=9, color='#3498db')
                ax.text((hot_threshold + warm_threshold)/2, -0.15, "Warm", ha='center', fontsize=9, color='#f39c12')
                ax.text((1 + hot_threshold)/2, -0.15, "Hot", ha='center', fontsize=9, color='#e74c3c')
                
                # Add score text
                ax.text(score, 0, f"{score:.2f}", ha='center', va='center', fontsize=11, 
                       fontweight='bold', color='white')
                
                # Clean up the plot
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.2, 0.2)
                ax.set_axis_off()
                
                st.pyplot(fig)
            
            with col2:
                st.markdown(f"### {category}")
                st.markdown(f"**Score: {score:.2f}**")
                
                # Add explanation based on category
                if category == "Hot":
                    st.markdown("🔥 **High priority lead!** Follow up immediately and focus on closing.")
                elif category == "Warm":
                    st.markdown("🔆 **Good potential.** Follow up soon with a tailored proposal.")
                elif category == "Cool":
                    st.markdown("❄️ **Limited potential.** Follow up when time permits with a standard offer.")
                else:  # Cold
                    st.markdown("🧊 **Low priority.** Consider an automated follow-up or add to nurture campaign.")
            
            # Detailed breakdown
            with st.expander("Detailed Score Analysis", expanded=True):
                # Show the breakdown of how the score was calculated
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Score Components")
                    st.markdown(f"Base Score: **{score_breakdown.get('raw_score', 0):.3f}**")
                    
                    if 'decayed_score' in score_breakdown and score_breakdown['decayed_score'] != score_breakdown['raw_score']:
                        decay_amount = score_breakdown['raw_score'] - score_breakdown['decayed_score']
                        st.markdown(f"Time Decay: **-{decay_amount:.3f}**")
                    
                    if 'region_boost' in score_breakdown and score_breakdown['region_boost'] > 1.0:
                        region_effect = score_breakdown['final_score'] - score_breakdown['decayed_score']
                        st.markdown(f"Region Boost: **+{region_effect:.3f}**")
                    
                    st.markdown(f"Final Score: **{score_breakdown.get('final_score', score):.3f}**")
                
                with col2:
                    st.markdown("### Decay Analysis")
                    
                    days_since = lead_data.get(days_since_field, 0)
                    st.markdown(f"Days Since Inquiry: **{days_since}**")
                    
                    # Get category-specific half-life if available
                    category_half_lives = metrics.get('category_half_lives')
                    if category_half_lives is not None and category_field and category_field in lead_data:
                        lead_category = lead_data[category_field]
                        if lead_category in category_half_lives:
                            half_life = category_half_lives[lead_category]
                            st.markdown(f"Category Half-Life: **{half_life:.1f} days**")
                            
                            # Calculate remaining potential
                            decay_factor = 2 ** (-days_since / half_life)
                            remaining_potential = decay_factor * 100
                            
                            # Create progress bar showing remaining potential
                            st.markdown(f"Remaining Potential: **{remaining_potential:.1f}%**")
                            st.progress(decay_factor)
                
                # Top influencing factors
                st.markdown("### Top Influencing Factors")
                
                feature_impact = score_breakdown.get('feature_impact', {})
                if feature_impact:
                    impact_data = []
                    for feature, details in feature_impact.items():
                        impact_data.append({
                            'Feature': feature,
                            'Value': details['value'],
                            'Weight': details['weight'],
                            'Direction': details['direction'],
                            'Impact': '+' if details['direction'] == '+' else '-'
                        })
                    
                    impact_df = pd.DataFrame(impact_data)
                    st.dataframe(impact_df, use_container_width=True)
                else:
                    st.markdown("No feature impact data available.")
        else:
            st.warning("No enhanced lead scoring model found. Please click 'Generate Enhanced Model' to create one.")