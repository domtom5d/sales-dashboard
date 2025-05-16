"""
lead_scoring_tab.py - Lead Scoring Tab Module

This module provides the implementation for the Lead Scoring tab
in the Sales Conversion Analytics Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from derive_scorecard import score_lead
from evaluate import plot_score_distributions
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score

def generate_lead_scorecard(df):
    """
    Generate a lead scorecard based on the provided DataFrame
    
    Args:
        df (DataFrame): Processed dataframe with outcome column
        
    Returns:
        tuple: (weights_df, thresholds, metrics)
    """
    try:
        # Check if df is a DataFrame and has the outcome column
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if 'outcome' not in df.columns:
            raise ValueError("DataFrame must contain an 'outcome' column")
        
        # Verify we have enough data
        win_count = df['outcome'].sum()
        loss_count = len(df) - win_count
        
        if win_count < 5 or loss_count < 5:
            raise ValueError(f"Insufficient data for modeling: {win_count} wins, {loss_count} losses")
        
        # Select numeric features only
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        
        # Make sure outcome is included
        if 'outcome' not in numeric_df.columns:
            numeric_df['outcome'] = df['outcome']
        
        # Remove any columns with all NaN values
        numeric_df = numeric_df.dropna(axis=1, how='all')
        
        # Get features (exclude outcome)
        features = [col for col in numeric_df.columns if col != 'outcome']
        
        if len(features) < 2:
            raise ValueError("Not enough numeric features for modeling")
        
        # Prepare X and y
        X = numeric_df[features].fillna(0)
        y = numeric_df['outcome']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train a Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
        rf_model.fit(X_scaled, y)
        
        # Train Logistic Regression for interpretability
        lr_model = LogisticRegression(max_iter=1000, C=0.1, solver='liblinear')
        lr_model.fit(X_scaled, y)
        
        # Get feature importance from both models
        rf_importance = pd.Series(rf_model.feature_importances_, index=features)
        coefs = pd.Series(lr_model.coef_[0], index=features)
        
        # Create hybrid importance that uses magnitude from RF but direction from LR
        importance = rf_importance.copy()
        
        # Apply direction from logistic regression
        for feat in features:
            if coefs[feat] < 0:
                importance[feat] = -importance[feat]
        
        # Sort by absolute importance
        imp = importance.abs().sort_values(ascending=False)
        
        # Scale to a 0-10 point system
        weights = (imp / imp.iloc[0] * 10).round(2) if len(imp) > 0 else pd.Series()
        
        # Create feature direction mapping
        feature_direction = {feat: "+" if coefs[feat] > 0 else "-" for feat in features}
        
        # Create DataFrame for display
        result_df = pd.DataFrame({
            'feature': weights.index,
            'weight': [weights[feat] if feat in weights else 0 for feat in features],
            'direction': [feature_direction[feat] for feat in features]
        })
        
        # Sort by absolute weight
        result_df = result_df.sort_values(by='weight', key=abs, ascending=False)
        
        # Calculate model metrics
        y_probs = rf_model.predict_proba(X_scaled)[:, 1]
        
        # Store probabilities and true values in session state for plotting
        import streamlit as st
        st.session_state.y_probs = y_probs
        st.session_state.y_true = y
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds_roc = roc_curve(y, y_probs)
        roc_auc = roc_auc_score(y, y_probs)
        
        # Simple thresholds based on percentiles
        thresholds = {
            'hot': 0.75,   # Top 25%
            'warm': 0.5,   # Top 50%
            'cool': 0.25   # Top 75%
        }
        
        # Model metrics for display
        metrics = {
            'roc_auc': roc_auc,
            'accuracy': rf_model.score(X_scaled, y),
            'f1': f1_score(y, (y_probs >= 0.5).astype(int))
        }
        
        return result_df, thresholds, metrics
        
    except Exception as e:
        import traceback
        print(f"Error generating scorecard: {str(e)}")
        print(traceback.format_exc())
        return None, None, None
from evaluate import plot_score_distributions

def render_lead_scoring_tab(df):
    """
    Render the complete Lead Scoring tab
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    """
    st.subheader("Lead Scoring Model")
    st.markdown("Use this model to evaluate new leads and focus on the most promising opportunities.")
    
    # Add data shape debugging
    st.write(f"DEBUG: DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
    
    if 'outcome' in df.columns:
        # Generate scorecard if not already in session state
        if 'weights_df' not in st.session_state or st.session_state.weights_df is None:
            try:
                with st.spinner("Generating lead scoring model..."):
                    # Generate model
                    weights_df, thresholds, metrics = generate_lead_scorecard(df)
                    
                    # Store in session state
                    st.session_state.weights_df = weights_df
                    st.session_state.thresholds = thresholds
                    st.session_state.metrics = metrics
                    
                    st.success("Lead scoring model generated successfully!")
            except Exception as e:
                st.error(f"Error generating lead scoring model: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
                weights_df = None
                thresholds = None
                metrics = None
        else:
            weights_df = st.session_state.weights_df
            thresholds = st.session_state.thresholds
            metrics = st.session_state.metrics
        
        if weights_df is not None and not weights_df.empty:
            # Display model metrics
            st.markdown("### Model Performance")
            if 'metrics' in st.session_state and st.session_state.metrics is not None:
                metrics = st.session_state.metrics
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
                
                with col2:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                
                with col3:
                    st.metric("F1 Score", f"{metrics['f1']:.3f}")
            
            # Display feature weights
            st.markdown("### Feature Weights")
            st.markdown("These weights show how important each feature is for predicting won deals.")
            
            # Sort by absolute weight
            weights_df = weights_df.sort_values(by='weight', key=abs, ascending=False)
            
            # Create a horizontal bar chart of weights
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create color map (blue for positive, red for negative)
            colors = ['#1E88E5' if w >= 0 else '#f44336' for w in weights_df['weight']]
            
            # Plot data
            ax.barh(weights_df['feature'], weights_df['weight'], color=colors)
            
            # Customize plot
            ax.set_xlabel('Weight')
            ax.set_ylabel('Feature')
            ax.set_title('Feature Importance in Lead Scoring Model')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add labels with the actual weight values
            for i, v in enumerate(weights_df['weight']):
                ax.text(v + (0.01 if v >= 0 else -0.01), 
                        i, 
                        f"{v:.2f}", 
                        va='center', 
                        ha='left' if v >= 0 else 'right',
                        fontweight='bold')
            
            st.pyplot(fig)
            
            # Score distribution plot
            if 'y_probs' in st.session_state and 'y_true' in st.session_state:
                st.markdown("### Score Distribution")
                fig = plot_score_distributions(
                    st.session_state.y_probs, 
                    st.session_state.y_true,
                    thresholds
                )
                st.pyplot(fig)
            
            # Lead scoring calculator
            st.markdown("### Lead Scoring Calculator")
            st.markdown("Enter values for a new lead to calculate its score and conversion probability.")
            
            # Create a form for lead scoring
            with st.form("lead_scoring_form"):
                # Get input fields for the top features
                input_values = {}
                
                # Create two columns for the form
                form_col1, form_col2 = st.columns(2)
                
                # Show form fields for the top features
                top_features = weights_df.head(10)['feature'].tolist()
                for i, feature in enumerate(top_features):
                    # Alternate between columns
                    col = form_col1 if i % 2 == 0 else form_col2
                    
                    # Create input field based on feature name
                    if 'days' in feature.lower() or 'guests' in feature.lower() or 'bartenders' in feature.lower():
                        input_values[feature] = col.number_input(f"{feature}", min_value=0, step=1)
                    elif 'price' in feature.lower() or 'value' in feature.lower() or 'revenue' in feature.lower():
                        input_values[feature] = col.number_input(f"{feature}", min_value=0.0, step=100.0)
                    else:
                        input_values[feature] = col.number_input(f"{feature}", min_value=0.0, step=0.1)
                
                # Submit button
                submit_button = st.form_submit_button("Calculate Score")
            
            # Process form submission
            if submit_button:
                try:
                    score, probability = score_lead(input_values, weights_df)
                    
                    # Determine the lead category based on thresholds
                    category = "Cool 🧊"
                    if probability >= thresholds['hot']:
                        category = "Hot 🔥"
                    elif probability >= thresholds['warm']:
                        category = "Warm 🟠"
                    
                    # Display results
                    st.markdown("### Lead Score Results")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric("Raw Score", f"{score:.2f}")
                    
                    with result_col2:
                        st.metric("Win Probability", f"{probability:.1%}")
                    
                    with result_col3:
                        st.metric("Lead Category", category)
                    
                    # Provide explanation
                    st.markdown("### Score Explanation")
                    
                    # Calculate feature contributions
                    contributions = []
                    for feature, weight in zip(weights_df['feature'], weights_df['weight']):
                        if feature in input_values:
                            contribution = weight * input_values[feature]
                            contributions.append({
                                'feature': feature,
                                'value': input_values[feature],
                                'weight': weight,
                                'contribution': contribution
                            })
                    
                    # Convert to DataFrame and sort by absolute contribution
                    contrib_df = pd.DataFrame(contributions)
                    contrib_df = contrib_df.sort_values(by='contribution', key=abs, ascending=False)
                    
                    # Display the top positive and negative factors
                    positive_factors = contrib_df[contrib_df['contribution'] > 0].head(3)
                    negative_factors = contrib_df[contrib_df['contribution'] < 0].head(3)
                    
                    if not positive_factors.empty:
                        st.markdown("**Top Positive Factors:**")
                        for i, row in positive_factors.iterrows():
                            st.markdown(f"- **{row['feature']}**: Value of {row['value']} contributes +{row['contribution']:.2f} to the score")
                    
                    if not negative_factors.empty:
                        st.markdown("**Top Negative Factors:**")
                        for i, row in negative_factors.iterrows():
                            st.markdown(f"- **{row['feature']}**: Value of {row['value']} contributes {row['contribution']:.2f} to the score")
                except Exception as e:
                    st.error(f"Error calculating score: {str(e)}")
        else:
            st.warning("Unable to generate lead scoring model. Please ensure you have sufficient data with clear outcomes.")
    else:
        st.warning("Outcome column not found in the data. Lead scoring requires a column indicating won/lost status.")

    # Add guidance on how to use the scoring model
    with st.expander("How to use the Lead Scoring model"):
        st.markdown("""
        ### Understanding Lead Scores
        
        The lead scoring model analyzes your historical data to identify patterns that predict successful deals. Here's how to use it:
        
        1. **Enter Values**: Input the values for the new lead in the calculator above
        2. **Check Score**: Review the calculated score and probability
        3. **Focus on Hot Leads**: Prioritize "Hot" leads that have high win probabilities
        4. **Address Negative Factors**: For promising leads with negative factors, consider how to address those specific issues
        
        ### Score Categories
        
        - **Hot 🔥**: High probability of winning (80%+ chance)
        - **Warm 🟠**: Moderate probability (50-80% chance)
        - **Cool 🧊**: Lower probability (under 50% chance)
        
        These thresholds are calibrated based on your historical conversion data.
        """)