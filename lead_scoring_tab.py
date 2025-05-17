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
from derive_scorecard import score_lead, calculate_category_half_life
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
        
        # Get features (exclude outcome & ID columns)
        exclude_patterns = ['outcome', 'id', 'key', 'index']
        features = [col for col in numeric_df.columns 
                   if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
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
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds_roc = roc_curve(y, y_probs)
        roc_auc = roc_auc_score(y, y_probs)
        
        # Calculate optimal threshold using F1 score
        f1_scores = []
        threshold_range = np.linspace(0.1, 0.9, 81)
        for threshold in threshold_range:
            y_pred = (y_probs >= threshold).astype(int)
            f1 = f1_score(y, y_pred)
            f1_scores.append((threshold, f1))
        
        f1_df = pd.DataFrame(f1_scores, columns=['threshold', 'f1_score'])
        best_f1_idx = f1_df['f1_score'].argmax()
        best_threshold = f1_df.iloc[best_f1_idx]['threshold']
        max_f1_score = f1_df.iloc[best_f1_idx]['f1_score']
        
        # Thresholds for lead categorization
        thresholds = {
            'hot': best_threshold,           # Optimal threshold
            'warm': best_threshold * 0.66,   # 2/3 of optimal
            'cool': best_threshold * 0.33    # 1/3 of optimal
        }
        
        # Separate scores for won and lost leads
        won_scores = y_probs[y == 1]
        lost_scores = y_probs[y == 0]
        
        # Calculate comprehensive model metrics
        metrics = {
            'roc_auc': roc_auc,
            'accuracy': rf_model.score(X_scaled, y),
            'f1': max_f1_score,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds_roc,
            'best_threshold': best_threshold,
            'y_pred_proba': y_probs,
            'y_true': y,
            'won_scores': won_scores,
            'lost_scores': lost_scores,
            'features': features,
            'model': rf_model,
            'scaler': scaler
        }
        
        # Save model components for future predictions
        import streamlit as st
        st.session_state.model_metrics = metrics
        st.session_state.y_probs = y_probs
        st.session_state.y_true = y
        st.session_state.model = rf_model
        st.session_state.scaler = scaler
        st.session_state.model_features = features
        
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
    
    if 'outcome' not in df.columns:
        st.warning("Outcome column not found in the data. Lead scoring requires a column indicating won/lost status.")
        return
    
    # Create a button to generate or regenerate the model
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Generate Lead Scoring Model")
        st.markdown("This will analyze your historical data to identify factors that predict successful deals.")
    
    with col2:
        generate_button = st.button(
            "Generate Model", 
            help="Click to train a machine learning model on your data to predict which leads are most likely to convert"
        )
    
    # Generate scorecard if button is clicked or if not already in session state
    if generate_button or 'weights_df' not in st.session_state:
        try:
            with st.spinner("Generating lead scoring model..."):
                # Generate model
                weights_df, thresholds, metrics = generate_lead_scorecard(df)
                
                # Store in session state
                st.session_state.weights_df = weights_df
                st.session_state.thresholds = thresholds
                st.session_state.metrics = metrics
                
                # Always set model_metrics for other tabs to use
                if metrics is not None:
                    st.session_state.model_metrics = metrics
                    
                    # Ensure the specific y_pred_proba is directly available in session state
                    if 'y_pred_proba' in metrics:
                        st.session_state.y_pred_proba = metrics['y_pred_proba']
                
                st.success("Lead scoring model generated successfully!")
        except Exception as e:
            st.error(f"Error generating lead scoring model: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            return
    else:
        # Use existing model from session state
        if 'weights_df' in st.session_state and st.session_state.weights_df is not None:
            weights_df = st.session_state.weights_df
            thresholds = st.session_state.thresholds
            metrics = st.session_state.metrics
        else:
            st.warning("No lead scoring model found. Please click 'Generate Model' to create one.")
            return
    
    # Make sure we have valid data to display
    if weights_df is None or weights_df.empty:
        st.warning("Unable to generate lead scoring model. Please ensure you have sufficient data with clear outcomes.")
        return
    
    # Display model metrics
    st.markdown("### Model Performance")
    
    # Show metrics in columns with explanations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auc_value = metrics['roc_auc']
        auc_color = "normal"
        if auc_value > 0.8:
            auc_color = "good"
        elif auc_value < 0.6:
            auc_color = "poor"
        st.metric("ROC AUC", f"{auc_value:.3f}", delta=None, delta_color=auc_color)
        st.caption("Measures model's ability to distinguish wins from losses")
    
    with col2:
        accuracy = metrics['accuracy']
        st.metric("Accuracy", f"{accuracy:.1%}")
        st.caption("Percentage of correct predictions")
    
    with col3:
        f1 = metrics['f1']
        st.metric("F1 Score", f"{f1:.3f}")
        st.caption("Balance of precision and recall")
    
    # Display feature weights
    st.markdown("### Feature Importance")
    st.markdown("These features have the strongest influence on predicting won deals.")
    
    # Sort by absolute weight
    weights_df = weights_df.sort_values(by='weight', key=abs, ascending=False)
    
    # Create a horizontal bar chart of weights
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create color map (blue for positive, red for negative)
    colors = ['#1E88E5' if w >= 0 else '#f44336' for w in weights_df['weight']]
    
    # Plot data
    ax.barh(weights_df['feature'], weights_df['weight'], color=colors)
    
    # Customize plot
    ax.set_xlabel('Weight (positive = increases win probability)')
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
        st.markdown("This shows how your won and lost leads score in the model, and where the thresholds are set.")
        
        # Get distributions from session state
        y_probs = st.session_state.y_probs
        y_true = st.session_state.y_true
        
        # Create metrics dict for plot_score_distributions
        plot_metrics = {
            'won_scores': y_probs[y_true == 1],
            'lost_scores': y_probs[y_true == 0],
            'best_threshold': thresholds['hot'],
            'f1_threshold': thresholds['hot'],
            'y_true': y_true,
            'y_pred_proba': y_probs
        }
        
        try:
            fig = plot_score_distributions(plot_metrics, None)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting score distributions: {str(e)}")
    
    # Lead scoring calculator
    st.markdown("### Lead Scoring Calculator")
    st.markdown("Enter values for a new lead to calculate its score and conversion probability.")
    
    # Create a form for lead scoring
    with st.form("lead_scoring_form"):
        # Get input fields for the top features (limit to most important)
        input_values = {}
        
        # Create two columns for the form
        form_col1, form_col2 = st.columns(2)
        
        # Show form fields for the top features
        top_features = weights_df.head(8)['feature'].tolist()
        for i, feature in enumerate(top_features):
            # Alternate between columns
            col = form_col1 if i % 2 == 0 else form_col2
            
            # Set input field based on feature name
            if 'days' in feature.lower() or 'guests' in feature.lower() or 'bartenders' in feature.lower() or 'month' in feature.lower():
                input_values[feature] = col.number_input(f"{feature}", min_value=0, step=1)
            elif 'price' in feature.lower() or 'value' in feature.lower() or 'revenue' in feature.lower():
                input_values[feature] = col.number_input(f"{feature}", min_value=0.0, step=100.0)
            elif 'corporate' in feature.lower() or 'match' in feature.lower() or 'is_' in feature.lower():
                # Boolean features
                input_values[feature] = col.selectbox(f"{feature}", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            else:
                input_values[feature] = col.number_input(f"{feature}", min_value=0.0, step=0.1)
        
        # Submit button
        submit_button = st.form_submit_button("Calculate Score")
    
    # Process form submission
    if submit_button:
        try:
            # Calculate score
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
                probability_percentage = probability * 100
                st.metric("Win Probability", f"{probability_percentage:.1f}%")
            
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
                    
            # Recommendations based on score
            st.markdown("### Recommended Actions")
            if category == "Hot 🔥":
                st.success("This lead has a high probability of converting. Prioritize follow-up and consider expedited processing.")
            elif category == "Warm 🟠":
                st.info("This lead shows promise. Address any negative factors and provide detailed information to help them decide.")
            else:
                st.warning("This lead may need additional nurturing. Consider addressing specific concerns or offering incentives.")
                
        except Exception as e:
            st.error(f"Error calculating score: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

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
        
        - **Hot 🔥**: High probability of winning (top tier)
        - **Warm 🟠**: Moderate probability (middle tier) 
        - **Cool 🧊**: Lower probability (bottom tier)
        
        These thresholds are calibrated based on your historical conversion data.
        """)