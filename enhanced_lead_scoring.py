"""
enhanced_lead_scoring.py - Advanced lead scoring with category-specific decay and regional analysis

This module provides advanced lead scoring functionality that incorporates:
- Guest count influence
- Days since inquiry with category-specific decay
- Staff-to-guest ratio optimization
- Referral source tier influence
- Regional conversion rate differences
- Color-coded lead categories
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

def calculate_category_specific_half_life(df, category_column='booking_type'):
    """
    Calculate category-specific half-life values for improved lead decay modeling.
    
    Args:
        df (DataFrame): Processed dataframe with outcomes and dates
        category_column (str): The column containing category information (booking_type or event_type)
    
    Returns:
        pd.Series: Half-life values indexed by category
    """
    # Ensure we have the necessary columns
    required_cols = ['outcome', category_column, 'days_since_inquiry', 'inquiry_date']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Handle missing column situation
        default_half_life = pd.Series({'default': 30.0})  # Default 30-day half-life
        if 'category_column' in missing_cols:
            return default_half_life
        
    # Create a clean copy with only won leads and valid categories
    won_leads = df[df['outcome'] == 'Won'].copy()
    won_leads = won_leads.dropna(subset=[category_column, 'days_since_inquiry'])
    
    if len(won_leads) < 5:
        return pd.Series({'default': 30.0})  # Not enough data
    
    # Group by category and calculate median days to conversion
    category_cycles = won_leads.groupby(category_column)['days_since_inquiry'].median()
    
    # Convert to half-life values (apply formula or use directly)
    # Here we use median days * 1.2 as an empirical approximation
    half_lives = category_cycles * 1.2
    
    # Set minimum and maximum reasonable half-lives
    half_lives = half_lives.clip(lower=7, upper=90)
    
    # Add a default for any category not covered
    half_lives['default'] = half_lives.median()
    
    return half_lives

def apply_decay(score, days_since, half_life):
    """
    Apply exponential decay to a score based on time elapsed and category-specific half-life
    
    Args:
        score (float): Original score value to decay
        days_since (float): Days elapsed since the inquiry
        half_life (float): Half-life in days (category-specific)
        
    Returns:
        float: Decayed score value
    """
    # Ensure inputs are valid
    if days_since < 0:
        days_since = 0
    if half_life <= 0:
        half_life = 30.0  # Default to 30 days if invalid half-life
        
    # Apply exponential decay: score * 2^(-days/half_life)
    return score * (2 ** (-days_since / half_life))

def identify_high_performing_regions(df):
    """
    Identify regions (states, cities) with significantly higher conversion rates
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    
    Returns:
        tuple: (high_performing_states, high_performing_cities)
    """
    high_performing_states = {}
    high_performing_cities = {}
    
    # Check if we have state/region data
    state_col = None
    for col in ['state', 'State', 'region', 'Region']:
        if col in df.columns:
            state_col = col
            break
    
    city_col = None
    for col in ['city', 'City', 'location', 'Location']:
        if col in df.columns:
            city_col = col
            break
    
    # Get baseline conversion rate
    overall_conversion = df['outcome'].mean() if 'outcome' in df.columns else 0.2
    
    # Analyze states if available
    if state_col is not None:
        # Group by state and calculate conversion rate
        state_conversions = df.groupby(state_col)['outcome'].agg(['mean', 'count']).reset_index()
        state_conversions.columns = [state_col, 'conversion_rate', 'count']
        
        # Filter states with enough data (at least 5 leads)
        state_conversions = state_conversions[state_conversions['count'] >= 5]
        
        # Identify high performing states (>25% better than average)
        high_threshold = overall_conversion * 1.25
        high_states = state_conversions[state_conversions['conversion_rate'] > high_threshold]
        
        # Convert to dictionary for easy lookup
        high_performing_states = dict(zip(high_states[state_col], high_states['conversion_rate']))
    
    # Analyze cities if available
    if city_col is not None:
        # Group by city and calculate conversion rate
        city_conversions = df.groupby(city_col)['outcome'].agg(['mean', 'count']).reset_index()
        city_conversions.columns = [city_col, 'conversion_rate', 'count']
        
        # Filter cities with enough data (at least 3 leads)
        city_conversions = city_conversions[city_conversions['count'] >= 3]
        
        # Identify high performing cities (>30% better than average)
        high_threshold = overall_conversion * 1.3
        high_cities = city_conversions[city_conversions['conversion_rate'] > high_threshold]
        
        # Convert to dictionary for easy lookup
        high_performing_cities = dict(zip(high_cities[city_col], high_cities['conversion_rate']))
    
    return high_performing_states, high_performing_cities

def classify_referral_tiers(df):
    """
    Classify referral sources into tiers based on conversion performance
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    
    Returns:
        dict: Mapping of referral sources to tiers (1-3)
    """
    referral_tiers = {}
    
    # Check if we have referral source data
    ref_col = None
    for col in ['referral_source', 'ReferralSource', 'referral', 'Referral']:
        if col in df.columns:
            ref_col = col
            break
    
    if ref_col is None:
        return referral_tiers
    
    # Group by referral source and calculate conversion rate
    ref_conversions = df.groupby(ref_col)['outcome'].agg(['mean', 'count']).reset_index()
    ref_conversions.columns = [ref_col, 'conversion_rate', 'count']
    
    # Filter sources with enough data (at least 3 leads)
    ref_conversions = ref_conversions[ref_conversions['count'] >= 3]
    
    # Create tiers based on performance
    q75 = ref_conversions['conversion_rate'].quantile(0.75)
    q25 = ref_conversions['conversion_rate'].quantile(0.25)
    
    # Assign tiers
    for _, row in ref_conversions.iterrows():
        source = row[ref_col]
        rate = row['conversion_rate']
        
        if rate >= q75:
            referral_tiers[source] = 3  # Top tier
        elif rate <= q25:
            referral_tiers[source] = 1  # Bottom tier
        else:
            referral_tiers[source] = 2  # Middle tier
    
    return referral_tiers

def train_enhanced_lead_scoring_model(df):
    """
    Train an enhanced lead scoring model with focus on key variables
    
    Args:
        df (DataFrame): Processed dataframe with outcome column
        
    Returns:
        tuple: (model, scaler, feature_weights, thresholds, metrics)
    """
    # Check if we have the necessary outcome column
    if 'outcome' not in df.columns:
        return None, None, None, None, None
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure outcome is binary
    if df_copy['outcome'].dtype == 'object':
        df_copy['outcome'] = df_copy['outcome'].map({'Won': 1, 'Lost': 0})
    
    # Key features to prioritize
    priority_features = [
        # Guest count variables
        'number_of_guests', 'guest_count', 'NumberOfGuests', 'GuestCount',
        
        # Time-based variables
        'days_since_inquiry', 'days_until_event', 'DaysSinceInquiry', 'DaysUntilEvent',
        
        # Staff ratio variables
        'bartenders_needed', 'BartendersNeeded', 'staff_ratio',
        
        # Price variables
        'price_per_guest', 'total_price', 'deal_value',
        
        # Season variables
        'event_month', 'event_season', 'EventMonth'
    ]
    
    # Find which priority features exist in our data
    available_features = [f for f in priority_features if f in df_copy.columns]
    
    # If we don't have enough priority features, include other numeric features
    min_features = 3
    if len(available_features) < min_features:
        # Add more numeric features if needed
        numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
        # Exclude outcome and ID columns
        exclude_patterns = ['outcome', 'id', 'key', 'index']
        numeric_cols = [col for col in numeric_cols 
                      if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        # Add these to available features
        available_features.extend([col for col in numeric_cols if col not in available_features])
    
    # Create engineered features
    
    # 1. Staff-to-guest ratio if both columns exist
    staff_col = None
    guest_col = None
    
    for col in ['bartenders_needed', 'BartendersNeeded', 'staff_count', 'StaffCount']:
        if col in df_copy.columns:
            staff_col = col
            break
    
    for col in ['number_of_guests', 'guest_count', 'NumberOfGuests', 'GuestCount']:
        if col in df_copy.columns:
            guest_col = col
            break
    
    if staff_col and guest_col:
        # Replace zeros with NaN to avoid division by zero
        df_copy[guest_col] = df_copy[guest_col].replace(0, np.nan)
        # Create the ratio feature
        df_copy['staff_to_guest_ratio'] = df_copy[staff_col] / df_copy[guest_col]
        # Fill NaN with median
        median_ratio = df_copy['staff_to_guest_ratio'].median()
        df_copy['staff_to_guest_ratio'] = df_copy['staff_to_guest_ratio'].fillna(median_ratio)
        # Add to available features
        available_features.append('staff_to_guest_ratio')
    
    # 2. Add referral tier feature if referral_source exists
    referral_col = None
    for col in ['referral_source', 'ReferralSource', 'referral', 'Referral']:
        if col in df_copy.columns:
            referral_col = col
            break
    
    if referral_col:
        # Get the referral tiers
        referral_tiers = classify_referral_tiers(df)
        # Map to the dataframe
        df_copy['referral_tier'] = df_copy[referral_col].map(referral_tiers).fillna(2)  # Default to middle tier
        # Add to available features
        available_features.append('referral_tier')
    
    # 3. Add high-performing region feature if state or city exists
    high_performing_states, high_performing_cities = identify_high_performing_regions(df)
    
    state_col = None
    for col in ['state', 'State', 'region', 'Region']:
        if col in df_copy.columns:
            state_col = col
            break
    
    city_col = None
    for col in ['city', 'City', 'location', 'Location']:
        if col in df_copy.columns:
            city_col = col
            break
    
    if state_col or city_col:
        # Create high region feature
        df_copy['high_performing_region'] = 0
        
        # Update based on state
        if state_col and high_performing_states:
            for state, _ in high_performing_states.items():
                df_copy.loc[df_copy[state_col] == state, 'high_performing_region'] = 1
        
        # Update based on city (overrides state if present)
        if city_col and high_performing_cities:
            for city, _ in high_performing_cities.items():
                df_copy.loc[df_copy[city_col] == city, 'high_performing_region'] = 1
        
        # Add to available features
        available_features.append('high_performing_region')
    
    # Prepare data for modeling
    if not available_features:
        return None, None, None, None, None
    
    # Select final feature set
    X = df_copy[available_features].copy()
    y = df_copy['outcome']
    
    # Handle categorical columns and fill missing values properly
    for col in X.columns:
        # Special handling for month names if present
        if col in ['event_month', 'EventMonth'] and X[col].dtype == 'object':
            # Convert month names to numeric values
            month_mapping = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            # Apply the mapping, preserving NaN values
            X[col] = X[col].map(lambda x: month_mapping.get(x, x) if pd.notna(x) else x)
            
            # For any remaining non-numeric values, use one-hot encoding
            if X[col].dtype == 'object':
                # Fill missing values with median or most common value
                if X[col].mode().size > 0:
                    mode_value = X[col].mode()[0]
                    X[col] = X[col].fillna(mode_value)
                    # Create dummy variables if needed (less than 10 unique values)
                    if X[col].nunique() <= 10:
                        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                        X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                else:
                    # If no mode, use 6 (middle of year) as default
                    X[col] = X[col].fillna(6)
            else:
                # Now it's numeric, fill missing with median
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())
        
        # Special handling for seasons if present
        elif col in ['event_season', 'EventSeason'] and X[col].dtype == 'object':
            # Convert season names to numeric
            season_mapping = {
                'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4, 'Autumn': 4
            }
            # Apply the mapping, preserving NaN values
            X[col] = X[col].map(lambda x: season_mapping.get(x, x) if pd.notna(x) else x)
            
            # Handle remaining non-numeric values
            if X[col].dtype == 'object':
                # Fill missing and use one-hot encoding if needed
                most_frequent = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
                X[col] = X[col].fillna(most_frequent)
                if X[col].nunique() <= 10:
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
            else:
                # Now it's numeric, fill missing with median
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())
                
        # Standard handling for other categorical columns
        elif X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
            # For categorical data, fill with most frequent value
            most_frequent = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
            X[col] = X[col].fillna(most_frequent)
            
            # Convert categorical features to numeric using one-hot encoding
            if X[col].nunique() <= 10:  # Only encode if reasonable number of categories
                # Create dummy variables and drop the original column
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
        else:
            # For numeric columns, fill with median
            X[col] = X[col].fillna(X[col].median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train two models for better insights - Random Forest for performance, Linear for interpretability
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
    rf_model.fit(X_scaled, y)
    
    # Get RF model performance
    y_probs = rf_model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, thresholds_roc = roc_curve(y, y_probs)
    roc_auc = roc_auc_score(y, y_probs)
    
    # Train logistic regression for feature interpretation
    lr_model = LogisticRegression(max_iter=1000, C=0.1, solver='liblinear')
    lr_model.fit(X_scaled, y)
    
    # Get feature importance with direction
    rf_importance = pd.Series(rf_model.feature_importances_, index=available_features)
    coefs = pd.Series(lr_model.coef_[0], index=available_features)
    
    # Create hybrid importance (RF magnitude with LR direction)
    importance = rf_importance.copy()
    for feat in available_features:
        if coefs[feat] < 0:
            importance[feat] = -importance[feat]
    
    # Scale to a 0-10 point system
    weights = (importance.abs() / importance.abs().max() * 10).round(2)
    
    # Create dataframe for display
    feature_weights = pd.DataFrame({
        'feature': weights.index,
        'weight': [weights[feat] for feat in available_features],
        'direction': ['+' if coefs[feat] > 0 else '-' for feat in available_features],
        'coefficient': coefs.values
    })
    
    # Sort by absolute weight
    feature_weights = feature_weights.sort_values(by='weight', key=abs, ascending=False)
    
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
    
    # Define thresholds for lead categorization
    thresholds = {
        'hot': best_threshold,           # Optimal threshold
        'warm': best_threshold * 0.66,   # 2/3 of optimal
        'cool': best_threshold * 0.33    # 1/3 of optimal
    }
    
    # Calculate category-specific half-lives for decay function
    category_half_lives = None
    category_field = None
    for col in ['event_type', 'booking_type', 'clean_booking_type']:
        if col in df.columns and not df[col].isna().all():
            try:
                category_half_lives = calculate_category_specific_half_life(df, col)
                category_field = col
                break
            except Exception:
                pass
    
    # Collect metrics
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
        'won_scores': y_probs[y == 1],
        'lost_scores': y_probs[y == 0],
        'features': available_features,
        'model': rf_model,
        'scaler': scaler,
        'category_half_lives': category_half_lives,
        'category_field': category_field,
        'referral_tiers': referral_tiers if referral_col else None,
        'high_performing_regions': {
            'states': high_performing_states,
            'cities': high_performing_cities
        }
    }
    
    return rf_model, scaler, feature_weights, thresholds, metrics

def score_lead_enhanced(lead_data, model, scaler, feature_weights, metrics):
    """
    Score a lead using the enhanced lead scoring model
    
    Args:
        lead_data (dict): Dictionary of lead data
        model: Trained machine learning model
        scaler: Feature scaler
        feature_weights: Feature weights dataframe
        metrics: Model metrics dictionary
    
    Returns:
        tuple: (score, category, score_breakdown)
    """
    # Default values
    if model is None or scaler is None:
        return 0.0, "Unknown", {}
    
    # Get the features used by the model
    model_features = metrics['features']
    
    # Initialize lead features
    lead_features = {}
    
    # Extract values for model features
    for feature in model_features:
        if feature in lead_data:
            lead_features[feature] = lead_data[feature]
        else:
            # Default to median for missing features
            lead_features[feature] = 0.0
    
    # Prepare feature vector
    X = pd.DataFrame([lead_features])
    
    # Ensure all model features are present
    for feature in model_features:
        if feature not in X.columns:
            X[feature] = 0.0
    
    # Reorder columns to match the model's expected order
    X = X[model_features]
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Get the raw score from the model
    raw_score = model.predict_proba(X_scaled)[0, 1]
    
    # Apply category-specific decay if applicable
    category_half_lives = metrics.get('category_half_lives')
    category_field = metrics.get('category_field')
    
    # Check if we have days_since_inquiry and category_half_lives
    days_col = None
    for col in ['days_since_inquiry', 'DaysSinceInquiry', 'days_since']:
        if col in lead_data:
            days_col = col
            break
    
    # Apply decay if we have the necessary data
    if days_col and category_half_lives is not None:
        days_since = float(lead_data[days_col])
        
        # Get the lead's category
        category = None
        if category_field and category_field in lead_data:
            category = str(lead_data[category_field])
        
        # Determine the appropriate half-life to use
        half_life_days = 30.0  # Default
        if category and category in category_half_lives:
            half_life_days = category_half_lives[category]
        elif 'default' in category_half_lives:
            half_life_days = category_half_lives['default']
        
        # Apply decay
        decayed_score = apply_decay(raw_score, days_since, half_life_days)
    else:
        decayed_score = raw_score
    
    # Apply high-performing region boost if applicable
    high_performing_regions = metrics.get('high_performing_regions', {})
    high_states = high_performing_regions.get('states', {})
    high_cities = high_performing_regions.get('cities', {})
    
    region_boost = 1.0  # Default (no boost)
    
    # Check if lead is from a high-converting state
    for state_col in ['state', 'State', 'region', 'Region']:
        if state_col in lead_data and lead_data[state_col] in high_states:
            # Boost by 10%
            region_boost = 1.1
            break
    
    # Check if lead is from a high-converting city (overrides state if present)
    for city_col in ['city', 'City', 'location', 'Location']:
        if city_col in lead_data and lead_data[city_col] in high_cities:
            # Boost by 15%
            region_boost = 1.15
            break
    
    # Apply region boost
    final_score = decayed_score * region_boost
    
    # Cap at 1.0
    final_score = min(final_score, 1.0)
    
    # Determine lead category
    thresholds = metrics.get('thresholds', {'hot': 0.7, 'warm': 0.5, 'cool': 0.3})
    
    if final_score >= thresholds['hot']:
        category = "Hot"
    elif final_score >= thresholds['warm']:
        category = "Warm"
    elif final_score >= thresholds['cool']:
        category = "Cool"
    else:
        category = "Cold"
    
    # Create score breakdown
    score_breakdown = {
        'raw_score': raw_score,
        'decayed_score': decayed_score if 'decayed_score' in locals() else raw_score,
        'region_boost': region_boost,
        'final_score': final_score,
        'thresholds': thresholds,
        'category': category
    }
    
    # Add feature contributions for top influencing factors
    if len(feature_weights) > 0:
        # Get top features and their impact
        top_features = feature_weights.head(5)['feature'].tolist()
        feature_impact = {}
        
        for feature in top_features:
            if feature in lead_data:
                value = lead_data[feature]
                weight = feature_weights.loc[feature_weights['feature'] == feature, 'weight'].iloc[0]
                direction = feature_weights.loc[feature_weights['feature'] == feature, 'direction'].iloc[0]
                
                feature_impact[feature] = {
                    'value': value,
                    'weight': weight,
                    'direction': direction
                }
        
        score_breakdown['feature_impact'] = feature_impact
    
    return final_score, category, score_breakdown

def plot_enhanced_lead_score_visualization(metrics, title="Lead Score Distribution"):
    """
    Create an enhanced visualization of lead scores with clear category boundaries
    
    Args:
        metrics (dict): Model metrics dictionary
        title (str): Plot title
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Get score distributions from metrics
    won_scores = metrics.get('won_scores', [])
    lost_scores = metrics.get('lost_scores', [])
    
    if len(won_scores) == 0 and len(lost_scores) == 0:
        # Create a dummy figure if no scores
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Not enough data to plot score distribution",
                horizontalalignment='center', verticalalignment='center',
                fontsize=12)
        return fig
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get thresholds - ensure we have a proper dictionary with required keys
    default_thresholds = {'hot': 0.7, 'warm': 0.5, 'cool': 0.3}
    thresholds = metrics.get('thresholds', default_thresholds)
    
    # Check if thresholds is not a dictionary or missing required keys
    if not isinstance(thresholds, dict):
        thresholds = default_thresholds
    
    # Ensure all required threshold keys exist
    for key in ['hot', 'warm', 'cool']:
        if key not in thresholds:
            thresholds[key] = default_thresholds[key]
    
    # Plot distributions
    bins = np.linspace(0, 1, 20)
    
    if len(won_scores) > 0:
        sns.histplot(won_scores, bins=bins, alpha=0.7, label='Won Leads', 
                    ax=ax, color='#2ecc71', stat='density', edgecolor='white')
    
    if len(lost_scores) > 0:
        sns.histplot(lost_scores, bins=bins, alpha=0.7, label='Lost Leads', 
                    ax=ax, color='#e74c3c', stat='density', edgecolor='white')
    
    # Add colored background regions for categories
    ax.axvspan(0, thresholds['cool'], alpha=0.1, color='blue', label='Cold')
    ax.axvspan(thresholds['cool'], thresholds['warm'], alpha=0.1, color='cyan', label='Cool')
    ax.axvspan(thresholds['warm'], thresholds['hot'], alpha=0.1, color='orange', label='Warm')
    ax.axvspan(thresholds['hot'], 1, alpha=0.1, color='red', label='Hot')
    
    # Add threshold lines
    ax.axvline(thresholds['cool'], linestyle='--', color='#3498db', linewidth=1.5)
    ax.axvline(thresholds['warm'], linestyle='--', color='#f39c12', linewidth=1.5)
    ax.axvline(thresholds['hot'], linestyle='--', color='#e74c3c', linewidth=1.5)
    
    # Label the threshold lines
    ax.text(thresholds['cool'] - 0.02, plt.ylim()[1] * 0.9, 'Cool', 
           rotation=90, ha='right', va='top', color='#3498db', fontweight='bold')
    ax.text(thresholds['warm'] - 0.02, plt.ylim()[1] * 0.9, 'Warm', 
           rotation=90, ha='right', va='top', color='#f39c12', fontweight='bold')
    ax.text(thresholds['hot'] - 0.02, plt.ylim()[1] * 0.9, 'Hot', 
           rotation=90, ha='right', va='top', color='#e74c3c', fontweight='bold')
    
    # Set plot styling
    ax.set_xlabel('Lead Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add ROC AUC info if available
    if 'roc_auc' in metrics:
        roc_auc = metrics['roc_auc']
        
        # Add quality assessment based on ROC AUC
        quality_text = ""
        if roc_auc >= 0.8:
            quality_text = "Excellent"
        elif roc_auc >= 0.7:
            quality_text = "Good"
        elif roc_auc >= 0.6:
            quality_text = "Fair"
        else:
            quality_text = "Poor"
        
        ax.text(0.02, 0.95, f"Model Quality: {quality_text} (AUC = {roc_auc:.2f})",
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Annotate the score regions with explanations
    y_level = 0.1
    ax.annotate('Cold: Minimal chance', xy=(thresholds['cool']/2, y_level), xycoords='data',
               ha='center', color='#3498db', fontsize=9, fontweight='bold')
    
    ax.annotate('Cool: Low priority', xy=((thresholds['warm'] + thresholds['cool'])/2, y_level), 
               xycoords='data', ha='center', color='#3498db', fontsize=9, fontweight='bold')
    
    ax.annotate('Warm: Follow up soon', xy=((thresholds['hot'] + thresholds['warm'])/2, y_level), 
               xycoords='data', ha='center', color='#f39c12', fontsize=9, fontweight='bold')
    
    ax.annotate('Hot: High priority', xy=((1 + thresholds['hot'])/2, y_level), 
               xycoords='data', ha='center', color='#e74c3c', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig