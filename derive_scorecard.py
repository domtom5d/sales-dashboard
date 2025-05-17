import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import cross_val_score
import database as db
import sqlite3
import streamlit as st
import matplotlib.pyplot as plt

def generate_lead_scorecard(use_sample_data=True):
    """
    Generate a lead scorecard based on historical data.
    
    Args:
        use_sample_data (bool): Whether to use sample data or database data
    
    Returns:
        tuple: (weights DataFrame, suggested thresholds dict, model_metrics dict)
        
        The model_metrics dict contains these keys:
        - roc_auc: Area under the ROC curve
        - pr_auc: Area under the Precision-Recall curve
        - fpr: False positive rates for ROC curve
        - tpr: True positive rates for ROC curve
        - precision: Precision values for PR curve
        - recall: Recall values for PR curve
        - best_threshold: Optimal probability threshold (Youden's J)
        - confusion_matrix: Confusion matrix at optimal threshold
        - y_pred_proba: Predicted probabilities for all samples
        - y_true: True labels
        - won_scores: Score distribution for won leads
        - lost_scores: Score distribution for lost leads
    """
    try:
        # 1) Load your exports
        if use_sample_data:
            leads_df = db.get_lead_data()
            ops_df = db.get_operation_data()
            
            if leads_df is None or ops_df is None:
                return None, None
                
        else:
            # For uploaded data, we'd use the database tables
            leads_df = db.get_lead_data()
            ops_df = db.get_operation_data()
            
            if leads_df is None or leads_df.empty:
                return None, None

        # 2) Merge and clean
        # Make sure we have an Outcome field for the ML model to use
        if 'outcome' in leads_df.columns:
            # Use existing outcome field if it exists (from database processing)
            leads_df['Outcome'] = leads_df['outcome']
        elif 'Status' in leads_df.columns:
            # Normalize status field
            leads_df['Status'] = leads_df['Status'].astype(str).str.lower().str.strip()
            # Use your correct status mapping (definite and tentative as wins)
            win_statuses = ['definite', 'definte', 'tentative', 'win', 'won']
            lost_statuses = ['lost']
            # Keep only rows with definitive outcomes
            leads_df = leads_df[leads_df['Status'].isin(win_statuses + lost_statuses)]
            leads_df['Outcome'] = leads_df['Status'].isin(win_statuses).astype(int)
        elif 'Lead Trigger' in leads_df.columns or any('lead trigger' in col.lower() for col in leads_df.columns):
            # Use Lead Trigger if Status isn't available
            trigger_col = 'Lead Trigger' if 'Lead Trigger' in leads_df.columns else next(col for col in leads_df.columns if 'lead trigger' in col.lower())
            leads_df['Lead Trigger'] = leads_df[trigger_col].astype(str).str.lower().str.strip()
            # Map Lead Trigger values to outcomes
            win_triggers = ['hot', 'warm', 'super lead']
            lost_triggers = ['cold', 'cool']
            # Keep only rows with Lead Trigger values that have clear mapping
            leads_df = leads_df[leads_df['Lead Trigger'].isin(win_triggers + lost_triggers)]
            leads_df['Outcome'] = leads_df['Lead Trigger'].isin(win_triggers).astype(int)
        elif 'Won' in leads_df.columns and 'Lost' in leads_df.columns:
            # Use already processed Won/Lost fields if they exist
            leads_df['Outcome'] = leads_df['Won'].astype(int)
        else:
            # If we can't find any relevant fields, we can't build a model
            print("No usable Status, Lead Trigger, or Won/Lost fields found in data")
            return None, None
        
        # Log info about our training data
        win_count = leads_df['Outcome'].sum()
        loss_count = len(leads_df) - win_count
        print(f"Training data: {win_count} wins, {loss_count} losses, {len(leads_df)} total")
        
        # Need minimum samples for training
        if win_count < 10 or loss_count < 10:
            print("Insufficient training data: need at least 10 examples of each outcome")
            return None, None

        # Merge with operations data if it exists
        if ops_df is not None and not ops_df.empty:
            try:
                df = pd.merge(
                    leads_df,
                    ops_df[['box_key', 'actual_deal_value']],
                    left_on='box_key', right_on='box_key', how='left'
                )
            except:
                # Fall back to using just leads data
                df = leads_df
                df['actual_deal_value'] = np.nan
        else:
            df = leads_df
            df['actual_deal_value'] = np.nan

        df['actual_deal_value'] = pd.to_numeric(df['actual_deal_value'], errors='coerce')

        # 3) Feature engineering
        # Handle various column name formats - add all possible names for robustness
        price_col = next((col for col in df.columns if any(name in col.lower() for name in ['actual_deal_value', 'actual deal value', 'price'])), None)
        guests_col = next((col for col in df.columns if any(name in col.lower() for name in ['number of guests', 'number_of_guests', 'guests'])), None)
        days_until_col = next((col for col in df.columns if any(name in col.lower() for name in ['days until event', 'days_until_event'])), None)
        days_since_col = next((col for col in df.columns if any(name in col.lower() for name in ['days since inquiry', 'days_since_inquiry'])), None)
        bartenders_col = next((col for col in df.columns if any(name in col.lower() for name in ['bartenders needed', 'bartenders_needed', 'bartenders'])), None)
        event_type_col = next((col for col in df.columns if any(name in col.lower() for name in ['event type', 'event_type', 'booking_type', 'booking type'])), None)
        referral_col = next((col for col in df.columns if any(name in col.lower() for name in ['referral source', 'referral_source', 'marketing_source', 'marketing source'])), None)
        state_col = next((col for col in df.columns if 'state' in col.lower()), None)
        phone_col = next((col for col in df.columns if 'phone' in col.lower()), None)
        
        print(f"Found columns: price={price_col}, guests={guests_col}, days_until={days_until_col}, bartenders={bartenders_col}")

        # Price per guest
        if price_col and guests_col:
            df['PricePerGuest'] = pd.to_numeric(df[price_col], errors='coerce') / pd.to_numeric(df[guests_col], errors='coerce').replace(0, np.nan)
        else:
            df['PricePerGuest'] = np.nan
            
        # Days until event
        if days_until_col:
            df['DaysUntilEvent'] = pd.to_numeric(df[days_until_col], errors='coerce')
        else:
            df['DaysUntilEvent'] = np.nan
            
        # Days since inquiry
        if days_since_col:
            df['DaysSinceInquiry'] = pd.to_numeric(df[days_since_col], errors='coerce')
        else:
            df['DaysSinceInquiry'] = np.nan
            
        # Number of guests
        if guests_col:
            df['NumberOfGuests'] = pd.to_numeric(df[guests_col], errors='coerce')
        else:
            df['NumberOfGuests'] = np.nan
            
        # Bartenders needed
        if bartenders_col:
            df['BartendersNeeded'] = pd.to_numeric(df[bartenders_col], errors='coerce')
        else:
            df['BartendersNeeded'] = np.nan
            
        # Is corporate event
        if event_type_col:
            df['IsCorporate'] = df[event_type_col].astype(str).str.lower().str.contains('corporate').astype(int)
        else:
            df['IsCorporate'] = 0
            
        # Referral tier - expanded with more social sources
        tier_map = {
            'referral': 3, 'friend': 3, 'facebook': 2, 'google': 1, 
            'instagram': 2, 'social': 2, 'linkedin': 2, 'twitter': 2, 
            'tiktok': 2, 'pinterest': 2, 'yelp': 1, 'thumbtack': 1
        }
        if referral_col:
            df['ReferralTier'] = df[referral_col].astype(str).str.lower().map(tier_map).fillna(1)
        else:
            df['ReferralTier'] = 1
            
        # Phone match with area code - expanded with more states
        state_code_map = {
            'California': '415', 'New York': '212', 'Texas': '214', 'Florida': '305', 
            'Illinois': '312', 'Massachusetts': '617', 'Colorado': '303', 'Pennsylvania': '215',
            'Ohio': '216', 'Michigan': '313', 'Georgia': '404', 'Washington': '206',
            'Arizona': '602', 'Nevada': '702', 'Oregon': '503', 'New Jersey': '201',
            'Oklahoma': '405', 'New Mexico': '505'
        }
        
        if phone_col and state_col:
            df['PhoneArea'] = df[phone_col].astype(str).str.replace(r'\D', '', regex=True).str[:3]
            df['PhoneMatch'] = df.apply(
                lambda r: int(r['PhoneArea'] == state_code_map.get(r[state_col], '')), 
                axis=1
            )
        else:
            df['PhoneMatch'] = 0

        # Drop rows without outcomes
        df = df.dropna(subset=['Outcome'])

        # 4) Prepare X/y
        features = [
            'PricePerGuest', 'DaysUntilEvent', 'DaysSinceInquiry',
            'NumberOfGuests', 'BartendersNeeded', 'IsCorporate',
            'ReferralTier', 'PhoneMatch'
        ]
        
        # Add richer features
        # Calculate price per guest if we have the data
        if all(col in df.columns for col in ['actual_deal_value', 'number_of_guests']):
            df['PricePerGuest'] = df['actual_deal_value'] / df['number_of_guests'].replace(0, 1)
        
        # Add seasonality features
        if 'event_date' in df.columns:
            df['Month'] = pd.to_datetime(df['event_date']).dt.month
            df['DayOfWeek'] = pd.to_datetime(df['event_date']).dt.dayofweek
            # Add month and day of week to features
            features.extend(['Month', 'DayOfWeek'])
            
        # Add urgency factor if we have both dates
        if all(col in df.columns for col in ['inquiry_date', 'event_date']):
            df['InquiryToEventDays'] = (pd.to_datetime(df['event_date']) - pd.to_datetime(df['inquiry_date'])).dt.days
            # Replace negative values with 0
            df['InquiryToEventDays'] = df['InquiryToEventDays'].clip(lower=0)
            features.append('InquiryToEventDays')
            
        # Add geographic features if available
        if 'state' in df.columns:
            # Convert state to categorical to capture regional differences
            df['Region'] = df['state'].astype('category').cat.codes
            features.append('Region')
        
        # Select only features that have data
        available_features = []
        for feature in features:
            if feature in df.columns and not df[feature].isna().all():
                available_features.append(feature)
        
        if len(available_features) < 3:
            # Not enough features to build a model
            return None, None
            
        X = df[available_features].fillna(0)
        y = df['Outcome']

        # 5) Prepare data and train both models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train a Random Forest model (better performance)
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
        
        # Use cross-validation to evaluate model robustness
        cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='roc_auc')
        print(f"Cross-validation ROC AUC scores: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Fit the model on all training data
        rf_model.fit(X_scaled, y)
        
        # Also train a Logistic Regression for interpretability
        lr_model = LogisticRegression(max_iter=1000, solver='liblinear')
        lr_model.fit(X_scaled, y)
        
        # Use Random Forest for predictions (better performance)
        model = rf_model
        
        # But use Logistic Regression for feature importance (more interpretable)
        interpretation_model = lr_model

        # 6) Extract feature importance from both models
        # For Random Forest, use built-in feature_importances_
        rf_importance = pd.Series(model.feature_importances_, index=available_features)
        
        # For Logistic Regression, use coefficients (more interpretable direction)
        coefs = pd.Series(interpretation_model.coef_[0], index=available_features)
        
        # Create a hybrid importance metric that uses magnitude from RF but direction from LR
        importance = rf_importance.copy()
        
        # Apply direction from logistic regression
        for feat in available_features:
            if coefs[feat] < 0:
                importance[feat] = -importance[feat]
                
        # Sort by absolute importance
        imp = importance.abs().sort_values(ascending=False)
        
        # Scale to a 0–10 point system proportional to relative size
        max_points = 10
        weights = (imp / imp.max() * max_points).round().astype(int)
        
        # For display, use the direction from logistic regression
        feature_direction = {feat: "+" if coefs[feat] > 0 else "-" for feat in available_features}
        
        # Create DataFrame for display
        result_df = pd.DataFrame({
            'Feature': weights.index,
            'Direction': [feature_direction[feat] for feat in weights.index],
            'Coefficient': [coefs[feat] for feat in weights.index],
            'RandomForest': [rf_importance[feat] for feat in weights.index],
            'Points': weights.values
        })
        
        # 7) Calculate model performance metrics
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of positive class
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds_roc = roc_curve(y, y_pred_proba)
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        # Compute precision-recall curve
        precision, recall, thresholds_pr = precision_recall_curve(y, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Compute optimal threshold using F1 score maximization (instead of Youden's J)
        # First calculate F1 scores for different thresholds
        f1_scores = []
        for threshold in thresholds_roc:
            if threshold > 0 and threshold < 1:  # Skip edge cases
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y, y_pred, zero_division=0)
                f1_scores.append((threshold, f1))
        
        # Find threshold that maximizes F1 score
        if f1_scores:
            f1_df = pd.DataFrame(f1_scores, columns=['threshold', 'f1_score'])
            best_f1_idx = f1_df['f1_score'].argmax()
            best_f1_threshold = f1_df.iloc[best_f1_idx]['threshold']
            max_f1_score = f1_df.iloc[best_f1_idx]['f1_score']
            print(f"Best F1 Score: {max_f1_score:.3f} at threshold {best_f1_threshold:.3f}")
        else:
            best_f1_threshold = 0.5  # Default if we can't calculate
            
        # Also compute Youden's J statistic (sensitivity + specificity - 1) as an alternative
        j_scores = tpr + (1 - fpr) - 1
        j_best_idx = np.argmax(j_scores)
        j_best_threshold = thresholds_roc[j_best_idx]
        
        # Use F1 maximizing threshold (better for imbalanced classes) instead of Youden's J
        best_idx = j_best_idx  # Keep for reference
        best_threshold = best_f1_threshold if 'best_f1_threshold' in locals() else j_best_threshold
        
        # Get confusion matrix at the optimal threshold
        optimal_preds = (y_pred_proba >= best_threshold).astype(int)
        cm = confusion_matrix(y, optimal_preds)
        
        # Get score distributions
        won_scores = y_pred_proba[y == 1]
        lost_scores = y_pred_proba[y == 0]
        
        # Store model evaluation metrics
        model_metrics = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'thresholds_roc': thresholds_roc,
            'best_threshold': best_threshold,
            'f1_threshold': best_f1_threshold if 'best_f1_threshold' in locals() else best_threshold,
            'j_threshold': j_best_threshold,
            'confusion_matrix': cm,
            'y_pred_proba': y_pred_proba,
            'y_true': y,
            'won_scores': won_scores,
            'lost_scores': lost_scores,
            'best_idx': best_idx,
            'j_scores': j_scores
        }
        
        # Add f1 scores if calculated
        if 'f1_df' in locals() and 'max_f1_score' in locals():
            model_metrics['f1_scores'] = f1_df
            model_metrics['max_f1_score'] = max_f1_score
        
        # 8) Set data-driven thresholds
        # Convert model probabilities to point scale for easier interpretation
        total_pts = weights.sum()
        prob_to_points = lambda p: int(p * total_pts)
        
        # Find second threshold at 1/3 of the way between best threshold and 0
        second_threshold = best_threshold / 2
        third_threshold = best_threshold / 4
        
        # Define thresholds using data-driven approach
        thresholds = {
            'Hot': prob_to_points(best_threshold),           # Optimal threshold from ROC curve
            'Warm': prob_to_points(second_threshold),        # Halfway point
            'Cool': prob_to_points(third_threshold),         # Quarter point
            'Cold': 0
        }
        
        print(f"Data-driven thresholds: Hot: {thresholds['Hot']}, Warm: {thresholds['Warm']}, Cool: {thresholds['Cool']}")
        print(f"ROC AUC: {roc_auc:.3f}, Best threshold: {best_threshold:.3f}")
        
        return result_df, thresholds, model_metrics
        
    except Exception as e:
        print(f"Error generating scorecard: {str(e)}")
        return None, None, None

def calculate_category_half_life(df, category_col='event_type'):
    """
    Calculate the average days-to-close for each event category
    
    Args:
        df (DataFrame): DataFrame with lead data
        category_col (str): Column name for event category
        
    Returns:
        pd.Series: Series with half-life days for each category
    """
    if df is None or df.empty or category_col not in df.columns:
        # Return default half-life if we can't calculate
        return pd.Series({'default': 30.0})
        
    # Ensure we have the necessary columns
    required_cols = ['outcome', 'days_since_inquiry']
    if not all(col in df.columns for col in required_cols):
        return pd.Series({'default': 30.0})
    
    # Filter to only closed leads (won or lost)
    closed_leads = df[df['outcome'].isin([0, 1])].copy()
    
    if closed_leads.empty:
        return pd.Series({'default': 30.0})
    
    # Calculate days-to-close for each category
    half_lives = closed_leads.groupby(category_col)['days_since_inquiry'].mean()
    
    # Rename to 'half_life_days'
    half_lives = half_lives.rename('half_life_days')
    
    # Ensure all values are positive and reasonable
    half_lives = half_lives.clip(lower=7.0, upper=90.0)
    
    # Calculate the global average as fallback
    global_half_life = half_lives.mean()
    
    # Add a default entry
    half_lives['default'] = global_half_life
    
    # Fill any NaN values with the global average
    half_lives = half_lives.fillna(global_half_life)
    
    return half_lives

def score_lead(lead_data, scorecard, category_half_lives=None):
    """
    Score a lead based on the generated scorecard with category-specific half-life
    
    Args:
        lead_data (dict): Dictionary of lead data
        scorecard (DataFrame): Scorecard with features and points
        category_half_lives (dict or Series, optional): Half-life values by category
        
    Returns:
        tuple: (score, category)
    """
    if scorecard is None:
        return 0, "Unknown"
        
    total_score = 0
    feature_contributions = {}
    
    # Get the lead's event category
    event_category = None
    for category_field in ['event_type', 'booking_type', 'clean_booking_type']:
        if category_field in lead_data and lead_data[category_field]:
            event_category = str(lead_data[category_field])
            break
    
    # Determine the appropriate half-life to use
    half_life_days = 30.0  # Default
    if category_half_lives is not None:
        if event_category and event_category in category_half_lives:
            half_life_days = category_half_lives[event_category]
        elif 'default' in category_half_lives:
            half_life_days = category_half_lives['default']
    
    for _, row in scorecard.iterrows():
        feature = row['Feature']
        points = row['Points']
        coef_sign = 1 if row['Coefficient'] > 0 else -1
        
        # Check if feature exists in lead data
        if feature in lead_data:
            feature_value = lead_data[feature]
            
            # For numeric features, we need to apply appropriate scaling
            if feature in ['NumberOfGuests', 'DaysUntilEvent', 'DaysSinceInquiry', 'PricePerGuest', 'BartendersNeeded']:
                # Apply points proportional to the value for numeric fields
                # But normalize first using domain knowledge
                
                if feature == 'NumberOfGuests' and feature_value:
                    # More guests is usually better (up to a reasonable limit)
                    normalized_value = min(float(feature_value) / 100.0, 1.0)  # Normalize guests with cap at 100
                    contribution = points * coef_sign * normalized_value
                    total_score += contribution
                    feature_contributions[feature] = contribution
                    
                elif feature == 'DaysUntilEvent' and feature_value:
                    # Closer events might be more likely to close (if coefficient is negative)
                    # For positive coefficient, farther events are better
                    # Cap at 365 days (1 year)
                    normalized_value = min(float(feature_value) / 365.0, 1.0)
                    contribution = points * coef_sign * normalized_value
                    total_score += contribution
                    feature_contributions[feature] = contribution
                    
                elif feature == 'DaysSinceInquiry' and feature_value:
                    # Use category-specific half-life for decay calculation
                    # Convert days to a decay factor using exponential decay model
                    # decay = 2^(-days/half_life)
                    days = float(feature_value)
                    decay_factor = 2 ** (-days / half_life_days)
                    
                    # For scoring, we invert this if coefficient is negative
                    # (older leads are usually worse, so higher days = lower score)
                    if coef_sign < 0:
                        # For negative coefficient, invert the decay (1-decay)
                        # So newer leads (low days) get high scores
                        normalized_value = 1.0 - decay_factor
                    else:
                        # For positive coefficient, use decay directly
                        normalized_value = decay_factor
                    
                    contribution = points * abs(coef_sign) * normalized_value
                    total_score += contribution
                    feature_contributions[feature] = contribution
                    
                elif feature == 'PricePerGuest' and feature_value:
                    # Higher price per guest is usually better (if coefficient is positive)
                    # Cap at $100 per guest
                    normalized_value = min(float(feature_value) / 100.0, 1.0)
                    contribution = points * coef_sign * normalized_value
                    total_score += contribution
                    feature_contributions[feature] = contribution
                    
                elif feature == 'BartendersNeeded' and feature_value:
                    # More bartenders usually means more revenue (if coefficient is positive)
                    # Cap at 10 bartenders
                    normalized_value = min(float(feature_value) / 10.0, 1.0)
                    contribution = points * coef_sign * normalized_value
                    total_score += contribution
                    feature_contributions[feature] = contribution
                
            # For boolean/categorical features, apply full points if present
            elif feature in ['IsCorporate', 'PhoneMatch', 'ReferralTier']:
                if feature_value:
                    # For ReferralTier, scale by the tier value (1-3)
                    if feature == 'ReferralTier':
                        tier_value = min(float(feature_value) / 3.0, 1.0)
                        contribution = points * coef_sign * tier_value
                    else:
                        # For boolean features, apply full points
                        contribution = points * coef_sign if feature_value else 0
                        
                    total_score += contribution
                    feature_contributions[feature] = contribution
            
            # For any other features, use simple presence check
            elif feature_value:
                contribution = points * coef_sign
                total_score += contribution
                feature_contributions[feature] = contribution
                
    # Print info for debugging
    print(f"Lead scored {total_score} points")
    for feature, contribution in feature_contributions.items():
        print(f"  - {feature}: {contribution:.1f} points")
                
    return total_score

if __name__ == "__main__":
    # When run as a script, generate and print the scorecard
    scorecard, thresholds = generate_lead_scorecard()
    
    if scorecard is not None:
        print("\n===== LEAD SCORING MODEL =====\n")
        print(scorecard)
        print("\nSuggested Thresholds:")
        for category, threshold in thresholds.items():
            print(f" Score ≥ {threshold} → {category}")
    else:
        print("Unable to generate scorecard. Check your data.")