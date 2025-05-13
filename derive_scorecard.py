import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import database as db
import sqlite3
import streamlit as st

def generate_lead_scorecard(use_sample_data=True):
    """
    Generate a lead scorecard based on historical data.
    
    Args:
        use_sample_data (bool): Whether to use sample data or database data
    
    Returns:
        tuple: (weights DataFrame, suggested thresholds dict)
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

        # 5) Standardize & fit logistic regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, solver='liblinear')
        model.fit(X_scaled, y)

        # 6) Extract & normalize coefficients
        coefs = pd.Series(model.coef_[0], index=available_features)
        # Absolute importance
        imp = coefs.abs().sort_values(ascending=False)
        # Scale to a 0–10 point system proportional to relative size
        max_points = 10
        weights = (imp / imp.max() * max_points).round().astype(int)
        
        # Create DataFrame for display
        result_df = pd.DataFrame({
            'Feature': weights.index,
            'Coefficient': [coefs[feat] for feat in weights.index],
            'Points': weights.values
        })
        
        # 7) Suggested threshold buckets
        thresholds = {
            'Hot': int(weights.sum() * 0.6),
            'Warm': int(weights.sum() * 0.4),
            'Cool': int(weights.sum() * 0.2),
            'Cold': 0
        }
        
        return result_df, thresholds
        
    except Exception as e:
        print(f"Error generating scorecard: {str(e)}")
        return None, None

def score_lead(lead_data, scorecard):
    """
    Score a lead based on the generated scorecard
    
    Args:
        lead_data (dict): Dictionary of lead data
        scorecard (DataFrame): Scorecard with features and points
        
    Returns:
        tuple: (score, category)
    """
    if scorecard is None:
        return 0, "Unknown"
        
    total_score = 0
    feature_contributions = {}
    
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
                    # Faster follow-up is better (if coefficient is negative)
                    # Cap at 30 days
                    normalized_value = min(float(feature_value) / 30.0, 1.0)
                    contribution = points * coef_sign * normalized_value
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