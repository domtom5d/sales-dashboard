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
        # Normalize status field
        if 'Status' in leads_df.columns:
            leads_df['Status'] = leads_df['Status'].astype(str).str.lower().str.strip()
            leads_df = leads_df[leads_df['Status'].isin(['definite', 'definte', 'lost', 'win', 'won'])]
            leads_df['Outcome'] = leads_df['Status'].isin(['definite', 'definte', 'win', 'won']).astype(int)
        else:
            # Use Won/Lost fields if Status isn't available
            if 'Won' in leads_df.columns and 'Lost' in leads_df.columns:
                leads_df['Outcome'] = leads_df['Won'].astype(int)
            else:
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
        # Handle various column name formats
        price_col = next((col for col in df.columns if 'actual_deal_value' in col.lower()), None)
        guests_col = next((col for col in df.columns if 'number of guests' in col.lower() or 'number_of_guests' in col.lower()), None)
        days_until_col = next((col for col in df.columns if 'days until event' in col.lower() or 'days_until_event' in col.lower()), None)
        days_since_col = next((col for col in df.columns if 'days since inquiry' in col.lower() or 'days_since_inquiry' in col.lower()), None)
        bartenders_col = next((col for col in df.columns if 'bartenders needed' in col.lower() or 'bartenders_needed' in col.lower()), None)
        event_type_col = next((col for col in df.columns if 'event type' in col.lower() or 'event_type' in col.lower() or 'booking_type' in col.lower()), None)
        referral_col = next((col for col in df.columns if 'referral source' in col.lower() or 'referral_source' in col.lower() or 'marketing_source' in col.lower()), None)
        state_col = next((col for col in df.columns if 'state' in col.lower()), None)
        phone_col = next((col for col in df.columns if 'phone' in col.lower()), None)

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
            
        # Referral tier
        tier_map = {'referral': 3, 'friend': 3, 'facebook': 2, 'google': 1, 'instagram': 2, 'social': 2}
        if referral_col:
            df['ReferralTier'] = df[referral_col].astype(str).str.lower().map(tier_map).fillna(1)
        else:
            df['ReferralTier'] = 1
            
        # Phone match with area code
        state_code_map = {
            'California': '415', 'New York': '212', 'Texas': '214', 'Florida': '305', 
            'Illinois': '312', 'Massachusetts': '617', 'Colorado': '303'
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
    
    for _, row in scorecard.iterrows():
        feature = row['Feature']
        points = row['Points']
        coef_sign = 1 if row['Coefficient'] > 0 else -1
        
        # Check if feature exists in lead data
        if feature in lead_data:
            # Apply points if the feature has a value
            if lead_data[feature]:
                total_score += points * coef_sign
                
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