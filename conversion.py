"""
conversion.py - Conversion analysis and phone matching functionality

This module contains functions for analyzing conversion rates, phone matching,
time to conversion, and related data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_phone_matches(df):
    """
    Analyze how phone area codes affect conversion rates
    
    Args:
        df (DataFrame): DataFrame with lead data including phone numbers
    
    Returns:
        tuple: (
            DataFrame: Conversion rates by phone match status,
            DataFrame: Counts by phone match status
        )
    """
    if 'phone_number' not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    
    # Create a copy to avoid SettingWithCopyWarning
    analysis_df = df.copy()
    
    # Extract area codes from phone numbers
    analysis_df['area_code'] = analysis_df['phone_number'].apply(extract_area_code)
    
    # Extract state from area codes
    analysis_df['phone_state'] = analysis_df['area_code'].apply(map_area_code_to_state)
    
    # Check if phone state matches lead state
    analysis_df['phone_state_match'] = (analysis_df['phone_state'] == analysis_df['state']) & (~analysis_df['phone_state'].isna())
    
    # Create more descriptive labels for the match status
    analysis_df['match_status'] = analysis_df['phone_state_match'].map({
        True: 'Area Code Matches State',
        False: 'Area Code Different from State'
    })
    
    # Filter for rows with valid phone data
    valid_data = analysis_df.dropna(subset=['phone_number'])
    
    # If we have valid data, continue with the analysis
    if len(valid_data) > 0:
        # Calculate conversion rates by phone-state match status
        match_conversion = valid_data.groupby('match_status')['outcome'].agg(['mean', 'count']).reset_index()
        match_conversion.columns = ['Phone-State Match', 'Conversion Rate', 'Count']
        match_conversion['Conversion Rate'] = (match_conversion['Conversion Rate'] * 100).round(1).astype(str) + '%'
        
        # Count by phone-state match status for visualization
        match_counts = valid_data['match_status'].value_counts().reset_index()
        match_counts.columns = ['Phone-State Match', 'Count']
        
        return match_conversion, match_counts
    else:
        # Return empty frames if no valid data
        return pd.DataFrame(columns=['Phone-State Match', 'Conversion Rate', 'Count']), pd.DataFrame(columns=['Phone-State Match', 'Count'])

# Clean and normalize phone numbers
def clean_phone(phone):
    if not phone or pd.isna(phone) or phone == "#ERROR!":
        return None
    # Remove non-numeric characters
    try:
        cleaned = ''.join(filter(str.isdigit, str(phone)))
        # Keep only the last 10 digits if longer
        if len(cleaned) > 10:
            cleaned = cleaned[-10:]
        return cleaned if cleaned else None
    except:
        return None

def extract_area_code(phone):
    """
    Extract area code from phone number
    
    Args:
        phone (str): Phone number
    
    Returns:
        str: Area code or None
    """
    if pd.isna(phone) or not phone:
        return None
    
    # Clean the phone number
    clean_num = clean_phone(phone)
    if not clean_num or len(clean_num) < 10:
        return None
    
    # Get the area code (first 3 digits of a 10-digit number)
    return clean_num[:3]

def map_area_code_to_state(area_code):
    """
    Map area code to state
    
    Args:
        area_code (str): Area code
    
    Returns:
        str: State abbreviation or None
    """
    if not area_code:
        return None
    
    # This is a simplified mapping of area codes to states
    # In a real application, this would be more comprehensive
    area_code_map = {
        '201': 'NJ', '202': 'DC', '203': 'CT', '205': 'AL',
        '206': 'WA', '207': 'ME', '208': 'ID', '209': 'CA',
        '210': 'TX', '212': 'NY', '213': 'CA', '214': 'TX',
        '215': 'PA', '216': 'OH', '217': 'IL', '218': 'MN',
        '219': 'IN', '224': 'IL', '225': 'LA', '228': 'MS',
        '229': 'GA', '231': 'MI', '234': 'OH', '239': 'FL',
        '240': 'MD', '248': 'MI', '251': 'AL', '252': 'NC',
        '253': 'WA', '254': 'TX', '256': 'AL', '260': 'IN',
        '262': 'WI', '267': 'PA', '269': 'MI', '270': 'KY',
        '276': 'VA', '281': 'TX', '301': 'MD', '302': 'DE',
        '303': 'CO', '304': 'WV', '305': 'FL', '307': 'WY',
        '308': 'NE', '309': 'IL', '310': 'CA', '312': 'IL',
        '313': 'MI', '314': 'MO', '315': 'NY', '316': 'KS',
        '317': 'IN', '318': 'LA', '319': 'IA', '320': 'MN',
        '321': 'FL', '323': 'CA', '325': 'TX', '330': 'OH',
        '331': 'IL', '334': 'AL', '336': 'NC', '337': 'LA',
        '339': 'MA', '340': 'VI', '347': 'NY', '351': 'MA',
        '352': 'FL', '360': 'WA', '361': 'TX', '386': 'FL',
        '401': 'RI', '402': 'NE', '404': 'GA', '405': 'OK',
        '406': 'MT', '407': 'FL', '408': 'CA', '409': 'TX',
        '410': 'MD', '412': 'PA', '413': 'MA', '414': 'WI',
        '415': 'CA', '417': 'MO', '419': 'OH', '423': 'TN',
        '424': 'CA', '425': 'WA', '430': 'TX', '432': 'TX',
        '434': 'VA', '435': 'UT', '440': 'OH', '442': 'CA',
        '443': 'MD', '445': 'PA', '469': 'TX', '470': 'GA',
        '475': 'CT', '478': 'GA', '479': 'AR', '480': 'AZ',
        '484': 'PA', '501': 'AR', '502': 'KY', '503': 'OR',
        '504': 'LA', '505': 'NM', '507': 'MN', '508': 'MA',
        '509': 'WA', '510': 'CA', '512': 'TX', '513': 'OH',
        '515': 'IA', '516': 'NY', '517': 'MI', '518': 'NY',
        '520': 'AZ', '530': 'CA', '540': 'VA', '541': 'OR',
        '551': 'NJ', '559': 'CA', '561': 'FL', '562': 'CA',
        '563': 'IA', '567': 'OH', '570': 'PA', '571': 'VA',
        '573': 'MO', '574': 'IN', '575': 'NM', '580': 'OK',
        '585': 'NY', '586': 'MI', '601': 'MS', '602': 'AZ',
        '603': 'NH', '605': 'SD', '606': 'KY', '607': 'NY',
        '608': 'WI', '609': 'NJ', '610': 'PA', '612': 'MN',
        '614': 'OH', '615': 'TN', '616': 'MI', '617': 'MA',
        '618': 'IL', '619': 'CA', '620': 'KS', '623': 'AZ',
        '626': 'CA', '630': 'IL', '631': 'NY', '636': 'MO',
        '641': 'IA', '646': 'NY', '650': 'CA', '651': 'MN',
        '660': 'MO', '661': 'CA', '662': 'MS', '667': 'MD',
        '678': 'GA', '681': 'WV', '682': 'TX', '701': 'ND',
        '702': 'NV', '703': 'VA', '704': 'NC', '706': 'GA',
        '707': 'CA', '708': 'IL', '712': 'IA', '713': 'TX',
        '714': 'CA', '715': 'WI', '716': 'NY', '717': 'PA',
        '718': 'NY', '719': 'CO', '720': 'CO', '724': 'PA',
        '727': 'FL', '731': 'TN', '732': 'NJ', '734': 'MI',
        '740': 'OH', '743': 'NC', '747': 'CA', '754': 'FL',
        '757': 'VA', '760': 'CA', '762': 'GA', '763': 'MN',
        '765': 'IN', '769': 'MS', '770': 'GA', '772': 'FL',
        '773': 'IL', '774': 'MA', '775': 'NV', '779': 'IL',
        '781': 'MA', '785': 'KS', '786': 'FL', '801': 'UT',
        '802': 'VT', '803': 'SC', '804': 'VA', '805': 'CA',
        '806': 'TX', '808': 'HI', '810': 'MI', '812': 'IN',
        '813': 'FL', '814': 'PA', '815': 'IL', '816': 'MO',
        '817': 'TX', '818': 'CA', '828': 'NC', '830': 'TX',
        '831': 'CA', '832': 'TX', '843': 'SC', '845': 'NY',
        '847': 'IL', '848': 'NJ', '850': 'FL', '856': 'NJ',
        '857': 'MA', '858': 'CA', '859': 'KY', '860': 'CT',
        '862': 'NJ', '863': 'FL', '864': 'SC', '865': 'TN',
        '870': 'AR', '872': 'IL', '878': 'PA', '901': 'TN',
        '903': 'TX', '904': 'FL', '906': 'MI', '907': 'AK',
        '908': 'NJ', '909': 'CA', '910': 'NC', '912': 'GA',
        '913': 'KS', '914': 'NY', '915': 'TX', '916': 'CA',
        '917': 'NY', '918': 'OK', '919': 'NC', '920': 'WI',
        '925': 'CA', '928': 'AZ', '931': 'TN', '936': 'TX',
        '937': 'OH', '939': 'PR', '940': 'TX', '941': 'FL',
        '947': 'MI', '949': 'CA', '951': 'CA', '952': 'MN',
        '954': 'FL', '956': 'TX', '970': 'CO', '971': 'OR',
        '972': 'TX', '973': 'NJ', '978': 'MA', '979': 'TX',
        '980': 'NC', '985': 'LA', '989': 'MI'
    }
    
    return area_code_map.get(area_code)

def analyze_time_to_conversion(df):
    """
    Analyze the average time between lead inquiry and conversion
    
    Args:
        df (DataFrame): DataFrame with lead data including inquiry_date and won flag
        
    Returns:
        dict: Dictionary containing time to conversion statistics
        {
            'average_days': float,
            'median_days': float,
            'min_days': int,
            'max_days': int,
            'by_outcome': DataFrame,
            'by_booking_type': DataFrame,
            'histogram_data': DataFrame
        }
    """
    result = {}
    
    try:
        # Make sure required columns exist
        required_cols = ['inquiry_date', 'outcome', 'days_since_inquiry']
        
        if not all(col in df.columns for col in required_cols):
            # Try fallback to days_since_inquiry if available
            if 'days_since_inquiry' in df.columns and 'outcome' in df.columns:
                # Filter for won deals
                won_leads = df[df['outcome'] == 1]
                
                # Use days_since_inquiry directly
                result['average_days'] = won_leads['days_since_inquiry'].mean()
                result['median_days'] = won_leads['days_since_inquiry'].median()
                result['min_days'] = won_leads['days_since_inquiry'].min()
                result['max_days'] = won_leads['days_since_inquiry'].max()
                
                # Group by booking type if available
                if 'booking_type' in df.columns:
                    result['by_booking_type'] = won_leads.groupby('booking_type')['days_since_inquiry'].agg(
                        ['mean', 'median', 'count']).reset_index()
                
                # Create histogram data
                bins = [0, 1, 3, 7, 14, 30, 60, 90, float('inf')]
                labels = ['Same day', '1-3 days', '4-7 days', '8-14 days', '15-30 days', '31-60 days', '61-90 days', '90+ days']
                won_leads['time_bucket'] = pd.cut(won_leads['days_since_inquiry'], bins=bins, labels=labels)
                result['histogram_data'] = won_leads['time_bucket'].value_counts().reset_index()
                result['histogram_data'].columns = ['Time to Conversion', 'Count']
                
                # Sort properly
                result['histogram_data']['Time to Conversion'] = pd.Categorical(
                    result['histogram_data']['Time to Conversion'],
                    categories=labels,
                    ordered=True
                )
                result['histogram_data'] = result['histogram_data'].sort_values('Time to Conversion')
                
                return result
            else:
                raise ValueError("Required columns not found in dataframe")
        
        # Convert inquiry_date to datetime if it's not already
        if df['inquiry_date'].dtype != 'datetime64[ns]':
            df['inquiry_date'] = pd.to_datetime(df['inquiry_date'])
            
        # Calculate time to conversion for won deals
        won_leads = df[df['outcome'] == 1].copy()
        
        if len(won_leads) == 0:
            raise ValueError("No won leads found in the data")
        
        # If event_date exists, use it to calculate days between inquiry and event
        if 'event_date' in df.columns:
            # Convert event_date to datetime if it's not already
            if won_leads['event_date'].dtype != 'datetime64[ns]':
                won_leads['event_date'] = pd.to_datetime(won_leads['event_date'])
                
            # Calculate days between inquiry and event
            won_leads['days_to_conversion'] = (won_leads['event_date'] - won_leads['inquiry_date']).dt.days
        else:
            # Use days_since_inquiry as fallback
            won_leads['days_to_conversion'] = won_leads['days_since_inquiry']
        
        # Basic statistics
        result['average_days'] = won_leads['days_to_conversion'].mean()
        result['median_days'] = won_leads['days_to_conversion'].median()
        result['min_days'] = won_leads['days_to_conversion'].min()
        result['max_days'] = won_leads['days_to_conversion'].max()
        
        # By outcome (won vs lost)
        df_with_days = df.copy()
        if 'event_date' in df.columns:
            if df_with_days['event_date'].dtype != 'datetime64[ns]':
                df_with_days['event_date'] = pd.to_datetime(df_with_days['event_date'])
            df_with_days['days_to_conversion'] = (df_with_days['event_date'] - df_with_days['inquiry_date']).dt.days
        else:
            df_with_days['days_to_conversion'] = df_with_days['days_since_inquiry']
            
        result['by_outcome'] = df_with_days.groupby('outcome')['days_to_conversion'].agg(
            ['mean', 'median', 'count']).reset_index()
        result['by_outcome']['outcome'] = result['by_outcome']['outcome'].map({1: 'Won', 0: 'Lost'})
        
        # By booking type if available
        if 'booking_type' in won_leads.columns:
            result['by_booking_type'] = won_leads.groupby('booking_type')['days_to_conversion'].agg(
                ['mean', 'median', 'count']).reset_index()
        
        # Create histogram data
        bins = [0, 1, 3, 7, 14, 30, 60, 90, float('inf')]
        labels = ['Same day', '1-3 days', '4-7 days', '8-14 days', '15-30 days', '31-60 days', '61-90 days', '90+ days']
        won_leads['time_bucket'] = pd.cut(won_leads['days_to_conversion'], bins=bins, labels=labels)
        result['histogram_data'] = won_leads['time_bucket'].value_counts().reset_index()
        result['histogram_data'].columns = ['Time to Conversion', 'Count']
        
        # Sort properly
        result['histogram_data']['Time to Conversion'] = pd.Categorical(
            result['histogram_data']['Time to Conversion'],
            categories=labels,
            ordered=True
        )
        result['histogram_data'] = result['histogram_data'].sort_values('Time to Conversion')
        
        return result
    
    except Exception as e:
        # Return error in the result
        return {
            'error': str(e),
            'average_days': None,
            'median_days': None,
            'by_booking_type': pd.DataFrame(),
            'histogram_data': pd.DataFrame()
        }

def analyze_prediction_counts(y_scores, thresholds):
    """
    Analyze the distribution of predictions across categories
    
    Args:
        y_scores (array-like): Predicted scores
        thresholds (dict): Dictionary of thresholds for each category
    
    Returns:
        DataFrame: Counts by prediction category
    """
    if y_scores is None or thresholds is None:
        return pd.DataFrame()
    
    # Count predictions by category
    hot_count = (y_scores >= thresholds['hot']).sum()
    warm_count = ((y_scores >= thresholds['warm']) & (y_scores < thresholds['hot'])).sum()
    cool_count = ((y_scores >= thresholds['cool']) & (y_scores < thresholds['warm'])).sum()
    cold_count = (y_scores < thresholds['cool']).sum()
    
    # Create DataFrame
    counts_df = pd.DataFrame({
        'Category': ['Hot', 'Warm', 'Cool', 'Cold'],
        'Count': [hot_count, warm_count, cool_count, cold_count]
    })
    
    return counts_df