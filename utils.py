import pandas as pd
import numpy as np

def process_data(leads_df, operations_df=None):
    """
    Process and prepare the data for analysis.
    
    Args:
        leads_df (DataFrame): The leads data from Streak export
        operations_df (DataFrame, optional): The operations data from Streak export
    
    Returns:
        DataFrame: Processed dataframe with outcome and binned columns
    """
    # Clean status & define outcome
    leads_df['Status'] = leads_df['Status'].astype(str).str.strip().str.lower()
    leads_df['Won'] = leads_df['Status'].isin(['definite', 'definte'])
    leads_df['Lost'] = leads_df['Status'] == 'lost'
    
    # Filter to only definitive outcomes
    df = leads_df[leads_df['Status'].isin(['definite', 'definte', 'lost'])].copy()
    df['Outcome'] = df['Won'].astype(int)
    
    # Convert numeric fields
    for col in ['Number Of Guests', 'Days Until Event', 'Days Since Inquiry', 'Bartenders Needed']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Define bins for guests
    if 'Number Of Guests' in df.columns:
        bins_guests = [0, 50, 100, 200, np.inf]
        labels_guests = ['0–50', '51–100', '101–200', '200+']
        df['Guests Bin'] = pd.cut(df['Number Of Guests'], bins=bins_guests, labels=labels_guests)
    
    # Define bins for days until event
    if 'Days Until Event' in df.columns:
        bins_days = [0, 7, 30, 90, np.inf]
        labels_days = ['0–7 days', '8–30 days', '31–90 days', '91+ days']
        df['DaysUntilBin'] = pd.cut(df['Days Until Event'], bins=bins_days, labels=labels_days)
    
    # Merge in operations data if available
    if operations_df is not None:
        try:
            # Check if 'Box Key' and 'Actual Deal Value' columns exist
            if 'Box Key' in operations_df.columns and 'Actual Deal Value' in operations_df.columns:
                # Merge on Box Key
                df = pd.merge(df, operations_df[['Box Key', 'Actual Deal Value']], 
                            on='Box Key', how='left')
                
                # Convert to numeric
                df['Actual Deal Value'] = pd.to_numeric(df['Actual Deal Value'], errors='coerce')
        except Exception as e:
            print(f"Error merging operations data: {str(e)}")
    
    return df

def calculate_conversion_rates(df):
    """
    Calculate conversion rates by different categories.
    
    Args:
        df (DataFrame): Processed dataframe with outcome
    
    Returns:
        dict: Dictionary of dataframes with conversion rates by category
    """
    conversion_rates = {}
    
    # Event Type conversion rates
    if 'Event Type' in df.columns:
        conv_event_type = df.groupby('Event Type')['Outcome'].mean().reset_index(name='Conversion Rate')
        if not conv_event_type.empty:
            conversion_rates['Event Type'] = conv_event_type
    
    # Referral Source conversion rates
    if 'Referral Source' in df.columns:
        conv_referral = df.groupby('Referral Source')['Outcome'].mean().reset_index(name='Conversion Rate')
        if not conv_referral.empty:
            conversion_rates['Referral Source'] = conv_referral
    
    # Marketing Source conversion rates
    if 'Marketing Source' in df.columns:
        conv_marketing = df.groupby('Marketing Source')['Outcome'].mean().reset_index(name='Conversion Rate')
        if not conv_marketing.empty:
            conversion_rates['Marketing Source'] = conv_marketing
    
    # State conversion rates
    if 'State' in df.columns:
        conv_state = df.groupby('State')['Outcome'].mean().reset_index(name='Conversion Rate')
        if not conv_state.empty:
            conversion_rates['State'] = conv_state
    
    # Guests Bin conversion rates
    if 'Guests Bin' in df.columns:
        conv_guests = df.groupby('Guests Bin')['Outcome'].mean().reset_index(name='Conversion Rate')
        if not conv_guests.empty:
            conversion_rates['Guests Bin'] = conv_guests
    
    # Days Until Event Bin conversion rates
    if 'DaysUntilBin' in df.columns:
        conv_days = df.groupby('DaysUntilBin')['Outcome'].mean().reset_index(name='Conversion Rate')
        if not conv_days.empty:
            conversion_rates['DaysUntilBin'] = conv_days
    
    return conversion_rates

def calculate_correlations(df):
    """
    Calculate correlations between numeric features and outcome.
    
    Args:
        df (DataFrame): Processed dataframe with outcome
    
    Returns:
        DataFrame: Dataframe with correlations
    """
    # Identify numeric features
    numeric_features = [col for col in ['Days Since Inquiry', 'Days Until Event', 
                                        'Number Of Guests', 'Bartenders Needed']
                        if col in df.columns]
    
    if numeric_features and 'Outcome' in df.columns:
        # Calculate correlations
        try:
            corr_outcome = df[numeric_features + ['Outcome']].corr()['Outcome'].abs()
            corr_outcome = corr_outcome.sort_values(ascending=False).reset_index(name='Correlation with Outcome')
            return corr_outcome
        except Exception as e:
            print(f"Error calculating correlations: {str(e)}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()
